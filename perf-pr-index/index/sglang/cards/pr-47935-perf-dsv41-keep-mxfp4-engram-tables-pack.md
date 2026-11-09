# PR #47935: [Perf] dsv41: Keep MXFP4 Engram tables packed and decode E2M1 in the gather

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @weireweire |
| 合并 | 2026-11-04 |
| 分类 | Speculative Decoding(major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/47935 |

## 一句话结论

DeepSeek-V4.1 的 Engram 表保持 checkpoint 原生 MXFP4 打包、在 gather kernel 内解码 E2M1,host 内存减半,host gather 大批量最多快 23%。

## 动机

`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4` checkpoint 把两张 Engram 表存为 packed MXFP4(`uint8 [rows, 128]`,每字节两个 E2M1,偶数列在低 nibble,per-32 ue8m0 scale 不变),而 `EngramEmbedding` 分配的是 e4m3 `[rows, 256]` 表,加载即报 shape mismatch。若在加载时把行展开成 e4m3,显存/内存规模回到 FP8 checkpoint 量级:每张表每个 TP4 rank 约 25 GB,`placement: cpu` 时每节点约 190 GiB host RAM。保持打包、把解码移进 gather(vllm #61530 的做法)可将这部分内存减半。

## 优化点

`quantization_config.engram_embed_quant == "mxfp4"` 时(缺省视为 fp8,其他值报错),`EngramEmbedding` 在所有布局(device 表、per-rank 与共享 host 表、peer shards)按 `uint8 [rows, dim//2]` 分配,`_load_rows` 原样拷贝打包行。Triton gather kernel 新增 `PACKED_FP4` 分支:读字节、按奇偶列取 nibble、从 E2M1 位模式直接构造 fp32(magnitude code m≥2 时 bits 为 (m+252)<<22,0/0.5 有特例),再做不变的 e8m0 scale 乘;符号位直接写进 fp32 位而非用一元负号(避免 Triton <3.8 把 `-x` 降为 `0 - x`,把 -0 变 +0)。每个 E2M1 值在 e4m3 中都可精确表示,因此输出与"展开表 gather"bit-identical(含 -0)。`engram_gather`/`engram_gather_peer`、staged dequant、peer 与 torch fallback 全链路传 mode;未声明 `engram_embed_quant` 的 packed checkpoint 在加载时报错并点名该 key,而不是以双倍内存静默加载。

## 关键改动

```text
python/sglang/kernels/ops/embeddings/engram_gather.py: 新增 _dequant_rows 的 MXFP4 分支与 unpack_mxfp4_rows torch 对应实现
python/sglang/srt/layers/engram.py: EngramEmbedding/_PeerShards/_HostTable 按 mxfp4 分配 uint8 半宽表,_engram_table_is_mxfp4 解析 checkpoint
python/sglang/srt/layers/engram_staging.py: 注释同步(staged rows 类型)
test/registered/kernels/ops/embeddings/test_engram_gather.py: 新增;全部 256 字节值 bit 级对比 E2M1 spec 解码
```

```python
if MXFP4:
    byte = tl.load(w_base.to(tl.pointer_type(tl.uint8)) + row * (DIM // 2) + offs // 2, ...).to(tl.int32)
    nib = tl.where(offs % 2 == 0, byte & 0xF, byte >> 4)
    m = nib & 0x7
    bits = tl.where(m >= 2, (m + 252) << 22, m * (252 << 22))
    # Set the sign bit directly: before Triton 3.8, -x is 0 - x, which turns -0 into +0.
    vals = (bits | ((nib & 0x8) << 28)).to(tl.float32, bitcast=True)
```

## 性能声明

- 内存(每 TP4 rank 每张表):host 表 24,170 MiB(widened)→ 12,451 MiB(packed);每节点 host RAM(两表四 rank)约 189 GiB → 约 97 GiB。GPU 显存加载后不变(77.3 GB,该部署把表放在 host)。
- Gather kernel 时间(单张 GB300,24M 行表即一个 TP4 rank shard,random ids,µs):device gather 基本不变(256 ids:fp8 9.4 vs packed 6.7;131,072 ids:75.6 vs 75.6);pinned host + sorted ids:小 batch 略慢(256 ids 69.6 → 73.9;2,048 ids 73.6 → 82.8),大 batch 更快(16,384 ids 141.6 → 131.6;131,072 ids 326.5 → 252.5),即大 batch 最多快 23%,原因是经 C2C 只读一半字节。
- 精度:GB300 TP4 单节点,SGLang nightly 2026-10-30 + 本改动,DSpark(block size 5),`--engram-config '{"placement": "cpu", ...}'`,GSM8K(sgl-eval 0.1.3,1319 题,并发 64,temperature 1.0,top_p 0.95,thinking on):packed 96.82% / accept length 3.38,widened e4m3 96.82% / 3.41,持平。

## 验证与回归风险

新增注册测试 `test_engram_gather.py`:全部 256 字节值 x 随机 scale,packed 与 widened 两路 gather 对 E2M1 spec 解码按 int16 位比较(覆盖 signed zero);GB300 上另跑 device/pinned-host/sharded/sorted、per-rank host 表、staged dequant、torch fallback 及四卡 peer kernel(真实 `_PeerShards` export/import)的逐位一致检查。风险:GSM8K 与计时数据在"直接设符号位"改动之前采集(在 Triton 3.8 上两版本 bits 相同);checkpoint 与配置的 mxfp4 声明不匹配会显式报错加载失败;FP8 checkpoint 路径不变。

## 关联

vllm-project/vllm#61530(vLLM 侧同类"保持打包、gather 内解码"实现,作为方法参照)。pack 未列出其他同分类 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
