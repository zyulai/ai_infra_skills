# PR #59923: [Perf][DSv4.1] Exchange packed FP8 Engram rows across DP and feed wkv MXFP8

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Juntian777 |
| 合并 | 2026-10-09 |
| 分类 | Speculative Decoding (tier: major) |
| Label | ready, deepseek, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);diff 可见 6 个文件(engram、query_quant、engram_peer、model 与测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/59923 |

## 一句话结论

DSv4.1 Engram 行保持 FP8 packed 直喂 wkv,相对 DP-shared 表 1024 token 起快 3.4-13.6x。

## 动机

PR 正文 Purpose:DeepSeek-V4.1 的 Engram 表本身已是 MXFP8(FP8 行 + 每 32 值一个 ue8m0 scale),`wkv` 也是 MXFP8 GEMM,但查找时先把每行反量化成 BF16、DP 副本逐层 all-gather 这些 512-byte 行,`wkv` 又把它们重新量化回去。本 PR 让行从 host 表到 `wkv` 始终保持 packed。

## 优化点

三条机制:① 查找直接按原样拷贝每行的 256 FP8 字节与 8 scale 字节,一个 unpack kernel 写成 FlashInfer MXFP8 activation layout(F8_128x4 scales),`wkv` 以 `QuantizedActivation` 接收,省去反量化与重量化且输出 bit-identical。② `EngramBatch` 把各层查到一块 packed buffer,用单次 all-to-all 只发各 rank 自己的 token(每 head row 264 而非 512 字节,为 all-gather 的 1/DP),再一次 unpack 所有层。③ 每 rank 少于 128 token 时 `EngramPeerExchange` 用对称内存发布 ids、查各 peer 的 head shard、barrier 后读回,替换 hash/逐层 all-gather。适用条件:DP-sharded 表、TP=1、head_dim%128==0、无 LoRA、`wkv` 接受 F8_128x4 scales;peer 还需 SM100、CPU offload、2/4/8 全互联 GPU,否则全走 all-to-all。其余平台/配置保持原 BF16 路径。

## 关键改动

```text
vllm/models/deepseek_v41/common/engram.py: 查找 kernel 增加 PACKED 路径,输出 QuantizedActivation 直供 wkv;引入 ENGRAM_A2A_MIN_SLOT=128 分界
vllm/models/deepseek_v41/nvidia/ops/engram_peer.py: decode 的 EngramPeerExchange 对称内存 peer 交换
vllm/models/deepseek_v41/nvidia/model.py: 接通 batched / peer 交换路径
vllm/models/deepseek_v41/common/ops/query_quant.py: 抽出 F8_128x4 scale offset/size 与 can_fuse_query_quant 复用
tests/distributed/test_engram_dp_shard.py、tests/kernels/test_engram.py: packed 查找输出与 4-GPU 跨 peer/all-to-all 槽位测试
```

关键机制代码(packed 与分界):

```python
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
ENGRAM_A2A_MIN_SLOT = 128
...
    """Gather FP8 rows as packed values/scales or dequantized BF16."""
```

## 性能声明

Kernel microbenchmark(4× GB300 单节点,DeepSeek-V4.1-Flash 表 24 heads×16M rows / 94.4 GiB per layer,CUDA graph replay,slowest rank 40 样本中位数):DEP4 下 1 token 时 DP-shared 34 / THP+all-gather 80 / 本 PR 36 us(2.22x),127 token 136/136/50(2.72x),32768 16008/7860/3800(2.07x);正文称 1-64 token 相对 DP-shared 最多慢 8us,1024 token 起快 3.4-13.6x。E2E prefill(DEP4,GB300,fp8 KV,DeepGEMM mega MoE):TTFT 相对 DP-shared 为 0.99x~1.04x、相对 THP+all-gather 约 1.00x~1.02x;decode 中 Engram 占 TPOT 不足 1%。GSM8K(DEP4,5-shot)main DP-shared 92.87/92.95% strict,本 PR THP 93.33/93.10%,在 ±0.7 stderr 内,正文不主张精度增益;DSpark strict 92.57%→92.57%、acceptance length 3.365→3.398。

## 验证与回归风险

Test Plan:`pytest tests/kernels/test_engram.py tests/distributed/test_engram_dp_shard.py tests/kernels/core/test_fused_q_kv_rmsnorm.py -q` 与 pre-commit/mypy;4× GB200 上 202 passed(102 Engram kernel、27 distributed、73 fused_q_kv_rmsnorm)。另校验 staged MXFP8 字节与表行逐字节一致、`wkv` 输出(CuTe-DSL/CUTLASS)与 BF16 路径 bitwise identical、graph replay 与 level-2 sleep 后一致。影响路径:DSv4.1 的 DP-sharded Engram;TP、DP-shared 表、单副本、ROCm 保持原 BF16 路径。

## 关联

正文 "Not a duplicate" 提及 #56989(TP head exchange 打包 FP8 行)、#57893(小查找 tile 调优)、#58766(曾通过现有 gathers 喂 MXFP8 给 wkv,作者关闭);E2E 基于 #59532、DSpark 对比 #59511 的 DEP4 base。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
