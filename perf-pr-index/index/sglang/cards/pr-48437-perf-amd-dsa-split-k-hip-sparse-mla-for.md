# PR #48437: [Perf] [AMD] dsa: Split-K HIP sparse MLA for decode/verify

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @RolaoDenthu |
| 合并 | 2026-11-04 |
| 分类 | Speculative Decoding |
| Label | jit-kernel |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 可见改 4 个文件(HIP kernel cuh/py、triton_sparse_mla.py、environ.py) |
| 链接 | https://github.com/sgl-project/sglang/pull/48437 |

## 一句话结论

在 MI355X 上为 sparse MLA decode/verify 小 batch 增加 HIP split-K 形态,sparse MLA decode 步骤约快 6 倍,TP8 投机解码并发 32 时吞吐 +6.9%。

## 动机

MI355X(gfx950)上 Kimi-K3.1 跑 TP8 投机解码时,target-verify batch 走 `triton_sparse_mla_fwd` 的 Triton split-K kernel;每个请求贡献 bs×6 行(DFLASH MLA5 draft,5 个 draft token),一个 batch 是 16-512 行查询。在 TP8 bs48 decode 步中 sparse MLA 占约 18% 步时。已入库的 gfx950 HIP sparse MLA prefill kernel 每行一个 CTA,在这种行数下填不满 GPU。

## 优化点

给 `sparse_mla_prefill_gfx950` 增加编译期 `kSplit` 模板形态(`run_split` / `sparse_mla_split_gfx950`):每行的 top-k key tiles 拆分到 grid.y 个 CTA,各算一段区间,partial 结果(unnormalized output、running max、row sum,均为 fp32)写入 workspace,再由新增的 `sparse_mla_split_combine_kernel` 做 fp32 log-sum-exp 合并——合并不引入额外舍入。Python 侧 `sparse_mla_decode_splits(rows)` 按行数选 split 数(≤6 行 S16、≤128 行 S8、≤512 行 S4、更大 S2),来自 per-shape microbenchmark。路由层 `triton_sparse_mla_fwd`:16-512 行(H≤16 且形状被 `supports_sparse_mla_prefill_gfx950` 支持)走 HIP split-K;6/12 行的小 batch 留在 Triton split-K(实测更快);512 行以上维持单 CTA HIP 路径;仅非 DCP。新增环境开关 `SGLANG_ROCM_SPARSE_MLA_DECODE_HIP`(默认开,置 0 回退 Triton)。

## 关键改动

```text
python/sglang/kernels/jit/csrc/attention/sparse_mla_prefill_gfx950.cuh: kSplit 模板 + split combine kernel
python/sglang/kernels/ops/attention/dsa/sparse_mla_prefill_gfx950.py: run_split wrapper 与 sparse_mla_decode_splits 启发式
python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py: 16-512 行路由到 HIP split-K
python/sglang/srt/environ.py: 新增 SGLANG_ROCM_SPARSE_MLA_DECODE_HIP 开关
```

split 数选择与合并:

```python
def sparse_mla_decode_splits(rows: int) -> int:
    if rows <= 6: return 16
    if rows <= 128: return 8
    if rows <= 512: return 4
    return 2
# combine: out = bf16(sum_s acc_s * 2^(m_s - M) / sum_s l_s * 2^(m_s - M)), M = max_s m_s
```

## 性能声明

Kernel time(MI355X,12 heads,top-2048,单位 µs):

| Rows | Triton split-K | HIP split-K |
|---:|---:|---:|
| 24 | 28.6 | 20.5 |
| 96 | 89.5 | 31.2 |
| 192 | 239 | 46.3 |
| 288 | 103(单 CTA HIP) | 70.8 |

端到端(MI355X,Kimi-K3.1,InferenceX fixed-seq 8k/1k,模拟 accept length 3.69,与同树关闭本改动配对):每 GPU 吞吐并发 4 +1.3%、16 +4.0%、32 +6.9%。sparse MLA decode 步本身约快 6 倍。小 batch(6/12 行)Triton 更快(12.8/17.8 vs 18.4/19.5 µs),故保留 Triton。

## 验证与回归风险

精度验证:Kimi-K3.1 on MI355X 与关闭本改动的同树对比,GSM8K(DCP8 c48, TP8 c4)不变,real accept length 不变。PR 未新增单元测试(diff 无 test 文件);CI 状态显示 Base/Extra/AMD ROCm 10 均为失败标记(x),正文未解释。风险点:`triton_sparse_mla_fwd` 是共享路由层,新增 16-512 行分支改变了 verify batch 的内核选择,依赖 `is_gfx95_supported()` 与形状支持谓词;kernel 对 q stride 16B 对齐、KV pool 31-bit 偏移、`kStages∈[2,4]` 有硬约束;workspace 分配(part_acc/part_ml)按 seq×splits×H×512 fp32 每次调用分配,大 batch 下显存开销上升。

## 关联

同区域优化:同文件的 gfx950 HIP sparse MLA prefill kernel(单 CTA/行,本 PR 的前序);`SGLANG_ENABLE_ROCM_DSA_OPUS_DECODE`(environ.py 相邻条目,opus decode MXFP8 内核)是同一 decode/verify 路径的另一 AMD 优化;同批 PR #33599(同作者,Kimi-K3 attn-residual 聚合)与 PR #35770(Kimi-K3 Triton MLA prefill)均属 Kimi-K3 AMD 优化系列。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
