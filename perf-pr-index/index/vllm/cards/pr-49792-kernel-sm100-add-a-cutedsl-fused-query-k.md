# PR #49792: [Kernel][SM100] Add a CuTeDSL fused query kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zhou9402 |
| 合并 | 2026-08-04 |
| 分类 | Attention / Kernel |
| Label | performance, ready |
| Diff | pack 未提供 diffstat(+/- 与 files 数缺失,详见 PR 页面) |
| 链接 | https://github.com/vllm-project/vllm/pull/49792 |

## 一句话结论

为 DSA sparse attention 的 fused_q 预处理新增 SM100 CuTeDSL 实现(MQA 打包 fp8 + indexer RoPE/UE8M0 量化一内核完成),系列整体 decode 吞吐 +21%。

## 动机

GLM-5.2 / DeepSeek-V3.2 的 DSA sparse attention 在每步 decode 前要做 fused query 预处理:把 `ql_nope + q_pe` 打包成单个 fp8 MQA query、对 indexer query 做 RoPE 与 UE8M0 fp8 量化、折叠 index weights。原实现是 Triton kernel,在 SM100 上受限于 grid 形状 `(3, num_tokens, grid_heads)` 与逐 head 的发射方式,是 decode 热路径上的固定开销。

## 优化点

新增 `fused_q_cutedsl.py`(537 行 CuTeDSL kernel)并在 `fused_q` 入口做运行时分发:`is_fused_q_cutedsl_supported` 仅在支持组合下启用(SM100、quantize_mqa、bf16、q_pe dim=64、ql_nope dim=512、head 数 %4==0,indexer head 数 %16==0 且 dim=128),其余保持 Triton fallback。内核机制:单个 1D grid 内做 CTA 特化——前 `num_mqa_ctas` 个 CTA 跑 MQA 打包(每 warp 处理一个 head group,256-bit `CopyUniversalOp` 批量加载,nope/rope 分支处理,fp8 scale 取 `1/q_scale`),其余 CTA 跑 indexer(8-lane subwarp,RoPE interleave/非 interleave 两种布局,warp 内 `shuffle_sync_bfly` 求 amax 后用位操作算 UE8M0 的 2 的幂 scale,折叠 index weights);全程 PDL(`use_pdl=True`,`griddepcontrol_wait/launch_dependents`)与后续 kernel 重叠。配套 `vllm/cute_utils` 增加 Float32 dtype 映射、`fp32x2_to_fp8x2` PTX 转换与 bf16 recast 支持。

## 关键改动

```text
vllm/models/deepseek_v32/nvidia/ops/fused_q_cutedsl.py: CuTeDSL FusedQKernel(新增)
vllm/models/deepseek_v32/common/kernels.py: fused_q 入口按 is_fused_q_cutedsl_supported 分发
vllm/cute_utils/{__init__,cvt}.py: Float32 dtype + fp32x2_to_fp8x2 PTX + bf16 recast
benchmarks/kernels/benchmark_fused_q_cutedsl.py: Triton vs CuTeDSL 对比基准(新增)
```

## 性能声明

8x B300 TP8,`nvidia/GLM-5.2-NVFP4`,MTP=5,`VLLM_USE_V2_MODEL_RUNNER=1`,fp8 KV cache,8192 input / 1024 output tokens,concurrency 1,64 prompts:main(torch-compiled)446.7 output tok/s / 4019.8 total tok/s / median TPOT 1.94 ms;本 PR 单独 448.4 / 4035.3 / 1.93 ms;full series(#48597 re-split 全系列)542.5 / 4882.7 / 1.56 ms(即全系列 output +21.4%、TPOT -19.6%;本 PR 单独贡献很小)。未提供 kernel 级微基准数字(基准脚本已加入)。

## 验证与回归风险

分发条件硬编码 dtype/shape 组合,不满足即回退 Triton,ROCm 路径不受影响(`current_platform.is_cuda()` 才 import CuTeDSL 模块);新增 benchmark 脚本强制两条路径对比。风险面限于 SM100 + DSA 模型的 query 预处理;`fused_q` 共享给 ROCm,assert 新增 `positions.dtype == int64` 与 `q_scale` 约束。属 #48597 re-split 的一部分,需与其余系列 PR 合并序一致。

## 关联

系列:#48597(tracker,merge order)。同区域(DSA/DeepSeek-V3.2):PR #37421(persistent TopK scheduler)、PR #46076(FLASHINFER_MLA_SPARSE DCP)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
