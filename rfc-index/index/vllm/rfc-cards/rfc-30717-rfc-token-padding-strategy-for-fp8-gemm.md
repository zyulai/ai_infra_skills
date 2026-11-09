# RFC #30717: [RFC]: Token Padding Strategy for FP8 GEMM Performance Optimization

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#30717](https://github.com/vllm-project/vllm/issues/30717) |
| 作者 | 0xjunhao |
| 创建日期 | 2025-12-15 |
| 状态 | ⚪ closed(关闭于 2026-05-07) |
| labels | RFC, stale |
| 热度 | 👍 0 · 💬 15 |
| 分类 | Quantization |

## 一句话结论
提议在执行循环中把 token 数 pad 到 4 的倍数再跑 FP8 GEMM,B200 上端到端吞吐最高提升 26.2%。

## 提案动机
FP8 GEMM kernel(DeepGEMM 与 CUTLASS 实现)对输入维度敏感:B200 上的基准显示,token 数(M)未对齐特定边界时吞吐明显劣化。当前 `gpu_model_runner.py` 把原始 token 数直接传给执行后端;CUDA Graph 捕获只为小批(通常到 512)做 padding,既不保证所有 graph slot 满足 M%4==0,也不覆盖超过捕获上限的大批。对类生产流量(ShareGPT,3000 prompts,并发 100)的分析显示,>512 token 的批占引擎步骤的显著部分,且其对 4 的余数均匀分布,约 74% 的大批目前跑在未优化的 kernel 路径上。

## 提出的方案
- 在 `gpu_model_runner.py` 的 `_determine_batch_execution_and_padding` 中加入 padding 逻辑:把 num_tokens 补齐到最近的 4 的倍数。
- 选择性触发:仅当后端 kernel 受益于对齐时(包括 NVIDIA Blackwell/Hopper 上的 DeepGEMM/FP8 与 CUTLASS/FP8),而不是笼统地按 `quantization == "fp8"` 判断。
- 安全性:确保 padded 的 dummy token 被掩蔽或妥善处理,不影响模型精度、不触发越界(OOB)内存访问。
- 已否决的替代方案:Option A 在 kernel 层面 pad——深层 LLM 每次前向需数百次多余拷贝,运行时开销过高;Option B 靠扩大 CUDA Graph 捕获范围——捕获尺寸通常按 8 或 16 步进、默认上限 512,扩到大 M 显著增加 VRAM,且开销比 mod-4 padding 更大。

## 优化点与收益
正文给出量化数字(端到端:`vllm bench serve`,ShareGPT,2000 prompts,--max-concurrency 100):
- B200 + Qwen/Qwen3-8B-FP8:总 token 吞吐 15455.37 → 17350.65 tok/s,**10.9%**。
- B200 + Qwen/Qwen3-32B-FP8:6350.89 → 8016.02 tok/s,**26.2%**。
- H200 + Qwen3-8B-FP8:**2.0%**;H200 + Qwen3-32B-FP8:**1.8%**。
- GEMM-only(B200,`benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py`,n=24576/k=1536):如 m=63→64,DeepGEMM 103.1→63.8 μs(TFLOPS 46.1→75.7);m=4095→4096,vLLM CUTLASS 778.0→204.3 μs(TFLOPS 397.4→1513.9);对齐收益显著。H200 的 GEMM 基准要求 m 为 4 的倍数,故未跑。

## 现状与落地
已关闭(2026-05-07)。正文附有 Draft PR #30524;关闭时改动是否合入从正文无法看出,落地情况需查关联 PR。

## 关联
- [Draft PR #30524](https://github.com/vllm-project/vllm/pull/30524)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
