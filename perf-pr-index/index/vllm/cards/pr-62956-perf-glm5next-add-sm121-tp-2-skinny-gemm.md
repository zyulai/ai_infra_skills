# PR #62956: [Perf][glm5next] Add SM121 TP=2 skinny-GEMM plans for GLM-5.3-Flash

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @TimRen01 |
| 合并 | 2026-10-31 |
| 分类 | Speculative Decoding \| tier: major |
| Label | ready, glm |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/62956 |

## 一句话结论

GLM-5.3-Flash 在 2× DGX Spark SM121 TP2 把 BF16 decode GEMM 换成 CuTe DSL skinny GEMM，conc1 TPOT −15%。

## 动机

`nvidia/GLM-5.3-Flash-NVFP4` 的 attention 投影（KDA 与 MLA）、shared experts 与 LM head 仍为 BF16，在 cuBLAS gemv 上约占 68 ms decode step 中的 46 ms。`glm5next` 尚无 low-latency GEMM 模块，本 PR 参照 `qwen4_exp/nvidia/low_latency_gemm.py`（#59753、#59632、#61437）新增一个，并为 TP=2 的本地形状加入实测的 SM121 skinny-GEMM plans，从 `Glm5NextForCausalLM` 与 `Glm5NextMTP` 安装。Supersedes #62949。

## 优化点

把 BF16 decode GEMM 从 cuBLAS gemv 改走 CuTe DSL skinny GEMM：为 7 个本地 `(N, K)` 形状在 M=1/2/4 上登记 20 条 plan，仅在 capability `(12, 1)`（SM121）分派，其他 GPU、dtype、token 数与形状仍走标准 linear 路径（cuBLAS）。plan 来自对 block_size、outputs_per_block、k_unroll、vector_width、static_k 的 CUDA-graph sweep（weight 移出 L2），只保留比 cuBLAS 快 >5% 的条目；`MLA o_proj` 的 M=1 因收益仅 3% 未纳入，M=8/16 与 cuBLAS 相差 2% 内也未列出。替换方法跟随 layer 携带的 quant method（使 `ParallelLMHead` 拿到 `UnquantizedLinearMethod`）。未覆盖：TP=1、KDA `f_b`/`g_b` `(4096,128)`（#57617）、MTP `eh_proj`、M=3。正文还说明这些是 cold-L2 数字（decode 每步读一次权重、45 层放不进 24 MB L2），按各形状在 decode step 中的出现频率加权，M=1 时约省 13 ms/step、M=2/4 约 3 ms，而 server 因 GEMM 与 MoE、attention 交错实际只拿到 10 ms；nsys census 显示 skinny GEMM 在无 speculation 时占 GPU 时间 45%、MTP 下 39.5%。

## 关键改动

```text
vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm_plans.py: 新增 SM121 GLM-5.3-Flash TP=2 的 20 条 skinny-GEMM plan
vllm/model_executor/layers/linear.py / vocab_parallel_embedding.py / utils.py: 按 capability 分派 skinny GEMM 与 maybe_use_skinny_gemm hook
vllm/model_executor/warmup/kernel_warmup.py: 安装 / 预热 skinny GEMM
tests/kernels/test_bf16_skinny_gemm.py: 新增 glm5next 表有效性、capability 路由与选中形状测试
```

## 性能声明

kernel：GB10、CUDA graph replay、weight 移出 L2、nightly `8e2beaa3bf`，每条 plan 输出与 `F.linear` 的 cosine 为 1.000000；M=1 比 cuBLAS gemv 快 1.35x–1.73x（238–283 GB/s 对 162–176 GB/s），M=2/4 快 1.04x–1.36x（表中最快 1.73x 为 shared down `(4096,1024)` M=1 的 51→30 µs）。端到端：`nvidia/GLM-5.3-Flash-NVFP4` TP=2、2× DGX Spark、FP8 KV、SPEED-Bench 2K in / 1K out、8 requests、conc1、`--temperature 0`、每 arm 交错 2 次、nightly `8e2beaa3bf`：无 speculation 时 TPOT p50 64.17/63.83 → 54.14/54.10 ms（均值 −15.4%）、ITL p50 63.94/63.68 → 54.03/53.96（−15.4%）、Output tok/s 15.27/15.33 → 18.02/18.03（+17.8%）、GSM8K(n=100) 93 → 94，正文概述"TPOT 64.0 -> 54.1 ms (-15%), 15.3 -> 18.0 tok/s"。MTP k=3（同 nightly 与采样）：TPOT −2.7%、ITL −3.6%、tok/s 26.7 → 27.8（+3.9%）、acceptance length 2.87/2.87 → 2.85/2.90。

## 验证与回归风险

`pytest tests/kernels/test_bf16_skinny_gemm.py -k glm5next` 报 26 passed（GB10），覆盖表有效性、capability 路由、hook 安装与 20 个选中形状对 `F.linear` 的 cosine > 0.999；selected-shape 测试在 capability `(12,1)` 之外 skip，而 CI 无该设备。正文称未选中的形状与 M=3 回退 cuBLAS、其他 GPU/dtype 不受影响；GSM8K(n=100) 93 → 94 作为精度对照。正文另注明 sweep 脚本、模块与测试使用了 Claude Code 辅助，全部代码与测量均由作者复核并运行。正文未列出其他回归风险。

## 关联

正文明确列出：#59753 / #59632 / #61437（参照的 `qwen4_exp` low-latency GEMM 系列）、#62949（本 PR supersedes）、#57617（KDA `f_b`/`g_b`，未覆盖）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
