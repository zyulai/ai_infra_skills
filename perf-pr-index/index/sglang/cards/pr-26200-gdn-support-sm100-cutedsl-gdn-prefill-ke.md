# PR #26200: [GDN] Support SM100 CuTeDSL GDN Prefill Kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-05-26 |
| 分类 | Attention / Kernel(tier: major) |
| Label | run-ci, run-ci-extra |
| Diff | diffstat 缺失(素材未统计行数);diff 超长,6 个文件被省略(含 3 个 kernel 文件),素材保留 benchmark、cute_utils、gdn_backend 等 |
| 链接 | https://github.com/sgl-project/sglang/pull/26200 |

## 一句话结论

把 vLLM 的 SM100 CuTeDSL GDN prefill kernel 移植进 SGLang,Blackwell 上 prefill 内核 geomean 1.78x,e2e 总吞吐 +12.7%。

## 动机

vLLM PR #43273 提供了 Blackwell 专用的 CuTeDSL chunkwise GDN prefill kernel,在多请求/混合长度/低 head 数的生产 serving workload 上同时优于 Triton FLA 与 FlashInfer 的 GDN Blackwell 路径。SGLang 原有 `gdn_cutedsl.py` 只接了 decode 路径,Blackwell 上 prefill 只能用 Triton。本 PR 按字节级移植上游 kernel 到 `gdn_blackwell/`,集成成本收敛为 `CuteDSLGDNKernel` 新增一个 `extend()` 方法与 `GDNKernelDispatcher` 的一个分支。

## 优化点

新增 3 个 CuTeDSL kernel 与共享 helper,沿用 FLA 设计:`kkt_inv_uw_kernel` 融合 FLA 的 chunk_scaled_dot_kkt_fwd、solve_tril、recompute_w_u_fwd;`h_kernel` 对应 chunk_gated_delta_rule_fwd_h;`o_kernel` 对应 chunk_fwd_o;并原生支持 BF16 state(无需 BF16->FP32 转换)。`cute_utils/` 提供 tcgen05/tmem/TMA bulk-swizzle 与 PTX BF16 封装。`extend()` 适配 SGLang 现有契约:kernel 外做 l2norm、从 state pool 取初始 state、用 `prepare_metadata_cutedsl` 构建 chunk 索引、回写最终 state;lazy import 使 SM90 机器仍能只为 decode 构造该 kernel。dispatcher 在 `prefill_backend=cutedsl` 且 `supports_prefill`(SM>=10)时路由到 CuTeDSL,否则打一行日志回退 Triton——SM90 路径行为不变。

## 关键改动

```text
python/sglang/srt/layers/attention/linear/kernels/gdn_blackwell/: 移植的 3 个 CuTeDSL kernel + chunk_gated_delta_rule_cutedsl / prepare_metadata_cutedsl 入口
python/sglang/srt/layers/attention/cute_utils/: tcgen05 / tmem / TMA / PTX BF16 共享 helper(Apache-2.0,源自 vllm/cute_utils)
python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl.py: CuteDSLGDNKernel.extend() 新增;decode 不变
python/sglang/srt/layers/attention/linear/gdn_backend.py: dispatcher 按 supports_prefill 路由 prefill,SM90 回退 Triton
benchmark/bench_linear_attention/bench_gdn_prefill_cutedsl.py: 19 形状正确性 sweep + do_bench_cudagraph 性能表
```

## 性能声明

内核微基准:NVIDIA B200,bfloat16,`do_bench_cudagraph`,Qwen3-Next preset(K=V=128,pool_size=256),14 个形状(B 4..128,H 16/32,T/seq 256/1024)对 Triton FLA baseline 加速 1.67x-1.92x,geomean 约 1.78x,无任何形状回退(例:B=4 H=32 T/seq=1024:1.482 ms -> 0.791 ms,1.87x)。

端到端:Qwen3.6-27B,B200 TP=1,random workload(input 4096 / output 256,64 并发 prompt),两后端 decode 均为 triton,仅 prefill 后端不同:总吞吐 13,557 -> 15,273 tok/s(+12.7%),input/output 吞吐均 +12.7%,Mean TTFT 8,243 -> 6,842 ms(-17.0%),P99 TTFT -14.8%,Mean TPOT 48.0 -> 44.5 ms(-7.4%)。

## 验证与回归风险

`test/registered/attention/test_gdn_prefill_cutedsl.py`:6 用例(num_seqs 1/5/257 x state_dtype bf16/fp32)以 `fused_recurrent_gated_delta_rule` 为基准,B200 6/6 通过,注册 4-gpu-b200 CI,SM<10 自动 skip;KDA kernel 回归 6/6;独立 sweep 19/19 PASS;SM90(H200)验证了 fallback 日志与 Triton prefill。GSM8K 5-shot(200 题)CuteDSL 0.725 vs Triton 0.620/0.640,正文将差异归因于 invalid 率数值方差而非质量差异。风险:kernel 硬编码 BT=64、K_dim=V_dim=128;KDA(per-K-dim gate)不能复用;上游遗留两个 reviewer 标记(TMEM 硬编码基址、H kernel TMEM dealloc 竞争)随移植带入;依赖 CUDA >= 13、`nvidia-cutlass-dsl[cu13]==4.5.0`、`quack-kernels>=0.3.3`(requirements/cuda.txt pin 为待办)。

## 关联

上游来源:vLLM PR #43273(@gau-nernst;原始 CUDA C++ 出自 MLSys 2026 FlashInfer Kernel competition 团队)。decode 侧既有 `cutedsl_fused_sigmoid_gating_delta_rule_update` 保持不变;同分类其他 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
