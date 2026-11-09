# PR #28354: [FlashInfer v0.6.16] Support FlashInfer CuTe DSL NVFP4 MoE quantization

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zianglih |
| 合并 | 2026-08-14 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, quant, deepseek, blackwell, run-ci, bypass-fastfail |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);1 file dropped |
| 链接 | https://github.com/sgl-project/sglang/pull/28354 |

## 一句话结论

`--quantization nvfp4_online` 接入 FlashInfer CuTe DSL v2 MoE:BF16/FP16/FP8 专家权重加载期转 NVFP4,激活走 per-token FP32 动态 scale。

## 动机

此前 `nvfp4_online`(在线 NVFP4 权重转换 + 在线 per-token FP32 激活 scale)只支持 `flashinfer_trtllm`/`flashinfer_trtllm_routed` runner。本 PR 把 `flashinfer_cutedsl` v2 纳入该契约(限 no A2A 或 FlashInfer A2A),并厘清与 `modelopt_fp4`(per-tensor scale)的边界;目标是让 Nemotron 的 target 与 embedded MTP draft 自动落在同一 per-token 契约上(draft 继承 `nvfp4_online` + CuTe DSL v2/no-A2A,无需显式配置)。

## 优化点

(1) `nvfp4_online` 放行 `flashinfer_cutedsl` runner;DeepEP `low_latency` 仍保持 per-tensor CuTe DSL v1,不被计入 per-token 支持。激活量化改走 `flashinfer.nvfp4_quantize(per_token_activation=True, backend="cute-dsl")`,把返回的 per-token FP32 row scale 传给 runner;FlashInfer-A2A dispatch 保持 BF16,因为标量预量化 dispatch 无法表达 per-token 契约。(2) checkpoint 源的 exclude 层不再抑制在线转换,embedded MTP 专家可继承 `nvfp4_online`;源格式白名单从 fp8 扩至 modelopt_fp8。(3) CuTe DSL runner 构造并缓存 per-token 4over6 global scale;通过 `copy_or_rebind_param` + `refresh_cutedsl_standard_scales_for_weight_update` 原地更新 scale 张量(decode CUDA graph 已捕获地址,不能换张量),保证 `/update_weights_from_disk` 后参数身份不变。(4) 新增 `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE`(默认 true,FlashInfer fused atomic CUTLASS/CuTe DSL finalize;deterministic inference 强制关闭),仅 fused-finalize 的 selected-tactic replay 做跨 rank 同步。(5) 依赖 FlashInfer 0.6.16.post1 的 per-token NVFP4 API(#3645)与 deterministic non-fused finalize(#3976);tactic replay 顺序问题(#4192 未发布)保留窄 workaround。

## 关键改动

```text
python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py: per-token nvfp4_quantize 分支 + 4over6 scale 缓存与 reload 原地刷新
python/sglang/srt/layers/quantization/nvfp4_online.py: cutedsl 放行、exclude 层不抑制在线转换、MTP 继承
python/sglang/srt/layers/quantization/modelopt_quant.py: copy_or_rebind_param 保持参数身份;use_per_token_activation 传入 quant_info
python/sglang/srt/server_args.py: cutedsl+nVFP4_online 的 A2A 约束(discarteep 拒绝)
test/registered/backends/test_flashinfer_nvfp4_online_moe_backend.py: 新文件,cutedsl/trtllm 后端 GSM8K E2E 测试
```

## 性能声明

PR 明确 "No dedicated speed benchmark was run",accuracy 测试的吞吐不作为 benchmark 证据。精度数据:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8,4x B200(TP4/EP4),EAGLE 投机(3 steps、topk 1、4 draft tokens),temperature 1.0/top_p 0.95,200 例 GSM8K:得分 0.965,avg_spec_accept_length 3.1067(阈值 2.5),per-request 均值 3.1385;加载日志确认 target/draft 均在 `nvfp4_online`。

## 验证与回归风险

fast-prefill planner 2 passed;CuTe DSL MoE suite 6 passed + 14 subtests;cold-autotune Nemotron MTP/EAGLE E2E 1 passed;`/update_weights_from_disk` 1 passed(3 subtests,含 CUDA graph 与 flush_cache 两种,reload 间文本/token/logprob 一致)。环境:8x B200,flashinfer-python/cubin 0.6.16.post1。风险:已知 flashinfer#4486——SM100/SM103 上 TRTLLM_GEN tile-192 BMM 自 0.6.16.post4 起返回 non-finite,相关 trtllm 测试类被 skip(#34629);fused finalize 默认开启、deterministic 路径关闭;CI Extra 失败(x)、Base 通过。

## 关联

正文明确的相关链:#26083(initial online NVFP4)、#31382(embedded-draft reachability)、#33115(per-tensor 行为归入 modelopt_fp4 + 共享在线 weight loader)、#33092(FlashInfer 0.6.16.post1 依赖 bump)、#23317(cross-rank tactic sync 追踪)。
