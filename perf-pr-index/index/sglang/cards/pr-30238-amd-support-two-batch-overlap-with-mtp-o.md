# PR #30238: [AMD] Support two batch overlap with MTP on DeepSeekV4

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @At1a8 |
| 合并 | 2026-07-16 |
| 分类 | Speculative Decoding |
| Label | deepseek, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 含 3 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/30238 |

## 一句话结论

修复 DeepSeek-V4 MTP 与 prefill TBO 同时启用时的失败,在 AMD MI35x 上把两者叠加,吞吐再 +4.8%~(累计 +25.85%)。

## 动机

DeepSeek-V4 开启 MTP 后 TBO(two-batch-overlap)直接失败。根因有二:一是代码把 `TARGET_VERIFY`(MTP target-verify)请求误放进 prefill-only 的 TBO 策略——开启投机解码后 target-verify batch 也上报 `is_extend()`,但它们并非真正的 prefill;二是 DP 下 prefill TBO 会把一个 micro-batch 按 token 切分,某个 rank 上可能切出 0 query token 的 ubatch,而 DeepSeek-V4 compressor prefill plan 的 JIT 内核有 `batch_size <= num_q_tokens` 运行时检查,0 token 时直接崩溃,且该 rank 不能单方面跳过 TBO(全局 TBO 决策需各 rank 一致)。

## 优化点

两处修复。其一,`_can_run_tbo` 的门控从 `global_forward_mode.is_extend()` 改为 `is_extend_without_speculative()`,只有真正的 prefill/extend batch 走 prefill TBO 策略,decode/target-verify 由其他路径处理。其二,给 DeepSeek-V4 compressor 的 prefill plan 生成加 0-token guard(仅 AMD HIP 生效):`num_q_tokens == 0` 时返回空 `CompressorPrefillPlan` 作为本地 no-op,下游 compressor 处理 0 token 等价于跳过,避免 c_plan.cuh 的 `RuntimeCheck(batch_size <= num_q_tokens)` 失败,同时保持各 rank 的 TBO 决策统一。典型触发场景:DP0 prefill batch 中 seq0 extend_len=1,TBO 切成两个 ubatch 后 ubatch1 分到 0 个 query token。

## 关键改动

```text
python/sglang/srt/models/deepseek_v4.py: _can_run_tbo 门控改用 is_extend_without_speculative(),防止 target-verify 进入 prefill TBO
python/sglang/jit_kernel/dsv4/compress.py: num_q_tokens==0 且 is_hip() 时返回空 plan(DP-safe empty-batch guard)
test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py: 新增 MI35x 8 卡 DeepSeek-V4-Pro FP4 + DP TBO + MTP 端到端 gsm8k 测试(断言 accuracy > 0.91)
```

## 性能声明

测试环境:MI35x(MI355)ROCm 7.2,DeepSeek-V4-Pro(1.6T)FP4,TP8 + DP8(enable-dp-attention),EAGLE MTP(num-steps 3, topk 1, draft-tokens 4),bench_serving random 数据集。

- 8k/1k 输入/输出,conc=256:TPDP baseline 29627.78 tok/s(TTFT 18448.88 ms,TPOT 59.11 ms);+prefill TBO(PR #29362)31260.94 tok/s(+5.51%);+MTP 35629.73 tok/s(+20.26%,TTFT 6904.60 ms,TPOT 56.08 ms);**+MTP+prefill TBO(本 PR)37285.57 tok/s(+25.85%,TTFT 6099.64 ms,TPOT 53.57 ms)**。gsm8k:baseline 0.951 / TBO 0.948 / MTP 0.947 / MTP+TBO 0.949。
- 50k/0.5k,conc=64:TPDP+MTP 43081.12 tok/s(TTFT 19144.47 ms,TPOT 110.14 ms);+prefill TBO(本 PR)45139.73 tok/s(**+4.78%**,TTFT 18586.30 ms,TPOT 103.15 ms)。
- 独立 gsm8k 回归(1319 题):Accuracy 0.952;正文注明一次 GSM8K 回归问题由 PR #30333 修复。

## 验证与回归风险

新增 `test_deepseek_v4_pro_fp4_tbo_mtp.py` 注册到 nightly-amd-8-gpu-mi35x-deepseek-v4-pro-mtp 套件(nightly,est_time 14400s),启动 8 卡服务器跑 8-shot gsm8k 1319 题,断言 accuracy > 0.91(TBO+MTP 不得劣于非 TBO MTP baseline)。风险面:`is_extend_without_speculative()` 收紧了 TBO 门控,target-verify batch 改走非 prefill TBO 路径;0-token guard 仅 HIP 生效,CUDA 路径若遇到同样切分场景仍可能触发检查失败。

## 关联

前序:PR #29362(prefill TBO 基线);本 PR 修复的 GSM8K 回归由 PR #30333 处理。同区域为 DeepSeek-V4 系列 TBO/投机解码优化。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
