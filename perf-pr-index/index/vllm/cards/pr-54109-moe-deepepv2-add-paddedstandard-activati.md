# PR #54109: [MoE][DeepEPV2] Add PaddedStandard activation format

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @bnellnm |
| 合并 | 2026-11-05 |
| 分类 | MoE \| tier: major |
| Label | rocm, intel-gpu, ready, ci/build, deepseek, cpu, gpt-oss, nvidia, quantization, DSv4.1, nixl |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注 40 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/54109 |

## 一句话结论

引入 PaddedStandard 激活格式，显式传递 DeepEPV2 的 padding 契约，避免 kernel 读到未初始化行出错。

## 动机

正文 Motivation 说明，`DeepEPV2PrepareAndFinalize` 交给 experts 的激活张量行数是上界而非精确 token 数，含 inactive padding（形状仍是 2D `(num_tokens, hidden_dim)`，padding 在 token 维）。原因：dispatch 后每 rank 收到的 token 数依赖数据、host 侧不做 device sync 无法得知；decode/cudagraph 模式 recv buffer 按最坏情况 `R * num_max_tokens_per_rank` 分配，只写 `[0, num_recv_tokens)` 前缀、尾部未初始化；prefill 模式按 expert 分组并 pad 到 `expert_alignment`，gap 置零。inactive 行以 `topk_ids` 全设 `-1` 标记，真实/填充边界由 `ExpertTokensMetadata.psum_recv_per_rank` 在 device 上携带。此前这被通用 `Standard` 格式掩盖：能跳过 padding 的 kernel 无法 opt-in，不能处理 padding 的 kernel 无法拒绝布局——这是 correctness issue，decode 尾部未初始化，扫描每行建路由的 expert 可能把 garbage 误认合法 expert id 而污染真实 token。

## 优化点

新增独立的 `PaddedStandard` 激活格式，把 padding 契约从 prepare/finalize 显式传给 experts，只选择真正支持 padded input 的 expert。`PaddedStandard` 建模为 `Standard` 的超集（把无 padding 当作 zero-padding 情形），`BatchedExperts` 仍是 disjoint；每个 kernel 声明单一 `activation_format()`，超集关系由枚举上的单向 `is_superset` 助手表达（方向为 `experts_format.is_superset(prepare_finalize_format)`）。集中式格式选择：`FusedMoEConfig.activation_format` 一次解析（DeepEP-LL/NIXL-EP/batched_triton → BatchedExperts，DeepEP V2 → PaddedStandard，其余 Standard），各 dtype oracle selector 统一改用它；并修复 `oracle/fp8.py` 显式 DeepGEMM 分支 `== Standard` 会把 PaddedStandard 误路由到 BATCHED_DEEPGEMM 的潜在 bug。`TrtLlmMxfp4ExpertsModular` 拆分为 Base 与两个子类，仅 mxfp4+mxfp8 变体支持 PaddedStandard；`--moe-backend=deep_gemm` 改走 format-aware 的 `select_mxfp4_moe_backend` 以到达 `DeepGemmFP4Experts`。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/modular_kernel.py: 新增 FusedMoEActivationFormat.PaddedStandard 与 is_superset() 关系，_post_init_setup 用其断言格式兼容
vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py: DeepEPV2PrepareAndFinalize.activation_format 改报 PaddedStandard
vllm/model_executor/layers/fused_moe/config.py: 新增 FusedMoEConfig.activation_format 集中解析 all2all 格式，并移入 use_batched_activation_format
vllm/model_executor/layers/fused_moe/oracle/fp8.py 及各 dtype oracle: 改用 config.activation_format，修复 == Standard 的误判
tests/kernels/moe/test_deepgemm.py、test_trtllm_bf16_moe.py、test_deepep_v2_moe.py: padding 鲁棒性与格式兼容性测试
```

## 性能声明

PR 正文未给出定量性能数据。正文 Results 只列测试通过数与说明，并明确 "compute-savings of tile/block-granularity skipping is a benchmark concern, not a pytest assertion"；测试条件为 8× B200（SM100）、FlashInfer 0.6.18、CUDA 13.0（正文注明），并非性能 benchmark。

## 验证与回归风险

PR 引入/依赖的测试：新增 deep_gemm padding 测试、`test_trtllm_bf16_moe_padding_robust`（在 workspace arena / inactive rows 投毒 NaN，断言有效输出有限且与干净 padding 逐位一致），更新 `test_deepep_v2_moe.py` 仅用支持新格式的 experts。Results（8× B200 SM100、FlashInfer 0.6.18、CUDA 13.0）：`test_deepgemm.py` 20 passed、`test_trtllm_bf16_moe.py` 5 passed、`test_deepep_v2_moe.py` 18 passed、`test_unquantized_backend_selection.py` 25 passed、`test_moe_kernel_oracle.py` 1 passed、`test_auto_round.py` 97 passed/5 failed；正文说明 5 个失败是未修改树上同样出现的 `_C` 扩展 stale、`_C::gptq_marlin_repack()` 签名不匹配与 MXFP4 GEMM/TMA 初始化失败，与本改动无关。`test_deepep_v2_moe.py` 容差改为按参考输出幅值缩放（`_scaled_tolerances`）。PR 未注明回归风险；改动覆盖所有 `FusedMoEExperts` 的 `activation_format()` 声明、各 dtype oracle selector 与 DeepEP V2/TRTLLM/DeepGEMM 路径，未声明 PaddedStandard 的 expert 在 DeepEP V2 下会被 assert 拒绝（设计意图）。

## 关联

无（正文虽提到把其余 mxfp4 call sites 接入 oracle 是 "a follow-up PR"，但未给出 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
