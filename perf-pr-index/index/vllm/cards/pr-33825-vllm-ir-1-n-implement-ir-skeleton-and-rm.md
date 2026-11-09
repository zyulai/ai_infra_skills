# PR #33825: [vLLM IR] 1/N Implement IR skeleton and rms_norm op

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ProExpertProg |
| 合并 | 2026-04-01 |
| 分类 | Model-specific Fast Path \| tier: major |
| Label | rocm, intel-gpu, ready, torch.compile, ci/build, nvidia, ready-run-all-tests, vllm-ir |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注 34 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/33825 |

## 一句话结论

引入 vLLM IR 骨架与 rms_norm op，B200 上 Qwen3-0.6B default 延迟 0.197s→0.196s。

## 动机

PR 正文 Purpose 说明，本 PR 实现 vLLM IR（functional IR）的基础设施，把 op 语义与实现、分派分离，作为更高层的 torch dialect，带来的好处是 streamlined compilation passes、single source of truth dispatching、简单可扩展的 op/kernel 注册。这是 addressing RFC #32358 系列的第一个 PR，先从 `rms_norm` 落地；此前 RMSNorm 依赖脆弱的 `MatcherRMSNorm` 模式匹配工具，本 PR 用 `torch.ops.vllm_ir.rms_norm` 取而代之。

## 优化点

机制改动（均来自正文）：用 `vllm.ir.register_op` 注册装饰器返回含元数据与工具的 `IrOp`，`IrOp.register_impl` 返回 `IrOpImpl`；eager 模式下 `IrOp.dispatch` 按 priority list 与 runtime `support_args` 谓词把调用分派到所选实现；`VllmIRLoweringPass` 在 post-grad custom post-passes 末端按同样的 priority 与谓词把 `vllm_ir` torch op 降级到实现，保证 eager 与编译路径一致；新增 `KernelConfig` 中的 `IROpPriorityConfig` 与顶层 CLI flag，各 platform 定义默认 op priority 并在每次 forward 起始下传。kernel provider 分四类：vllm_c、aiter、xpu_kernels、oink。正文明确本 PR 保留 `fused_add_rms_norm` 与 batch invariant 不动。

## 关键改动

```text
vllm/ir/ops/layernorm.py: rms_norm op 定义；实现按 provider 分置于 vllm/kernels/*.py（正文注明）
vllm/compilation/passes/ir/lowering_pass.py: 新增 VllmIRLoweringPass，按 priority list 与 support_args 谓词把 vllm_ir op 降级到所选实现
vllm/compilation/passes/fusion/matcher_utils.py、qk_norm_rope_fusion.py、allreduce_rms_fusion.py: 用 torch.ops.vllm_ir.rms_norm 替换 MatcherRMSNorm 的脆弱匹配
vllm/compilation/passes/pass_manager.py: 把 IR lowering pass 接入 post-grad custom post-passes 末端
tests/ir/test_op.py、tests/kernels/ir/test_layernorm.py、tests/compile/passes/ir/test_lowering.py: 新增 IR 注册/分派/优先级、RMSNorm 实现、lowering pass 单测
```

## 性能声明

正文给出 Qwen-0.6B latency configuration sweep on B200，命令 `vllm bench latency --model=Qwen/Qwen3-0.6B`，Median latency（main → PR）：`--enforce-eager` 1.037s → 1.010s；`-cc.cudagraph_mode=FULL_DECODE_ONLY -cc.mode=NONE` 0.229s → 0.229s；`-cc.cudagraph_mode=FULL_DECODE_ONLY` 0.210s → 0.210s；`-cc.custom_ops+=+rms_norm --ir-op-priority.rms_norm=vllm_c` 0.214s → 0.216s；`(default)` 0.197s → 0.196s。测试条件：B200、Qwen/Qwen3-0.6B、`vllm bench latency`；精度与 workload 正文未注明。正文另给出 B200/H100 的 lm_eval gsm8k 数值（见验证）。

## 验证与回归风险

PR 引入/依赖的测试：新增 `tests/ir/test_op.py`（注册/分派/优先级）、`tests/kernels/ir/test_layernorm.py`（RMSNorm 实现）、`tests/compile/passes/ir/test_lowering.py`（lowering），以及现有 compile/custom pass 单测；E2E lm_eval、E2E perf、人工检查 Inductor graph，并用 `qwen0.6B --enforce-eager` 测 worst-case dispatching overhead。Test Result 写明 CI passing，Deepseek-V3.1 的 Inductor 输出代码手工确认一致；B200/H100 gsm8k exact_match 数值与 main 基本持平（B200 Qwen3-30B-A3B flexible 0.8582→0.8620、Llama-3.1-70B-FP8 0.8052→0.8029、DeepSeek-V3.1 0.9613→0.9583）。回归风险：正文说明 `fused_add_rms_norm` 与 batch invariant 部分保持原样、留待后续 PR，worst-case 为 eager 模式的 dispatching overhead；其他平台/模型路径 PR 未注明。

## 关联

正文明确 addressing RFC #32358，并说明后续 2/N（PR #36816）与 3/N（PR #36823）分别处理 `fused_add_rms_norm` 与 batch invariant，之后可把 `RMSNorm` 从 `CustomOp` 迁移到 `PluggableLayer`。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
