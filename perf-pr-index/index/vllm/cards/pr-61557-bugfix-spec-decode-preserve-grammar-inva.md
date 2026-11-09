# PR #61557: [Bugfix][Spec Decode] Preserve grammar-invalid drafts for rejection sampling

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yzong-rh（Co-author @tomasruizt，基于 tomasruizt/vllm#18） |
| 合并 | 2026-10-27 |
| 分类 | Sampler / Detokenization（tier: major） |
| Label | bug, structured-output, speculative-decoding, ready, ci/build, scheduler |
| Diff | diffstat 缺失（素材记录为 +None/-None）；diff 含 10 个文件（structured_output 管理器与 xgrammar/guidance 后端、scheduler、engine core、5 个测试文件、CI yaml），1 个文件被剔除 |
| 链接 | https://github.com/vllm-project/vllm/pull/61557 |

## 一句话结论

修复投机解码 + 结构化输出的输出分布偏移（issue #60451）：不再提前丢弃 grammar 无效 draft，交给 rejection sampling 处理，异步调度路径吞吐 +2–3%。

## 动机

spec decode + structured outputs 组合下，调度器在 draft 阶段就调用 `validate_tokens` 把 grammar 无效的 draft token 丢弃/截断，导致实际采样分布与不开投机解码时显著不同（issue #60451）。正确做法是保留无效 draft 交由 rejection sampling 拒绝，使 grammar 只通过 bitmask 约束采样分布。本卡是 #60690 的替代方案。

## 优化点

(1) `StructuredOutputManager.validate_tokens` 更名 `trim_spec_tokens`：只裁剪 padding 与 grammar 无效后缀，但**多保留一个 token**（`grammar_start + len_validated + 1`）——被拒绝 token 保留在序列中，作为 rejection sampling 的输入而非直接消失；grammar 已终止时只返回约束边界前缀。(2) `grammar_bitmask` 允许接收未校验 draft：逐位置填 mask 时遇到 -1（padding）或 `accept_tokens` 失败即置 `padding_or_grammar_invalid`，后续行全部填 unconstrained mask（`apply_bitmask=False`），不再复制失败行 mask； FSM 前进后统一 rollback。(3) scheduler 的 `update_draft_token_ids_in_output` 移除 `validate_tokens` 调用（async 路径），draft 槽位保留、无效数记入 `num_invalid_spec_tokens`；`update_draft_token_ids` 改调 `trim_spec_tokens`（sync 路径）。(4) 后端原子性：`accept_tokens` 失败时 grammar 状态不变——xgrammar 侧失败即 `rollback(num_accepted)` 并回退计数；guidance 侧改用 `ll_matcher.try_consume_tokens` + 失败 rollback（替代 `consume_tokens`），并修正 EOS 在 token 序列中间时的终止语义（EOS 后还有 token 则拒绝、EOS 恰好消费完则 terminated）。`backend_types.py` 抽象契约新增"A false result must leave the grammar state unchanged"。

## 关键改动

```text
vllm/v1/structured_output/__init__.py: validate_tokens→trim_spec_tokens（多保留 1 个无效 token）；grammar_bitmask 支持未校验 draft、失败后填 unconstrained mask
vllm/v1/core/sched/scheduler.py: async 路径删除 validate_tokens，保留 draft 槽位交给 rejection sampling
vllm/v1/structured_output/backend_xgrammar.py: accept_tokens 失败时 rollback 已接受 token（原子化）
vllm/v1/structured_output/backend_guidance.py: try_consume_tokens + rollback；EOS 中间位置语义修正
tests/v1/e2e/spec_decode/test_distribution_shift.py: 新增，4000 样本 chi2 检验分布一致性
```

无效后 mask 全部 unconstrained 的逻辑：

```python
if padding_or_grammar_invalid:
    self._fill_bitmasks(((grammar, base_index + i, False, ()),))
    continue
```

## 性能声明

测试条件：Qwen/Qwen3-4B + RedHatAI/Qwen3-4B-speculator.dspark、15 draft tokens、adaptive verification、单张 H100 80GB、`run.py --num-repeats 10`、95% CI。结果：async scheduling 路径 c96 提升——broad 约束 +3.41%（10,017.3 → 10,358.9 tok/s）、tight 约束 +2.32%（17,066.4 → 17,462.7 tok/s），c1 基本持平（−0.32%/−0.31%）；sync 路径小幅回退约 **2%**（broad c1 −2.09%、broad c96 −1.92%、tight c96 −2.72%），tight c1 持平。对照 #60690：本卡在 async/broad/c96（+3.41% vs −0.47%）与 async/tight/c96（+2.32% vs −1.20%）上明显更优，sync 路径两者相近。作者总结：async 路径约 2% 收益、sync 路径约 2% 回退。

## 验证与回归风险

验证：`tests/v1/e2e/spec_decode/test_distribution_shift.py`（dflash 概率性 draft、regex 约束、4000 样本、sync/async × standard/block rejection，chi2 p>1e-3 断言分布一致）9 passed；调度器/结构化输出/grammar-invalid 单测 501 passed；新增 `test_grammar_backends.py::test_accept_tokens_rejection_is_atomic` 对 xgrammar/guidance/outlines/lm-format-enforcer 四后端断言失败原子性（lm-format-enforcer strict xfail）；`test_structured_output_manager.py` 增加 trim 场景矩阵并把 bitmask 流测试改为直接喂未校验 draft。风险面：(1) `trim_spec_tokens` 多保留一个 token 的语义改变所有 spec+grammar 请求的 rejection sampling 输入，依赖 worker 侧 rejection 逻辑正确处理无效 token；(2) `accept_tokens` 失败必须严格无副作用，任何后端违反复原子性契约都会引入状态漂移；(3) sync 路径约 2% 回退是已知代价（保留无效 token 增加验证开销）；(4) guidance EOS 语义重写，jump decoding 仍为 TODO。

## 关联

问题：#60451（分布偏移）；架构总览：#62818；替代方案：#60690；相关：#14702（spec decode + structured outputs 的 bitmask 机制源头）。CI：dflash nightly 分片新增 test_distribution_shift.py（NVIDIA 与 AMD 镜像两处）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
