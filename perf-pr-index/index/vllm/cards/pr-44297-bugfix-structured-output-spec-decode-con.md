# PR #44297: [Bugfix][Structured Output][Spec Decode] Constrain bitmask and trim grammar advance at the reasoning boundary

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yuyue0225sc |
| 合并 | 2026-07-04 |
| 分类 | Speculative Decoding（tier: major） |
| Label | bug, structured-output, speculative-decoding, ready, v1, verified |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/44297 |

## 一句话结论

修复严格 tool calling + MTP/EAGLE 下 `</think>` 触发的 FSM reject，端到端 HTTP 500 失败率 58%→0%。

## 动机

Fixes #44006：`VLLM_ENFORCE_STRICT_TOOL_CALLING=1` 与 MTP/EAGLE speculative decoding 组合时，引擎间歇拒绝 token 248069（Qwen 的 `</think>`），请求以 500 "Failed to advance FSM ... grammar rejected tokens" 结束。根因是 `StructuredOutputManager.grammar_bitmask` 的缺陷，以及 rebase 到含 #42452 的 main 后所需的 scheduler 侧修复。

## 优化点

四处修复：1) MTP/EAGLE 每 step 调度多位置，`</think>` 可落在 spec window 中部，旧代码整窗沿用 step 起点的 `apply_bitmask`；改为对窗口内每个 draft token 在 running prefix 上调 `reasoner.is_reasoning_end_streaming`，一旦观察到 marker 就翻 `apply_bitmask=True`，使后续位置（含 bonus row）受 grammar 约束，且 marker 本身跳过 grammar advance。2) async spec-decode 路径用 `-1` 填充无效 draft 槽，bonus row 会继承 `-1` 带来的 `apply_bitmask=False`；改为 `should_fill_bitmask(request) or apply_bitmask`。3) 对 post-marker drafts 仍尝试 `accept_tokens` 推进以反映 advanced 状态，但容忍拒绝（不再 assert），窗口内推进在末尾回滚。4) #42452 让 `should_advance` 在 reasoning-boundary 返回 True，scheduler 把含 reasoning 内容（如 `[198, 248069]`）的 token 直接喂给 `grammar.accept_tokens` 而死；改为记录 marker 绝对索引并在 advance 前 `trim_reasoning_for_advance` 丢弃 marker 及其之前内容。

## 关键改动

```text
vllm/v1/structured_output/__init__.py: grammar_bitmask 逐位置检测 reasoning-end 并约束 bitmask
vllm/v1/core/sched/scheduler.py: 边界步记录 marker 索引、advance 前裁剪 reasoning 内容
vllm/v1/structured_output/request.py: should_fill_bitmask / trim_reasoning_for_advance 相关状态
tests/v1/spec_decode/test_mtp_structured_output.py（新增）: 四个缺陷的回归用例
```

```python
+                detect_reasoning_end = (
+                    not apply_bitmask
+                    and reasoner is not None
+                    and not self.enable_in_reasoning
+                )
+                for i, token in enumerate(req_tokens):
+                    if reasoner.is_reasoning_end_streaming(simulated, [token]):
+                        # Reasoning ended mid-window. Constrain the rest
+                        # of the window via bitmask. Skip grammar advance
+                        apply_bitmask = True
+                        advance_grammar = False
```

## 性能声明

PR 正文未给出定量性能数据（本 PR 为正确性 bugfix，正文给出的是失败率等验证数据，见下节）。

## 验证与回归风险

单测：新增 `tests/v1/spec_decode/test_mtp_structured_output.py` 16 passed（`-1` padding 的 N+1 行 bitmask、bonus row 保持约束、mid-window reasoning-end、post-marker 拒绝不 assert、`trim_reasoning_for_advance` 等）；`tests/v1/structured_output/` 20 passed、spec-decode rejection-sampler/backup-token 子集 34 passed。端到端复现（2× H20-96GB、TP=2、Qwen/Qwen3.5-35B-A3B + MTP(k=1) + `--reasoning-parser qwen3` + `VLLM_ENFORCE_STRICT_TOOL_CALLING=1`，50 次请求）：upstream/main 21×200 OK / 29×500（58% 失败率），本 PR 50/50（0%）；rebase 后无 fix 4 为 26/24（48%），带 fix 4 为 50/0。风险面：涉及 strict-start grammar（`json_object`）、structural_tag、`n>1`、bonus row、`-1` padding 与 rebase 后 #42452 的 `should_advance` 交互。

## 关联

PR 正文提及 Fixes #44006；#25515 引入 `reasoning_ended` 机制；#42452（已合并）修复 structural tags 的 FSM advancement，本 PR 为其互补。素材未列其他关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
