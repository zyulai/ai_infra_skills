# PR #44297: [Bugfix][Structured Output][Spec Decode] Constrain bitmask and trim grammar advance at the reasoning boundary

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#44297](https://github.com/vllm-project/vllm/pull/44297) |
| 作者 | yuyue0225sc |
| 合并日期 | 2026-07-04 |
| labels | bug, structured-output, speculative-decoding, ready, v1, verified |
| 热度 | 👍 6 · 💬 36 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
MTP/EAGLE + 严格 tool calling 下,`</think>`(token 248069)落 spec 窗口中段导致 bitmask 缺约束/FSM 推进含推理内容,请求 500;四处修复。

## 问题现象
`VLLM_ENFORCE_STRICT_TOOL_CALLING=1` 与 MTP/EAGLE 投机解码组合时,引擎间歇性拒绝 token 248069(Qwen 的 `</think>`),请求以 HTTP 500 结束:`Failed to advance FSM ... grammar rejected tokens`。复现:2×H20-96GB TP=2、Qwen/Qwen3.5-35B-A3B + MTP(k=1)+ qwen3 reasoning parser + qwen3_coder tool parser,同一请求 50 次,upstream/main 基线 21 成功 / 29 次 HTTP 500(58% 失败率)。

## 根因
正文披露 `StructuredOutputManager.grammar_bitmask` 及调度器侧的四个缺陷:
1. **窗口中段的 reasoning-end 丢失 bitmask 切换**:`should_fill_bitmask` 反映的是 step **开始**时的 `reasoning_ended`;MTP/EAGLE 一步调度多个位置时 `</think>` 可落在 spec 窗口中间,整窗沿用步首的 `apply_bitmask`,标记之后的位置仍拿到无约束的 in-reasoning 掩码,模型可吐出 `structural_tag.excludes` 中的 token(如内容开始后又出现 `</think>`),下一步 `accept_tokens` 拒绝即杀请求。
2. **bonus 行从 `-1` padding 继承陈旧 `apply_bitmask`**:异步 spec-decode 路径把无效 draft 槽 pad 成 `-1`,旧循环对首个 `-1` 置 `apply_bitmask=False`,尾部的 `-1`(bonus 槽)因此继承 `False`,bonus 行被无约束掩码填充。
3. **窗口中段推进遇到语法非法 draft 崩溃**:修复 1 引入新路径——marker 之后的 draft 在语法激活前产生、不保证合法;strict-start 语法(`json_object`)会拒绝首个 post-marker draft,旧代码 raise `AssertionError`;而完全跳过推进又把 bitmask 冻在语法初始态,开头 token 重复(如 `{{`)。
4. **#42452 引入的调度器侧回归**:#42452 使 structural tags + 投机解码下 `should_advance` 在边界步返回 True、调度器把该步 token 直接喂 `grammar.accept_tokens`,但其中仍含到 marker 为止的推理内容(`[198, 248069]` = `"\n</think>"`),structural-tag 语法排除 marker,拒绝即 FINISHED_ERROR;pre-#42452 基座上该路径不可达。

## 修复方案
1. 对窗口内每个 draft 在运行前缀上调 `reasoner.is_reasoning_end_streaming(...)`;观察到 marker 即翻转 `apply_bitmask=True`,同步后续位置(含 bonus 行)全部受语法约束;marker 本身是推理内容,该位置跳过语法推进。
2. bonus 行改为 `bonus_apply = should_fill_bitmask(request) or apply_bitmask`,`-1` padding 不再能把 bonus 行翻成无约束。
3. 仍尝试推进 post-marker draft 以反映推进后状态,但容忍 `accept_tokens` 拒绝而非 assert;窗口内的推进最后 `grammar.rollback(state_advancements)` 回滚,持久语法态不变。
4. `should_advance` 在边界步记录 marker 的绝对索引(`reasoning_end_token_index`,`_find_reasoning_end_index`),新增 `trim_reasoning_for_advance` 在调度器推进前丢掉 marker 及其之前的内容;整步均为推理内容则跳过推进。调度器 `update_from_output` 改为对 trim 后的 token 调 `accept_tokens`,空列表短路。关键 diff 摘录(vllm/v1/structured_output/__init__.py):
```python
+                            apply_bitmask = True
+                            advance_grammar = False
+                            post_reasoning_end_in_window = True
...
-                        assert accepted, (token, req_id, scheduled_spec_decode_tokens)
+                        if accepted:
+                            state_advancements += 1
+                        elif not post_reasoning_end_in_window:
+                            raise AssertionError(
+                                (token, req_id, scheduled_spec_decode_tokens)
+                            )
```

## 影响与回归风险
- 影响窗口:正文未列具体版本范围;影响 spec-decode(MTP/EAGLE)+ 结构化输出/strict tool calling 的组合,rebase 进 #42452 后的 main 亦然。
- 回归面:`vllm/v1/structured_output/__init__.py`(grammar_bitmask 主循环重写、should_advance、新增 trim)、`structured_output/request.py`(新字段)、`v1/core/sched/scheduler.py`(推进前 trim)。xgrammar 与 guidance 两个后端均覆盖;非 spec-decode 或非 reasoning 场景走原路径。测试:新增 `test_mtp_structured_output.py` 16 passed(7 用例 × 2 后端 + 2 个 manager 级);`tests/v1/structured_output/` 20 passed;rejection-sampler/backup-token 子集 34 passed。E2E 50 试:本 PR 50/50 0%;rebase 后无 fix 4 为 48% 失败、有 fix 4 为 0%(失败签名复现 `grammar rejected tokens [198, 248069]`)。

## 关联
- #44006(所修 issue)
- #25515(引入 `reasoning_ended` 机制)
- #42452(should_advance 同步推进 structural tags;本 PR 补齐其 bitmask 缺口并 trim 推理内容)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
