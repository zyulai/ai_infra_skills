# PR #54442: [Bugfix] Don't let a structured-output request sample from an unmasked row

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54442](https://github.com/vllm-project/vllm/pull/54442) |
| 作者 | ArcheyChen |
| 合并日期 | 2026-10-01 |
| labels | bug, speculative-decoding, ready, mrv2, scheduler |
| 热度 | 👍 2 · 💬 29 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
投机解码下 draft 丢失时,grammar bitmask 遇 `-1` 占位后全用 `_full_mask`,接受过程可走进无约束行、输出语法不允许的 token;本 PR 做 fail-closed:把超限 draft 置为无效。

## 问题现象
结构化输出 + 投机解码时,请求会采样出 grammar 从未允许的 token。生产环境(2 节点 × 2 GPU,`pipeline_parallel_size=2`、`tensor_parallel_size=2`,DeepSeek-V4-Flash + dspark K=7、xgrammar):只要有第二个结构化输出请求驻留,`tool_choice=required` 与 `json_schema` 请求 100% 失败;修复后 72/72 并发 grammar 请求全部完成、0 次 `grammar rejected`、0 引擎错误。另有用户独立报告此 PR 修复了 `pipeline_parallel_size=2` × `tensor_parallel_size=8`、DSpark(K=3)、16×B200 的部署。端到端复现脚本在 main 上失败(`emitted [7, 7, 99, 7]`,99 是语法不允许的 token),本分支通过。

## 根因
PR 正文披露:`StructuredOutputManager.grammar_bitmask` 只在 `apply_bitmask` 成立时约束对应行,调度 draft 窗口里出现 `-1` 占位即置 `seen_padding`,其后每一行都填 `_full_mask`(全词表允许)。当 `-1` 表示"`validate_tokens` 拒绝了这个 draft"时是安全的(首个 `-1` 恰在被拒绝的 draft 上,该行仍被 mask);但当 `-1` 表示"draft 从未到达 scheduler"时不安全——`DraftTokensHandler` 只有一个被每次 `execute_model` 覆盖的槽位,且对无结构化输出请求的批次会整体清空 payload;而 worker 用自己 GPU 上的 `req_states.draft_tokens` 验证真实 draft,接受过程于是可以直接走进约束为空的行。

## 修复方案
fail-closed 的半边修复(不解决 hand-off 身份问题):为每个结构化输出请求携带 bitmask 实际约束了多少个前导 draft,并从该索引起把 draft 置为无效,使 `is_valid_draft = draft_sampled >= 0` 在采样走进宽松行之前钉住接受长度。共 6 个文件 +85/-3:
1. `scheduler.py` `get_grammar_bitmask`:用 `len(strip_speculative_padding(window))`(复用 #48200 引入的 helper)填 `num_acceptable_drafts`;
2. `sched/output.py`:`GrammarOutput.num_acceptable_drafts` 带默认值字段;
3. `structured_outputs.py` 新增 `grammar_invalid_drafts`:在 device 上按 `expanded_local_pos`(来自 device `cu_num_logits`,adaptive verification 裁剪 draft 时仍正确)对超过请求限额的 draft 生成 mask;全部达标时返回 None、零 launch;
4. `model_runner.py` `sample()`:应用 bitmask 后调用并把 mask 交给 rejection sampler;
5. `rejection_sampler.py`:验证改用 `draft_sampled.masked_fill(invalid_drafts, -1)`,`apply_sampling_params` 与 watermarking 仍用真实 draft id;
6. `batch_shard.py`:按 bitmask 同样方式分片新字段。
无 kernel 改动:`rejection_sampler_utils` 对 `-1` 位置本就直接从 target 分布重采样(而非 `max(p-q,0)` 残留)。

```diff
+        verify_draft_sampled = draft_sampled
+        if invalid_drafts is not None:
+            verify_draft_sampled = draft_sampled.masked_fill(invalid_drafts, -1)
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围(问题链见 #54437,含 main 上的 model-free 复现)。
- 回归面:投机解码 + 结构化输出的采样路径。draft 全部回填(正常情况)时无任何失效、行为不变;字段缺失(旧 scheduler、warmup)时整个窗口失效,是保守选择;仅验证环节看到 masked 副本,penalties/bad-words 不受影响。新增 `tests/v1/worker/test_grammar_invalid_drafts.py`(5 用例);Test Plan 套件 71 passed, 1 xfailed;gsm8k(RTX 4090、Qwen3-8B + eagle3 K=4)0.925 ± 0.0187 与 main 持平,确认投机解码未受影响。

## 关联
- Issue #54437(完整问题链、main 复现与实测数据)
- PR #56802(#54437 的另一半:hand-off 身份修复,与本 PR 互补)
- PR #48200(strip_speculative_padding helper 来源)
- [用户独立验证评论](https://github.com/vllm-project/vllm/pull/54442#issuecomment-5822629259)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
