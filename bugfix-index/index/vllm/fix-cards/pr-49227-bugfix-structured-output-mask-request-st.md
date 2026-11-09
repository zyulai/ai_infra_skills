# PR #49227: [Bugfix][Structured Output] Mask request stop tokens in xgrammar until grammar terminates

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49227](https://github.com/vllm-project/vllm/pull/49227) |
| 作者 | sfeng33 |
| 合并日期 | 2026-08-10 |
| labels | bug, structured-output, needs-rebase, v1 |
| 热度 | 👍 6 · 💬 4 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
xgrammar 只认识 tokenizer 单个 eos,请求其余 stop token 在 FSM 非终止态逃出 bitmask,结构化输出被截成非法 JSON;把 `all_stop_token_ids` 经 `override_stop_tokens` 交给 GrammarMatcher 掩蔽到语法终止。

## 问题现象
使用结构化输出(如 JSON schema)时,模型可能在 grammar FSM 仍处于对象中间态时采到 stop token,把生成截断成非法 JSON。Gemma 上该 token 是 `<end_of_turn>`(id 106)。复现(正文测试计划):`vllm serve google/gemma-4-31B-it --structured-outputs-config '{"backend":"xgrammar"}'`,用 logit_bias={"106": 100} + greedy 强推 106——修复前输出 `{"name": "` 即被截断(stop_reason=106),非合法 JSON;修复后输出完整 `{"name": "Elara Vance", "bio": "..."}` 合法 JSON(stop_reason=106)。

## 根因
正文披露:xgrammar 只知道 tokenizer 的**单个** eos,而请求真实的 stop 集合是 `generation_config` 的 eos 列表**加上**用户 `stop_token_ids`。Gemma-4 上这一差距具体存在:`generation_config` eos 为 `[1, 106, 50]`,tokenizer eos 只有 `<eos>`(1)。token 106/50 对 grammar 不可见,在非终止状态下逃出 bitmask。

## 修复方案
把请求的 `sampling_params.all_stop_token_ids` 一路透传到 xgrammar 原生 `GrammarMatcher(override_stop_tokens=...)`,由 xgrammar 把这些 stop token 门控到 grammar 终止态——对象中间态被掩蔽,schema 满足后才允许:

- `vllm/v1/structured_output/__init__.py` `_create_grammar`:`stop_token_ids = request.sampling_params.all_stop_token_ids`(sampling_params 非空时),传给 `compile_grammar(request_type, grammar_spec, stop_token_ids=...)`;
- `backend_types.py` 抽象签名与 `backend_guidance.py`、`backend_lm_format_enforcer.py`、`backend_outlines.py`、`backend_xgrammar.py` 的 `compile_grammar` 均加 `stop_token_ids: set[int] | None = None` 参数;
- `backend_xgrammar.py` 实际生效:

```diff
             matcher=xgr.GrammarMatcher(
                 ctx,
+                override_stop_tokens=list(stop_token_ids) if stop_token_ids else None,
                 max_rollback_tokens=self.num_speculative_tokens,
             ),
```

新增回归测试 `tests/v1/structured_output/test_backend_xgrammar_stop_tokens.py`:中途掩蔽注册的 stop token、语法可终止时放行、tokenizer 自身 eos 的默认终止行为保持不变。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:v1 结构化输出全部后端的 `compile_grammar` 签名(guidance/lm_format_enforcer/outlines 仅加参数,行为不变);xgrammar 后端在请求含额外 stop token 时 bitmask 行为改变(非终止态多掩蔽这些 token,终止态放行),tokenizer eos 语义不变;无 stop token 覆盖时传 None,行为与之前一致。

## 关联
- Fixes https://github.com/vllm-project/vllm/issues/42403

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
