# PR #59506: [Bugfix][Frontend] Don't crash beam search on an extreme length_penalty

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#59506](https://github.com/vllm-project/vllm/pull/59506) |
| 作者 | khluu |
| 合并日期 | 2026-11-17 |
| labels | bug, frontend, ready |
| 热度 | 👍 0 · 💬 15 |
| 分类 | Sampler / Detokenization |
| bug 类型 | crash |

## 一句话结论
beam search 请求带极大 `length_penalty` 时,`get_beam_search_score` 计算分母 `max(seq_len, 1) ** length_penalty` 会下溢为 0.0(负值)触发 ZeroDivisionError,或上溢(正值)触发 OverflowError,且都发生在生成完成后的 beam 排序阶段,把整个请求变成 500;修复改为饱和处理并对非有限值在参数校验层直接拒绝(400)。

## 问题现象
带大 `length_penalty` 的 beam search 请求返回 500 错误。该问题由 CI 上的 schemathesis 测试(`test_openapi_stateless[POST /v1/chat/completions]`,build 91958)在 `main` 上发现,调用栈:

```
vllm/entrypoints/generate/beam_search/online.py:196 beam_search
  -> utils.py:159 sort_beams_key -> utils.py:154 get_beam_search_score
ZeroDivisionError: float division by zero
```

崩溃发生在 beam 排序阶段(`sort_beams_key`),此时生成已经跑完,请求白白浪费算力后仍以错误告终。

## 根因
正文披露根因:`get_beam_search_score` 以 `cumulative_logprob / (max(seq_len, 1) ** length_penalty)` 计算分数,分母的幂运算对极大指数没有防护:

- 极大负值 `length_penalty` 使 `max(seq_len, 1) ** length_penalty` 下溢为 `0.0`,除零抛 `ZeroDivisionError`;
- 极大正值使幂运算抛 `OverflowError`。

#56249 加的 `max(seq_len, 1)` 保护只覆盖长度为 0 的情况,不覆盖极端指数。作者检索过开放 PR 中的 `length_penalty`、`get_beam_search_score` 与 beam-search ZeroDivisionError,确认不是重复修复。

## 修复方案
两层防护:

1. **饱和代替抛错**(`vllm/entrypoints/generate/beam_search/utils.py`):分母上溢取 `math.inf`(分数趋近 0),下溢为 0 时按 logprob 符号给出 `-math.inf` 或 `0.0`;只要分数有限,beam 排序仍然正确:

```python
try:
    denominator = max(num_generated_tokens, 1) ** length_penalty
except OverflowError:
    denominator = math.inf
if denominator == 0.0:
    return -math.inf if cumulative_logprob < 0 else 0.0
return cumulative_logprob / denominator
```

2. **参数入口拒绝非有限值**(`vllm/sampling_params.py`):`BeamSearchParams.__post_init__` 在既有 `beam_width` 校验旁增加 `math.isfinite(self.length_penalty)` 检查,`inf`/`nan` 抛 `VLLMValidationError`,API 返回 400(`length_penalty must be finite`)。

测试:`tests/test_request_input_bounds.py` 新增 4 个用例(`length_penalty` 取 `-1e4`/`1e4` 验证分数单调,`inf`/`nan` 验证参数校验),在 `h200-0` CI 镜像(main `deb148e7`)上 43 passed,4 个新用例在改动前均失败;`pre-commit`(ruff、mypy-3.10)通过。

## 影响与回归风险
- 行为变化:极端 `length_penalty` 下请求不再 500,要么在参数层被拒(400),要么返回饱和分数排序后的 beams;有限值路径完全不变。
- 饱和语义:上溢时分数约 0、下溢时 `-inf`(logprob ≥ 0 时为 0.0),beam 顺序在分数有限的所有位置仍由分数决定;极端值下的排序结果语义上属于"尽力而为",调用方不应依赖极端 `length_penalty` 的精确排序。
- 改动面小:仅 `beam_search/utils.py`、`sampling_params.py` 与测试文件,不触碰生成主路径。
- 备注(正文):该改动与测试由 Claude Code 从 CI 失败日志生成,@khluu 在合并前需逐行 review。

## 关联
- #56249(既有的 `max(seq_len, 1)` 零长度保护,本 PR 补其未覆盖的极端指数场景)。
- 涉及文件:`vllm/entrypoints/generate/beam_search/utils.py`、`vllm/sampling_params.py`、`tests/test_request_input_bounds.py`。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
