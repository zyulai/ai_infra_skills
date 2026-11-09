# PR #57990: [Bugfix] [Frontend] Preserve prompt_logprobs in /v1/completions/derender

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57990](https://github.com/vllm-project/vllm/pull/57990) |
| 作者 | akshay-babbar |
| 合并日期 | 2026-10-29 |
| labels | bug, frontend, ready |
| 热度 | 👍 0 · 💬 26 |
| 分类 | Frontend / Serving |
| bug 类型 | general |

## 一句话结论
非流式 /v1/completions/derender 丢弃引擎已算好的 prompt_logprobs 返回 null;组装 choice 时补挂该字段修复。

## 问题现象
非流式 `/v1/completions/derender` 响应中 `prompt_logprobs` 恒为 `null`——引擎其实已经计算并返回了该数据,但 derender 组装响应时丢弃。chat derender 路径与直接调用 `/v1/completions` 均能正常透传,仅 completion derender 受影响。正文未附具体请求/响应示例。Fixes #57929。

## 根因
正文披露:`OnlineDerenderer._derender_completion` 构建每个 choice 时没有带上 `prompt_logprobs` 字段,把引擎已计算好的数据丢掉了;chat 路径与直接 completions 端点原本就会透传,故仅此路径受影响。

## 修复方案
- `vllm/renderers/online_derenderer.py`:
  - `_derender_completion` 中调用新增的 `_resolve_prompt_logprobs(gen.prompt_logprobs, tokenizer)` 并把它挂到该响应的每个 choice 上(per-prompt 语义与直接端点一致)。
  - `_derender_chat` / `derender_chat` 返回签名改为 `tuple[list[ChatCompletionResponseChoice], PromptLogprobs | None]`,chat 的 `prompt_logprobs` 在解码文本的同一后台调用里填充;`_resolve_prompt_logprobs` 会为 `--tokens-only` 生成服务(无 tokenizer)返回的缺失 `decoded_token` 补解码,含 U+FFFD 修正,镜像 `v1/engine/logprobs.py` 的 `LogprobsProcessor._update_prompt_logprobs`。
- `vllm/entrypoints/scale_out/derender/serving.py`:`derender_chat_response` 改用返回的 `prompt_logprobs`(原来直接取 `gen.prompt_logprobs`)。
- 测试:新增 `test_derender_completion_prompt_logprobs_passthrough`(未修复时失败、修复后通过)与 `test_derender_chat_prompt_logprobs_decoded_token`;本地 37/37 通过;正文声明不影响模型行为(仅前端响应组装)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:derender completion/chat 响应组装与 `_derender_chat` 返回签名(所有调用方已同步更新);`_resolve_prompt_logprobs` 新解码逻辑影响 tokens-only 场景的 decoded_token 填充;不影响模型推理。

## 关联
- Fixes [#57929](https://github.com/vllm-project/vllm/issues/57929)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
