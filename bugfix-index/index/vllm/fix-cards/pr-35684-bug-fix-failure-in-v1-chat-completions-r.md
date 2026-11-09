# PR #35684: [Bug] Fix Failure in /v1/chat/completions/render for Multimodal Requests (https://github.com/vllm-project/vllm/issues/35665)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#35684](https://github.com/vllm-project/vllm/pull/35684) |
| 作者 | sergey-zinchenko |
| 合并日期 | 2026-03-14 |
| labels | bug, cpu, frontend, ready, v1 |
| 热度 | 👍 3 · 💬 27 |
| 分类 | Frontend / Serving |
| bug 类型 | general |

## 一句话结论
多模态请求打 /v1/chat/completions/render 时内部渲染产物无法 JSON 序列化触发 500;重构端点返回可序列化的 token-in GenerateRequest(含 MM features)修复。

## 问题现象
对多模态输入(base64 image_url + 文本)调用 `POST /v1/chat/completions/render` 时,端点返回内部 render artifacts,在 JSON 序列化时可能触发 HTTP 500(正文表述 "could trigger HTTP 500 on serialization for multimodal inputs")。影响 OpenAI render 预处理端点的多模态请求。

## 根因
正文披露:该端点此前直接返回内部 render artifacts 而非 JSON 可序列化对象,多模态输入下序列化失败导致 500;同时 `sampling_params` 无法干净地随请求透传。修复方向是让端点返回 RFC 对齐(issue #22817)的、JSON 可序列化的 token-in `GenerateRequest`。

## 修复方案
- `vllm/entrypoints/serve/disagg/protocol.py`:`GenerateRequest` 移到此处,`features` 从 `str | None` 改为新的 `MultiModalFeatures`(含 `mm_hashes` 与 `mm_placeholders`/`PlaceholderRangeInfo{offset,length}`),并加 token_ids 非负校验。
- `vllm/entrypoints/serve/render/serving.py`:`render_chat_request`/`render_completion_request` 改为构建并返回 `GenerateRequest`(带 request_id、token_ids、features、sampling_params、stream 等),新增 `_extract_mm_features` 提取多模态哈希与占位区间。
- `vllm/entrypoints/serve/render/api_router.py`:response_model 改为 `GenerateRequest` / `list[GenerateRequest]`,返回 `model_dump()`。
- `vllm/v1/serial_utils.py`:`PydanticMsgspecMixin` 增加序列化路径,剥除 `_` 前缀及显式排除字段,保证 msgspec.Struct 可 JSON 安全输出。
- `vllm/entrypoints/openai/server_utils.py`:engine 错误处理改为返回带错误信息的 JSONResponse。
- 新增 `tests/entrypoints/openai/cpu/test_render_multimodal.py`(base64 图片 + 文本、stream=True + stream_options)等测试。

注:pack 中 diff 在 40000 字符处被截断(文件粒度),截断部分未知。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:`/v1/chat/completions/render` 与 `/v1/completions/render` 的响应结构发生变化(由 [conversation, engine_prompts] / prompt 字典改为 GenerateRequest JSON),依赖旧响应格式的下游调用方需适配;engine 错误处理器响应体也有变化;`/tokenize` 与 render 的默认 chat template kwargs 透传行为变化(由 diff 推断)。

## 关联
- https://github.com/vllm-project/vllm/issues/35665
- https://github.com/vllm-project/vllm/issues/22817

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
