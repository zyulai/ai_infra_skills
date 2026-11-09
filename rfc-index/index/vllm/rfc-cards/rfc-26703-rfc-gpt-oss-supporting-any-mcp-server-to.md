# RFC #26703: [RFC]: [gpt-oss] Supporting Any MCP Server Tools

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#26703](https://github.com/vllm-project/vllm/issues/26703) |
| 作者 | alecsolder |
| 创建日期 | 2025-10-13 |
| 状态 | ⚪ closed(关闭于 2026-02-16) |
| labels | RFC, stale |
| 热度 | 👍 11 · 💬 5 |
| 分类 | Feature / Ecosystem |

## 一句话结论
以 MCP namespace 为核心统一 gpt-oss 工具集成,支持任意 MCP server 并清理模型专用代码路径。

## 提案动机
正文列出三点动机:1) vLLM 应支持与所有 MCP server 集成(TODO 中提到需用 GitHub 等企业级 MCP server 测试);2) 清理 vLLM 中 gpt-oss 工具专用代码路径——工具与模型会变,但 MCP 是面向所有模型的协议,此举为所有工具调用模型的完整 MCP 集成铺路;3) 避免因 OpenAI 输出类型有损而丢失信息:正文举例,ResponsesAPI 上某工具调用只能向用户返回 `type`/`url` 两个字段,而模型实际的 tool call 接口还带 `id`、`cursor`、`loc`、`num_lines`、`view_source`、`source` 等参数。正文称该特性类似 "connectors" 概念:尚不支持用户通过 URL 指定任意 MCP server,列表是静态的、由服务器持有,更接近 OpenAI MCP Connectors 的模式。

## 提出的方案
- **Tool Namespaces**:对 gpt-oss,namespace 代表一组工具(OpenAI Harmony 中部分 namespace 会被特殊处理);MCP server 拥有一组工具、由其 `serverInfo.name` 标识;ResponsesAPI 的 MCP Tool 类型用 `server_label` 字段标识 MCP server。代码库中这三个本是不同概念,将其作为外键(foreign key)统一可简化代码且保留所需功能——MCP server 与 Responses API 是通用的,同一 MCP server 应当既能用于 DeepSeek 也能用于 gpt-oss。
- **特殊行为按 namespace 组织**:随着工具 RL 流行,更多模型会对工具产生特殊处理,同一模型的不同微调也可能不同,因此把这类特殊处理从硬编码路径改为基于 namespace。需要特殊处理的 namespace 由 server/model 定义,正文称其为 "elevated"(并注明需要个新名字)。当前 vLLM 有特殊代码把 container namespace 视为 elevated;改后一个部署可以用环境变量提升某 namespace,另一个部署则无需如此。
- **围绕 MCP 标准化**:在保持与 OpenAI 内置工具类型(如 CodeInterpreter)向后兼容的同时,把所有工具类型映射为 MCP 工具;vLLM 自身不应负责执行工具(demo 除外),所有工具都将在 MCP server 上执行。由此用户可以对哪些 `server_label` 被视为 elevated 形成预期,代码也不再需要为不同 OpenAI 工具类型写特殊分支。
- **后续步骤**:所有 tool call 与输出统一采用 `McpCall` 类型,进一步简化逻辑,并允许在 ResponsesAPI 输入中用 `server_label` 处理 elevated 工具调用/输出。另注:若假设工具调用的 reasoning 消息与其后紧随的调用处于同一 channel(OpenAI 在其 Responses API 实现中如此假设),则存在一条仅用 Responses types(无需 Harmony 元数据)对 gpt-oss 无损使用 ResponsesAPI 的路径。
- **用法示例**(正文给出):运行 `tests/entrypoints/openai/memory_mcp_server.py` 启动 Memory MCP server(默认端口 8765);以 `vllm serve openai/gpt-oss-20b --tool-server=localhost:8765` 注册;然后向 `/v1/responses` 发送含 `{"type": "mcp", "server_label": "memory", "server_url": "ignored"}` 的请求并开启 `enable_response_messages`。

## 优化点与收益
正文未给出量化数字。收益为架构与生态层面:统一 MCP 路径消除 gpt-oss 专用代码与 OpenAI 工具类型特殊分支,支持任意 MCP server 接入,并解决 OpenAI 有损类型导致的工具调用信息丢失。

## 现状与落地
已关闭(2026-02-16),labels 含 stale;正文未写明是否已实现或被替代,落地情况需查关联 PR。

## 关联
- [OpenAI MCP Connectors 文档](https://platform.openai.com/docs/guides/tools-connectors-mcp#quickstart)(正文类比的 connector 模式)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
