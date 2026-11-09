# RFC #35213: [RFC]: Disaggregate RendererClient from EngineClient as Foundation for Disaggregated Frontend

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#35213](https://github.com/vllm-project/vllm/issues/35213) |
| 作者 | sagearc |
| 创建日期 | 2026-02-24 |
| 状态 | ⚪ closed(关闭于 2026-03-08) |
| labels | RFC |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Disaggregated Serving |

## 一句话结论
提议把仅 CPU 的渲染关注点从 EngineClient 拆出,落地为 OpenAIServingRender 与 `vllm launch render`,作为分离式前端的基础。

## 提案动机
`EngineClient`(protocol.py)是单一 ABC,把不需要推理引擎的关注点(tokenization、config、health checks)与需要的关注点(inference、profiling、cache management、scheduler control、weight transfer)混在一起;任何只需 CPU-only 特性的客户端都被迫实现或 stub 全部 GPU 方法。这一需求在 #34407 的讨论中浮出。正文论证现状已显示拆分形状:`OpenAIServing.__init__`(所有 serving 的基类)只提取 CPU-only 字段(model_config、renderer、io_processor、input_processor);GPU 方法(generate、encode)只被特定推理辅助方法(beam_search、_prepare_generators、_generate_with_builtin_tools)调用,且这些方法仅由推理子类(OpenAIServingChat、OpenAIServingCompletion)触发;不做推理的 serving 类(如 OpenAIServingTokenization)只需 CPU-only 组。

## 提出的方案
- **实际采用(实现于 #36166)**:不做完整 protocol 拆分,改为自包含的预处理类——`OpenAIServingRender`:镜像 chat/completion serving 路径,但返回渲染后的输入(chat template 渲染、tokenization、tool/reasoning parsing),不依赖任何推理引擎。
- 新 CLI 命令 `vllm launch render`:启动纯 CPU server,由 `OpenAIServingRender` 支撑。
- `init_render_app_state()`:从 `VllmConfig` 引导 renderer、io_processor 与 `OpenAIServingRender`,全程无 engine client。
- 移除 `LaunchEngineClient`(#34551 引入的 null-object stub);render 路径不再依赖 `EngineClient`。
- **原始提案(已推迟)**:在 `Client` ABC(is_running、is_stopped、errored、dead_error、check_health)之下分出 `RendererClient`(vllm_config、model_config、renderer、io_processor、input_processor、is_tracing_enabled)与 `EngineClient`(do_log_stats、get_supported_tasks、generate、encode、abort、add_lora/remove_lora、start_profile/stop_profile)两支。
- 使能的场景:`vllm launch render --model ...` 后经 `/v1/chat/completions/render` 端点调用;未来分离式部署中,Renderer 节点(CPU)跑 render server、处理 HTTP、tokenization 与输入输出处理,Engine 节点(GPU)做纯 token-in/token-out 推理。
- 正文遗留的开放问题:更好的命名?

## 优化点与收益
正文未给出量化数字。定性收益:仅 CPU 特性的消费者不再被迫实现或 stub GPU 方法;render 路径对 `EngineClient` 的依赖降为零;为「CPU renderer 节点 + GPU engine 节点」的分离式部署前端铺路。

## 现状与落地
已关闭(关闭于 2026-03-08)。正文写明 "Approach Taken (implemented in #36166)":渲染路径部分已按自包含类方案实现(LaunchEngineClient 移除、render 路径脱离 EngineClient);而 RendererClient/EngineClient protocol 层级的原始提案被明确标记为 deferred。protocol 拆分的最终去向需查关联 PR/后续讨论。

## 关联
- 实现 PR: https://github.com/vllm-project/vllm/pull/36166
- 需求来源讨论: https://github.com/vllm-project/vllm/issues/34407
- 相关: https://github.com/vllm-project/vllm/issues/22817
- 相关 PR: https://github.com/vllm-project/vllm/pull/33000 与 https://github.com/vllm-project/vllm/pull/33693
- LaunchEngineClient 出处: https://github.com/vllm-project/vllm/pull/34551

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
