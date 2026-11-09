# RFC #11162: [RFC]: Hardware pluggable

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#11162](https://github.com/vllm-project/vllm/issues/11162) |
| 作者 | wangxiyuan |
| 创建日期 | 2024-12-13 |
| 状态 | ⚪ closed (2025-06-10) |
| labels | RFC, stale |
| 热度 | 👍 31 · 💬 3 |
| 分类 | Hardware / Backend |

## 一句话结论
提出硬件可插拔架构:后端(worker/runner/attention 等)经 Platform 插件注册进 vLLM,让硬件后端与主库解耦。

## 提案动机
vLLM 已支持众多硬件后端(cpu、cuda、hpu、neuron、openvino、rocm、tpu、xpu),ascend、IBM Spyre 等也在寻求接入;后端越来越多带来问题:每个后端有自己的 executor、worker、runner、attention,后端特定代码散落各处使代码复杂;社区难以让后端持续可用(需要完整 CI 覆盖、维护者持续贡献);新特性也因后端情况复杂而难以加入。

## 提出的方案
- 站在两个已有 RFC 之上:#7131(已完成,通用插件系统)与 #9268(进行中,后端代码模块化解耦)。
- **插件流程**:vLLM 维护后端列表并提供注册 API → 树外后端插件调用注册 API 注册新 Backend → 用户用法不变,只需多装插件包(如 `pip install vllm-ascend-plugin`),推理即自动运行在对应硬件上。
- **Platform 对象**:后端相关对象为 executor、worker、model_runner、attention、custom ops、device_communicator;`Platform` 提供如 `get_attention_cls` 的 API 来初始化各组件。
- **逐项拆解**:Executor 要做到 backend agnostic(树外后端不再实现 XXXExecutor,见 PR #11256);Worker/ModelRunner/AttentionBackend 由树外实现、随 XXXPlatform 一起注册;Communicator 需先在 vLLM 实现基类(见 PR #11324);Custom OP 复用 torch dispatch key 机制、插件用新 dispatch key 创建,难点是加载插件编译的 .so。
- **运营配套**:CI/CD 用 buildkite self-host agent、硬件贡献者捐硬件;插件需与 V1 引擎(有自己的 Executor/Worker/Model Runner/Attention)兼容;插件仓库位置规范——进 `vllm-project` 做 "official support" 须满足:硬件 CI/CD 必需、持续贡献、发布节奏与 vLLM 同步。

## 优化点与收益
正文未给出量化数字。定性收益:后端解耦使代码更干净易维护;开发者可聚焦通用特性、不再被繁琐的后端类别困扰;各后端可自行演进,保证可用性与及时集成。

## 现状与落地
已关闭(2025-06-10),labels 含 stale。正文或结尾看不出「已实现/被替代/搁置」,落地情况需查关联 PR(#11256、#11324 等)。

## 关联
- [#7131](https://github.com/vllm-project/vllm/issues/7131)(通用插件系统,Done)
- [#9268](https://github.com/vllm-project/vllm/issues/9268)(后端模块化,In progress)
- [#7692](https://github.com/vllm-project/vllm/issues/7692)(ascend 接入诉求)
- [#9652](https://github.com/vllm-project/vllm/issues/9652)(IBM Spyre 接入诉求)
- [PR #11256](https://github.com/vllm-project/vllm/pull/11256)(executor backend agnostic)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
