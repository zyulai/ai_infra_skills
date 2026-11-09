# RFC #9652: [RFC]: Add support for IBM Spyre accelerator

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#9652](https://github.com/vllm-project/vllm/issues/9652) |
| 作者 | tdoublep |
| 创建日期 | 2024-10-24 |
| 状态 | ⚪ closed(关闭于 2025-03-22) |
| labels | RFC, stale |
| 热度 | 👍 10 · 💬 7 |
| 分类 | Feature / Ecosystem |

## 一句话结论
通过 SpyreExecutor 等一系列 PR 把 IBM Spyre 加速器接入 vLLM,路径对齐现有 TPU/Inferentia 支持。

## 提案动机
IBM 在 Hot Chips 2024 发布 Spyre AI 加速器,与 IBM Research 合作设计,用于 IBM Z 大型机及 Power 平台上企业 AI workload 的 scale-up。由于 IBM 把自己的推理栈构建在 vLLM 之上,希望让 Spyre 获得vLLM 框架内的支持。Spyre 通过 torch.compile 融入 PyTorch 生态:IBM Research 已开发新的 torch.compile 后端,把 torch FX 图编译到 Spyre 硬件执行,因此设想 Spyre 支持以类似今天 TPU 支持的方式接入。正文指出两个需要绕开的限制:1) Spyre 目前只能执行 IBM 开源 Foundation Model Stack(`fms`)的建模代码——这是临时限制,终期目标是通过 torch.compile 在 Spyre 上运行 vLLM 原生建模代码;2) Spyre 暂不支持 paged attention 与 continuous batching,prefill 完一个 batch 后必须持续 decode 到该 batch 内全部序列结束,团队正在近期内移除此限制。

## 提出的方案
按 PR 序列推进(正文称 P1-P3 大部分已在私有 fork 完成,将拆成小 PR 逐个 upstream):
- **P1 单卡支持**:新增 `SpyreExecutor` 等继承 vLLM 核心类的类集合,架构对齐 AWS Inferentia 的 `NeuronExecutor`/`NeuronWorker`/`NeuronModelRunner`/`NeuronCausalLM`,执行 `fms` 的建模代码;因 Spyre 走 torch.compile,所有编译在 init 时触发、不在服务关键路径上,并加入 warmup 例程预先编译全部所需 shape(prompt 长度、输出 token 数、batch size),执行前把 batch pad 到已编译 shape(类似 CUDA graphs 的做法);测试用 eager 或 inductor 后端在 CPU 上跑,附 `Dockerfile.spyre`。
- **P2 调度改造**:为绕开无 continuous batching,running batch 中所有 decode 完成前不调度新 prefill(一行改动);并按 prompt 长度与 max output tokens 组 batch 以匹配 warmup 编译的 shape;改动须条件化、不影响现有设备行为(在 scheduler 内检查 `is_spyre()` 或「插入」备用调度器为待定设计点);测试覆盖 `MQLLMEngine` 与在线运行。
- **P3 多卡 TP**:新增 `MultiprocessingSpyreExecutor`;把其与 `MultiprocessingGPUExecutor` 的公共代码重构到父类 `MultiprocessingExecutor` + 设备 mixin(GPU/Spyre),参照 `MultiprocessingXPUExecutor` 已有模式但保留更多设备特化空间。
- **P4**:为 Spyre 启用 paged attention 与 continuous batching(TBD)。
- **P5**:让 vLLM 原生建模代码可在 Spyre 上运行(TBD)。

## 优化点与收益
正文未给出量化数字。收益为生态扩展:使 vLLM 可运行于 IBM Spyre(IBM Z / Power 平台),复用 torch.compile 后端机制,并借 P3 的执行器重构减少多设备执行器之间的重复代码。

## 现状与落地
已关闭(2025-03-22),labels 含 stale;正文为 PR 路线图(P4/P5 标注 TBD),未写明最终是否实现或被替代,落地情况需查关联 PR。

## 关联
- [Foundation Model Stack (fms)](https://github.com/foundation-model-stack/foundation-model-stack)(Spyre 建模代码来源)
- [#7131](https://github.com/vllm-project/vllm/issues/7131)(可插拔调度器设计,正文引为备选方案)
- [vLLM TPU model runner](https://github.com/vllm-project/vllm/blob/main/vllm/worker/tpu_model_runner.py#L685-L688)(正文所引架构参照)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
