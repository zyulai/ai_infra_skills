# RFC #36219: [RFC]: Speculative Decoding Proposer Interface Unification Proposal

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36219](https://github.com/vllm-project/vllm/issues/36219) |
| 作者 | wangxiyuan |
| 创建日期 | 2026-03-06 |
| 状态 | ⚪ closed(关闭于 2026-04-29) |
| labels | RFC |
| 热度 | 👍 4 · 💬 7 |
| 分类 | Speculative Decoding |

## 一句话结论
提议统一 vLLM 投机解码各 proposer 的接口:BaseProposer 抽象基类 + 标准化 ProposerInput + 注册表工厂,解除 model runner 与各算法的耦合。

## 提案动机
vLLM 的投机解码模块在可维护性与可扩展性上面临显著挑战,既阻碍上游开发也阻碍硬件插件集成。问题一:各算法的 proposer 接口互不一致——NgramProposer 的 `propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None)`、SuffixDecodingProposer 的 `propose(input_batch, sampled_token_ids, slot_mappings=None)`、MedusaProposer 的 `propose(target_hidden_states, sampling_metadata, slot_mappings=None)`、EagleProposer 继承 `SpecDecodeBaseProposer`、DraftModelProposer 与 Eagle 类似但初始化不同;只有 Eagle 和 DraftModel 共享基类,其余完全独立实现。问题二:model runner 耦合:`gpu_model_runner.py` 里充斥按 method 分支的 if-else(ngram/suffix/medusa/eagle 各走不同代码路径),导致新增算法必须改 model runner、单算法难以独立测试、runner 代码随方法增多愈发复杂。

## 提出的方案
- 类继承:`BaseProposer(ABC)` 定义统一接口 `propose(proposer_input: ProposerInput) -> ProposerOutput`;硬件特定实现继承并按需覆写,如 `AscendEagleProposer(EagleProposer)` 只覆写 `prepare_inputs` 等 Ascend 特有部分。
- ProposerInput:重构后 model runner 统一调用 `self.drafter.prepare_inputs(...)` 得到 `proposer_input`,再 `self.drafter.propose(proposer_input)`;标准化的 `ProposerInput` dataclass 是解耦的关键。
- ProposerRegistry:在 `vllm/v1/spec_decode/registry.py` 维护 `PROPOSER_CLASSES` 字典(ngram/medusa/eagle/...),提供 `create_proposer(method, vllm_config, device)` 工厂;`gpu_model_runner.py` 只需 `create_proposer(spec_config.method, ...)` 一行。
- 硬件插件覆盖机制:提供 `@register_proposer(method)` 装饰器,vllm-ascend 导入时即可用 `AscendEagleProposer` 覆盖 vllm 默认的 eagle proposer,无需修改上游代码。

## 优化点与收益
正文未给出量化数字。收益为工程性:新增投机解码算法不再需要修改 model runner;单个算法可独立测试;model runner 复杂度不再随算法数量增长;硬件插件(如 vllm-ascend)可通过注册表覆盖 proposer 实现,改善硬件生态的集成成本。

## 现状与落地
已关闭(2026-04-29)。正文以方案草图(接口、注册表、覆盖示例)呈现,反馈期「per day」,节奏较快;具体落地 PR 需查关联。

## 关联
正文未列出

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
