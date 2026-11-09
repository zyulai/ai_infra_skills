# RFC #43545: [RFC]: Add Gumiho speculative decoding to vLLM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43545](https://github.com/vllm-project/vllm/issues/43545) |
| 作者 | userljz |
| 创建日期 | 2026-05-24 |
| 状态 | ⚪ closed (2026-09-25) |
| labels | RFC, stale |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Speculative Decoding |

## 一句话结论
把 Gumiho 混合 drafter(前 2 token 走 transformer head、其余并行 MLP head)作为 V1 新投机解码方法 `"method": "gumiho"` 接入 vLLM。

## 提案动机
Gumiho(ICML 2025 论文,由原作者提交)的出发点是早期投机 token 更重要:每个被接受的前缀 token 会解锁同批次后续所有 token。因此前两个 token 用小而高质量的 EAGLE 式 transformer head 自回归生成,其余 token 由 per-step MLP heads 并行生成。预期在 `num_speculative_tokens > 2` 时每步 drafter 延迟低于 EAGLE/EAGLE3,端到端接受 token 收益相近。作者已有 vLLM v0.16.0 上的 ROCm 原型并移植到 main。

## 提出的方案
- 新 drafter 模型 `vllm/model_executor/models/gumiho.py`、新 proposer(继承 EagleProposer,仅覆盖基类上默认 no-op 的两个 hook,不影响 EAGLE/EAGLE3/MTP/DFlash 等)、`GumihoConfig` 及 SpeculativeConfig 接线与注册表项。
- V2 GPU runner 显式抛 NotImplementedError;附文档、示例与 CPU 单测。
- 首 PR 范围外:FTA tree attention 验证、概率采样、V2 runner、多模态 drafter。
- 规模:5 个新文件(proposer 74 LoC、模型 437 LoC 等)+ 9 个修改文件(净 +110/−4 LoC)。

## 优化点与收益
ROCm smoke 测试:16-prompt 小评测、`num_speculative_tokens=3` 下 mean acceptance length ≈ 1.67,端到端已产出被接受 token;完整 EAGLE3 对比基准承诺随 PR 发布。正文未给出其他量化数字。

## 现状与落地
已关闭(2026-09-25),labels 含 stale;正文与结尾看不出是否实现,落地情况需查关联 PR #43544。

## 关联
论文 https://arxiv.org/abs/2503.10135;PR https://github.com/vllm-project/vllm/pull/43544;参考代码 https://github.com/AMD-AIG-AIMA/Gumiho;预训练 drafter https://huggingface.co/amd/Gumiho-llama3-8b

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
