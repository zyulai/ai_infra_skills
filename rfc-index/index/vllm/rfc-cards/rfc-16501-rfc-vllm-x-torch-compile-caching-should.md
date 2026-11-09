# RFC #16501: [RFC]: vLLM x torch.compile caching should be opt-out by default

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#16501](https://github.com/vllm-project/vllm/issues/16501) |
| 作者 | zou3519 |
| 创建日期 | 2025-04-11 |
| 状态 | ⚪ closed (关闭于 2025-11-19) |
| labels | RFC, good first issue, torch.compile |
| 热度 | 👍 10 · 💬 7 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
提议把 torch.compile 缓存哈希从 opt-in 改为 opt-out:默认哈希全部字段、只维护排除列表,避免新配置忘记更新哈希导致错误缓存。

## 提案动机
vLLM 决定是否复用 torch.compile 编译缓存的方式很脆弱:它维护一个需要哈希的配置列表,任一变化就判定需要全新 torch.compile。PR #16491 表明,给某个配置加新功能时很容易忘记更新哈希函数——该 PR 中 ModelConfig 的哈希函数没有涵盖所有可能改变编译结果的因素。

## 提出的方案
- 与 PyTorch Compiler 团队(cc @oulgen)讨论后,把哈希函数从 opt-in 改为 opt-out
- ModelConfig 的 compute_hash 改为默认包含 self.__dict__ 的全部字段,仅显式排除不影响编译的字段(示例中移除 enforce_eager、tokenizer_config 等)
- 理由:opt-out 更安全——错误缓存的风险是 (1) 模型意外报错、(2) 静默错误,因此宁可保守

## 优化点与收益
正文未给出量化数字。定性收益:降低「新增字段/重构后忘记更新哈希」引发错误缓存(意外报错或静默错误)的风险。

## 现状与落地
已关闭。方案方向明确(哈希函数 opt-out 化),正文未记录实现细节与结果,落地情况需查关联 PR。

## 关联
- PR #16491:https://github.com/vllm-project/vllm/pull/16491
- 哈希配置列表(vllm/compilation/backends.py):https://github.com/vllm-project/vllm/blob/70de35a8816e224663aede45b7f54eef250a5cfe/vllm/compilation/backends.py#L360-L394

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
