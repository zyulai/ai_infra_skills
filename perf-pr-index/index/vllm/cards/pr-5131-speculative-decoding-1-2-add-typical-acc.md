# PR #5131: [Speculative Decoding 1/2 ] Add typical acceptance sampling as one of the sampling techniques in the verifier

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @sroy745 |
| 合并 | 2024-06-18 |
| 分类 | Speculative Decoding(tier: major) |
| Label | 无 |
| Diff | diffstat 缺失(素材未提供 additions/deletions 与文件数);diff 另有 1 个文件因超长被略去 |
| 链接 | https://github.com/vllm-project/vllm/pull/5131 |

## 一句话结论

为 Speculative Decoding verifier 新增 typical acceptance 采样器,用基于目标分布熵的动态阈值接受 draft token,提升投机解码的接受长度。

## 动机

PR 之前 vLLM 的投机解码验证步骤只有 rejection sampling 一种准则,对所有 draft token 施加统一的严格接受条件。本 PR 按照 Medusa 论文(arXiv:2401.10774)第 3.3.1 节实现 typical acceptance sampling,作为 verifier 的另一种可选采样技术:当 draft token 在目标分布中的概率满足基于分布熵的准则即被接受,而不是仅依赖 draft/target 概率比的硬性判定。作者将其定位为系列改动的第一部分(1/2),本 PR 只引入采样器本身,后续 PR 再接入 spec_decode_worker。

## 优化点

核心机制:`TypicalAcceptanceSampler` 对每个 draft token 计算其在 target 分布中的概率,与动态阈值 `min(posterior_threshold, posterior_alpha * exp(-H))` 比较,其中 `H` 是 target 分布的熵;熵越高(分布越平缓)阈值越宽,越容易接受 draft token。接受过程在第一个被拒绝的 token 处停止;若全部被拒,则回退为对 target 分布 k=0 位置做 greedy 采样,保证每步至少输出一个 token;若全部接受,则追加 bonus token(默认 `disable_bonus_tokens=False`,与 rejection sampler 的默认值不同)。工程上抽出 `SpecDecodeBaseSampler` 基类,把输出组装(`_create_output`)、strict-mode 形状/dtype/device 检查和 accepted/emitted 统计计数从 `RejectionSampler` 上移,两个采样器共用。

## 关键改动

```text
vllm/model_executor/layers/typical_acceptance_sampler.py: 新增 TypicalAcceptanceSampler,实现熵阈值判定与 greedy 回退
vllm/model_executor/layers/spec_decode_base_sampler.py: 新增基类,收纳两采样器共用的输出组装与校验逻辑
vllm/model_executor/layers/rejection_sampler.py: 改为继承 SpecDecodeBaseSampler,删除重复代码
tests/samplers/test_typical_acceptance_sampler.py: 新增采样器单元测试
```

关键片段(熵阈值判定):

```python
candidates_prob = torch.gather(target_probs, dim=-1,
    index=draft_token_ids.unsqueeze(-1)).squeeze(-1)
posterior_entropy = -torch.sum(
    target_probs * torch.log(target_probs + epsilon), dim=-1)
threshold = torch.minimum(
    torch.ones_like(posterior_entropy) * self._posterior_threshold,
    torch.exp(-posterior_entropy) * self._posterior_alpha)
accepted_mask = candidates_prob > threshold
```

默认超参:`posterior_threshold=0.09`,`posterior_alpha=0.3`。

## 性能声明

PR 正文未提供 benchmark 数字,也未注明测试条件(正文仍是模板占位符 "FILL IN THE PR DESCRIPTION HERE",素材的 perf claims 检测为 none)。

## 验证与回归风险

新增 `tests/samplers/test_typical_acceptance_sampler.py`,参数化覆盖 k=1..5、vocab 30k/50k、batch 1..31,并验证:均匀分布应全部接受、零温度分布应全部拒绝并回退 greedy、混合分布逐序列正确、vocab 越界抛 AssertionError、bonus token 开关行为。回归风险较低:本 PR 未把新采样器接入 spec_decode_worker,默认投机解码路径行为不变;`RejectionSampler` 的重构属于等价迁移(逻辑上移到基类),影响面限于 spec decode 采样路径。

## 关联

系列后续 PR(2/2,接入 spec_decode_worker)编号在素材中未注明;修复 issue #5015。索引内同区域卡片:#13365(spec decode n-gram 匹配优化)、#14930(rejection sampling 优化)、#14702、#16937(EAGLE-3)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
