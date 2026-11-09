# PR #63971: [Bugfix][Mooncake] Let decodes reuse divergent local hybrid prefix hits

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#63971](https://github.com/vllm-project/vllm/pull/63971) |
| 作者 | whx-sjtu |
| 合并日期 | 2026-11-02 |
| labels | bug, kv-connector, mooncake |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
`MooncakeConnector` 未 opt-in `supports_divergent_local_hybrid_hits`,hybrid 模型 decode 每轮丢弃本地 attention 前缀命中、整段从 P 拉取;让 `kv_consumer` 角色 opt-in 修复,decode 前缀命中率 0.0% → 93.9%。

## 问题现象
hybrid 模型(attention + Mamba/GDN/KDA)的 decode 走 MooncakeConnector 时,本地 attention 前缀命中被丢弃,每轮都从 prefill 实例(P)拉取整个 prompt。Kimi-K3 1P1D(MI355X、Mooncake RDMA、concurrency 48 多轮 agentic 回放)实测 decode 本地前缀命中率为 0.0%。流程:decode 从不计算被拉取 prompt 的 block 边界,因此只缓存 attention blocks、没有任何 recurrent state;修复前 `get_computed_blocks` 只返回所有 group 共享的命中(0 token),Mooncake 请求 N-1 个 token、把所有 attention block 加 state slot 全部拉一遍。

## 根因
PR 正文披露:`MooncakeConnector` 没有 opt-in `supports_divergent_local_hybrid_hits`(#50344 引入的属性),调度器因此对 hybrid 模型只保留各 group 一致的命中,divergent 的本地 attention 命中被丢弃,重复外部传输。

## 修复方案
改 `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`:
1. 新增属性 `supports_divergent_local_hybrid_hits`,返回 `self._pulls_mamba_state`;
2. `__init__` 中计算 `_pulls_mamba_state = (kv_role == "kv_consumer" and any(isinstance(group.kv_cache_spec, MambaSpec) for group in kv_cache_config.transfer_groups))`。

opt-in 后调度器保留 attention 命中 H,`update_state_after_alloc` 本就会拉取每个 transfer group 未哈希的 blocks(state slot 也在内);无内容可拉时回退到 reconciled lookup(#50344)。只让 `kv_consumer` opt-in:per-group lookup 报告共享前缀边界为 0,会让本地 prefill 的角色(`kv_producer`/`kv_both`)禁用 junction checkpoints,故它们保持现状 lookup。新增调度器单测 `test_hybrid_gdn_remote_prefill_keeps_local_attention_prefix`(main 上失败 `assert 6 == 2`,本分支通过;Mooncake 单测 164 passed)。

```python
@property
def supports_divergent_local_hybrid_hits(self) -> bool:
    return self._pulls_mamba_state
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围(自 #50344 引入该属性后 Mooncake 一直未接入)。
- 回归面:仅 `kv_consumer` 且 transfer groups 含 MambaSpec 的 Mooncake 部署;`kv_producer`/`kv_both` 与非 hybrid 模型行为不变。局限(正文列明):serving 数据来自 vLLM 81198e97 + 强制 True 的 property,未在 main 上重跑;`MultiConnector` 下在 #63041 之前需要每个 child 都 opt-in;该负载上 TTFT 无变化(修复后 mean 3137ms / p50 1730ms,对比修复前 2984/1727ms)。draft acceptance 58.78% → 58.79%(与 MoRIIO READ 58.79% 持平)。

## 关联
- PR #50344(divergent local hybrid hits 机制来源)
- PR #63041(MultiConnector child opt-in 的后续依赖)
- PR #53730、#63420、#63425(正文说明与它们不重复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
