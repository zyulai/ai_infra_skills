# PR #48361: [Test] Regression test for hybrid-Mamba eagle cache-peek in Mooncake connector (#43559)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48361](https://github.com/vllm-project/vllm/pull/48361) |
| 作者 | puririshi98 |
| 合并日期 | 2026-07-27 |
| labels | bug, ready, ci/build, v1, kv-connector |
| 热度 | 👍 3 · 💬 7 |
| 分类 | Speculative Decoding |
| bug 类型 | regression |

## 一句话结论
为 Mooncake connector 的 hybrid-Mamba eagle cache-peek 越界(#43559,静默输出损坏)补回归测试;生产 gate 已由 #49502 独立合入,本 PR 只补测试与注释。

## 问题现象
正文描述的是机制而非用户可见故障:hybrid(全注意力 + Mamba)+ eagle 场景下,Mamba 组被授予一个 block 的 eagle peek margin 后,可以匹配到比 eagle 剪枝后的全注意力命中长度多一个 block 的位置;采用该长度会从"已验证 token 前缀之后"的 recurrent 状态恢复——这是静默输出损坏而非崩溃,正文因此强调需要测试来防止 gate 被静默移除。正文未描述具体用户端故障表现。负面对照实验:从已合入的 gate 中去掉 `and not isinstance(spec, MambaSpec)` 后,新测试按预期失败(`assert 64 == 48` 失败),证明测试不是空转。

## 根因
正文明确披露:`MambaManager` 忽略 `drop_eagle_block`(Mamba block 在位置 p 本身就是 (p+1)*block_size 个 token 之后的 recurrent 状态,没有东西需要重算),因此给 Mamba 组发放 eagle peek margin 会让它匹配越过 eagle 剪枝后的全注意力命中点。本 PR 打开时曾自行提出 `not isinstance(spec, MambaSpec)` gate;范围随后变更——#49502 已独立落地该 gate(并扩展了 fine-grained hash margin),因此本 PR 收缩为:回归测试 + 一段解释"为什么 recurrent 组不给 peek margin"的注释(已合入的 gate 并未解释这一点)。

## 修复方案
- `tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py`:新增 `test_lookup_with_eagle_hybrid_full_plus_mamba_no_overrun`——构造 Full+Mamba 两个 KV cache group(`_mamba(16)` 用 `MambaSpec`,mamba_cache_mode="align")、`use_eagle=True`、4 个 block hash,断言 `hit == 48`(全注意力命中 4 块、eagle 弹出 1 块 → 48 个已验证 token,Mamba 组必须服务 state@48 快照而不是 peek 到 state@64);
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py`:在 gate 处补 3 行注释,说明 recurrent(Mamba)组不给 eagle peek margin 的原因。

## 影响与回归风险
- 回归面:纯测试与注释改动,正文明确不影响输出、精度或服务行为;测试在 GB200(aarch64, Python 3.13)上 32 passed。唯一需留意的是测试对 gate(`not isinstance(spec, MambaSpec)`)的行为锁定。

## 关联
- https://github.com/vllm-project/vllm/issues/43559 (回归测试针对的 overrun)
- https://github.com/vllm-project/vllm/pull/49502 (已合入的生产 gate)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
