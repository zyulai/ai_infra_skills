# PR #51468: [BugFix] Preserve divergent FA hits with external Mamba state

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51468](https://github.com/vllm-project/vllm/pull/51468) |
| 作者 | majunze2001 |
| 合并日期 | 2026-08-08 |
| labels | bug, ready |
| 热度 | 👍 3 · 💬 3 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
混合 FA+Mamba 模型外接 KV connector 时,Mamba 组块数可合法少于对齐端点,`truncate_computed_blocks` 的 assert 误杀 prefill worker;仅对 MambaSpec 组钳制修复。

## 问题现象
PD 分离部署(6 节点:prefill 2 x TP8,decode 4 x TP1 → DP16/EP16),MultiConnector(NIXL + MooncakeStore),`--enable-prefix-caching --prefix-match-unit 128`,attention block 1536,模型 Kimi-K3,store 800 GB 对 1,980,479-token GPU pool。16 个 98,944 token(`64*1536+640`,尾部 128 对齐但非 1536 对齐)的请求、并发 8,prefill 在第 7 个请求处死亡:

```
vllm/v1/core/kv_cache_manager.py:792 in truncate_computed_blocks
    assert num_blocks <= len(group_blocks)
AssertionError
```

引擎致命崩溃,worker 中途被杀(崩溃前 connector 客户端 8→4,因 rank 死亡)。

## 根因
PR 正文披露:`truncate_computed_blocks()` 断言每个 KV cache 组都持有至少 `num_computed_tokens // block_size` 个块。对混合(full-attention + Mamba)模型 + KV connector,Mamba 组可以合法地更少——它的块是完整的循环状态(conv+ssm)而非逐 token KV,外部命中可以在本地组已不覆盖的边界提供该状态。assert 把这种良性分歧变成引擎致命错误。引入于 #49502(函数与其调用方),main 上仍存在。触发需同时满足两个条件(因此普通 prefix caching 不会踩中):`partial_tail != 0`(本地命中结束在非块边界的匹配处,要求 `--prefix-match-unit` 细于 KV block size)且 `ext_tokens > partial_tail`(connector 持有的前缀比本地缓存多,即本地被逐出而外部 store 仍保留)。长 chunked prefill 同时满足两者:请求按 chunk 反复重入调度并对自己的前序 chunk 重查,同时并发请求消耗 GPU pool 而外部 store 存活更久。

## 修复方案
`vllm/v1/core/kv_cache_manager.py` 的 `truncate_computed_blocks`:必须覆盖对齐端点的组保留 assert,仅对 `MambaSpec` 组用 `min()` 钳制(全面 `min()` 也能止崩,但会静默掩盖 attention 组变短这一真 bug,按 MambaSpec 限定可让问题保持显眼):

```diff
-            assert num_blocks <= len(group_blocks)
+            if isinstance(group.kv_cache_spec, MambaSpec):
+                num_blocks = min(num_blocks, len(group_blocks))
+            else:
+                assert num_blocks <= len(group_blocks)
```

纯切片、不动 refcount、不修改 lookup 结果。测试:新增 `test_truncate_computed_blocks_allows_short_mamba_group_only`(短 Mamba 组被放行、短 attention 组仍 AssertionError);`test_hybrid_per_group_hit_divergence_fa_deeper_no_external` 重构为参数化的 `test_hybrid_fa_deeper_hit_uses_external_mamba_state`(有外部命中时在外部 Mamba 状态可用的边界续跑,无外部命中时回退共同本地边界)。

## 影响与回归风险
- 影响窗口:#49502(2026-07-27 合并)引入起至本 PR 之间的 main 分支。
- 回归面:仅 `truncate_computed_blocks` 的组块数校验;Mamba 组从硬断言变为钳制,attention 等其他组仍硬断言。E2E 验证:同配方同负载修复后 16/16 完成且重复两次,assert 次数为 0,外部加载 64 次(16 req x 4 ranks),store 读取 `Get: Req=256/256, Item=7168/7168`;单元测试 20 passed + 4 passed(hybrid 相关)。

## 关联
- https://github.com/vllm-project/vllm/pull/49502(引入 assert 的 PR)
- https://github.com/vllm-project/vllm/pull/50507(相邻开放工作:partial-tail 混合模型扩展)
- https://github.com/vllm-project/vllm/pull/46453(相邻:混合模型 per-group 命中分歧)
- https://github.com/vllm-project/vllm/pull/50454(同类失败,offloading connector)
- https://github.com/vllm-project/vllm/pull/50630(相邻设计缺口讨论)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
