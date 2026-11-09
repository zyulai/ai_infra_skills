# PR #57050: [Bugfix] Fix incorrect Mamba block allocation estimate that prevents request admission

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57050](https://github.com/vllm-project/vllm/pull/57050) |
| 作者 | wzhao18 |
| 合并日期 | 2026-09-16 |
| labels | bug, kv-cache-manager, ready |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
Mamba 的 `get_num_blocks_to_allocate` 对从外部 KV 存储加载 prefix cache 的长请求算错物理块数,请求无法过准入门并永远卡在等待队列、系统停摆;改为把估算值 cap 到物理上限。

## 问题现象
Kimi K3 上发现:从外部 KV 存储(如 Mooncake)加载 prefix cache 时,长请求可能被拒绝准入(admission),永远卡在 waiting 队列,导致整个系统停 stall(正文:"causing entire system to stall")。复现方式:在 8 x B300 上按正文给出的配置 `vllm serve moonshotai/Kimi-K3 ...`(TP=8、DCP=8、`--mamba-ssm-cache-dtype bfloat16`、`--enable-prefix-caching`、MooncakeStoreConnector kv_both 等)起服务,发送一个 1M 长度请求,清掉 GPU 本地 KV cache 后再次发送同一请求,该请求无法通过准入门,整个系统 stall。

## 根因
正文披露:bug 位于 Mamba 的块分配估算调用 `get_num_blocks_to_allocate`,它对这类请求所需的物理块数计算不正确,并指出这可能是 PR #53614 引入的回归。(由 diff 推断)旧逻辑在 `num_new_blocks > 0` 时会**无条件覆盖**已算出的 `num_new_blocks`:`num_new_blocks = 1 + int(has_partial_hit) + checkpoint_block`(未分配过块时再加 speculative blocks),而不是把物理上限与逐 token 精确估算取小;在 checkpoint/admission 场景下该覆盖值与随后 `allocate_new_blocks` 的实际分配量不一致,准入估算偏低/偏高失配,长请求被错误拒之门外。

## 修复方案
`vllm/v1/core/single_type_kv_cache_manager.py` 的 `get_num_blocks_to_allocate`:把旧的无条件覆盖改为先算物理上限、再取最小值:

```diff
-                if not (checkpoint_block and blocks_allocated):
-                    num_new_blocks = 1 + int(has_partial_hit) + checkpoint_block
-                    if not blocks_allocated:
-                        num_new_blocks += self.num_speculative_blocks
+                physical_block_cap = 1 + int(has_partial_hit) + checkpoint_block
+                if not blocks_allocated or checkpoint_block:
+                    physical_block_cap += self.num_speculative_blocks
+                num_new_blocks = min(num_new_blocks, physical_block_cap)
```

另在 `tests/v1/core/test_single_type_kv_cache_manager.py` 新增参数化单测 `test_mamba_checkpoint_admission_matches_allocation`(block_size 896/1536 × speculative 0/1/4 × prompt 25121/704547),断言 admission 估算 == 分配估算 == 实际分配块数;main 上失败、本分支通过。

## 影响与回归风险
- 影响窗口:正文指出可能是 PR #53614 引入的回归;具体版本号未提及。
- 回归面:仅 `single_type_kv_cache_manager.py` 中 Mamba manager 的 `get_num_blocks_to_allocate` 估算路径;改动把「覆盖」语义改为「取 cap」语义,`not blocks_allocated or checkpoint_block` 时 speculative blocks 计入上限的条件比原来宽(checkpoint_block 且已分配过块的场景现在也加),估算结果更贴近实际分配;使用同一估算做准入判定的 Mamba/hybrid 模型(Kimi K3 等)的调度决策随之变化。

## 关联
- 可能的引入回归: https://github.com/vllm-project/vllm/pull/53614

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
