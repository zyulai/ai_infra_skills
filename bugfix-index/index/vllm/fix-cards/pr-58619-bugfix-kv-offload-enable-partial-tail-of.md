# PR #58619: [Bugfix][KV Offload] Enable partial-tail offload for hybrid models under DCP

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#58619](https://github.com/vllm-project/vllm/pull/58619) |
| 作者 | nvyutwu |
| 合并日期 | 2026-11-12 |
| labels | bug, documentation, ready, ci/build, kv-connector, nixl |
| 热度 | 👍 0 · 💬 25 |
| 分类 | KV Cache / Memory |
| bug 类型 | general |

## 一句话结论
partial-tail offload 门控要求单一 block size 且 dcp==1,混合模型在 DCP 下全关且源块寻址有误;改按 core hit alignment 门控并逐组寻址。

## 问题现象
`supports_partial_tail` 同时要求 `len(group_block_sizes) == 1` 与 `decode_context_parallel_size == 1`;DCP 下 full-attention offload block 为 `dcp ×` attention block 而 recurrent blocks 不分片(12288 vs 1536 at DCP 8,attention 组在 `build_offloading_config` / `resolve_dcp_kv_block_size` 中缩放,`MambaManager.__init__` 对 recurrent state 还原),两条件皆不成立——所有混合模型在 DCP 下 partial-tail offload 关闭。此外 store 路径用同一物理块索引 `boundary // self._partial_tail_block_size` 寻址所有组(block size 不同即错),且从 `kv_group_configs[0]` 读 block size(不一定是 full-attention 组)。硬件验证(Kimi-Linear-48B-A3B-Instruct,KDA+MLA,TP2,`--mamba-cache-mode align --prefix-match-unit 128`,16 GiB CPU pool,GB300):DCP2 下 main 在 1541/24069 prompt 上恢复 0 token、12549 恢复 12288、39685 恢复 38912;fix 分别恢复 1536/24064/12544/39680,每个 prompt 都恢复到 last hash boundary;DCP1 与 main 完全一致(无回归)。回放 greedy 输出(32 tokens)与 cold run 逐字节一致,0 allocation failures。单元测试(`pytest tests/v1/kv_connector/unit/offloading_connector/ -q`,GB300):main 690 passed/3 failed → 本 PR 704 passed/3 failed(3 个失败为两树共有的既有失败)。

## 根因
正文披露:(1) 门控条件在 DCP 混合模型上恒假;(2) store 对所有组用单一共享块索引、block size 取自 `kv_group_configs[0]`;(3) `assert len(boundaries) == 1` 在 DCP 下一个 hand-off 可携带多于一个 boundary(如最后一个完整 recurrent block 紧邻 prompt tail)时变成 engine-core crash,且存储内容依赖 hand-off 的分批方式。

## 修复方案
`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`:(1) 门控改为 core 的 resolved hit alignment——`cache_hit_alignment_tokens < alignment_tokens`(替代单一 block size 与 dcp==1 检查);混合 block sizes 允许的条件:每个 recurrent block 整除 full-attention block 且 hash 对齐、`spec.tokens_per_hash % dcp_world_size == 0`(保守条件:硬件只验证过该形状)、DCP>1 时 EAGLE 组仍禁用(proof position 在 DCP 下未验证);`_partial_tail_block_size` 改取 `alignment_tokens`(tail 可落入的窗口就是 full-attention offload block,`SchedulerOffloadConfig.alignment_tokens` 已有,不加新字段);(2) 每组用自己的 block size 寻址"结束于 boundary 的块":`(boundary - 1) // tokens_per_block`——旧的 `boundary // tokens_per_block` 在 boundary 恰落在组块边缘时指向下一个块(正文注:今天 inert,`blocks_per_chunk == 1` 时 worker 不用 `block_indices`,但一旦不成立即错);(3) tail 存到 core 的 prompt checkpoint `get_mamba_prefill_checkpoint_position(...)`(与 Mamba manager 相同实参),只存该 boundary 的 hand-off 条目,删除 `assert len(boundaries) == 1`,存储内容不再依赖 hand-off 的分批方式。dcp==1 时 store 路径解析到与之前相同的单块索引。

## 影响与回归风险
- 影响窗口:正文未给出。
- 回归面:OffloadingConnector scheduler 的 partial-tail 门控、store 寻址与 lookup/load;EAGLE/MTP partial tails 在 DCP 下保持禁用(dcp==1 时跟随 main #61019);与 #58618(lookup anchor)互补——本 PR 单独正确,但 CPU tier 只在完整块 hit 一个窗口内命中,需 #58618 才能命中更远;3 个单元测试失败为本 commit 上 main 既有(期望 recurrent group CPU `BlockStored.block_size` 为 0,而 #62613 后是 4),非本 PR 引入;pre-commit(ruff、mypy)通过;`tokens_per_hash % dcp == 0` 为保守门控,store/load 按整物理块拷贝并不需要它。

## 关联
- [#57216](https://github.com/vllm-project/vllm/pull/57216)(本 PR 的拆出来源,part 2 of 2)
- [#58618](https://github.com/vllm-project/vllm/pull/58618)(互补的 lookup anchor)
- [#53889](https://github.com/vllm-project/vllm/pull/53889)(complementary:config 侧按 DCP 缩放 groups)
- [#61019](https://github.com/vllm-project/vllm/pull/61019)(dcp==1 时 EAGLE 行为对齐 main)
- [#62613](https://github.com/vllm-project/vllm/pull/62613)(既有测试失败的来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
