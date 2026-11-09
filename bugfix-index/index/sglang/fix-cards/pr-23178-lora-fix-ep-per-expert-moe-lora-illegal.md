# PR #23178: [LoRA] Fix EP + per-expert MoE LoRA illegal memory access

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#23178](https://github.com/sgl-project/sglang/pull/23178) |
| 作者 | jybsuper |
| 合并日期 | 2026-04-22 |
| labels | lora, run-ci, jit-kernel |
| 热度 | 👍 3 · 💬 1 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论

EP + per-expert MoE LoRA 首次 forward 即非法访存;LoRA buffer 改按本地专家数分配并过滤 -1 哨兵 ID。

## 问题现象

`--tp N --ep M --enable-lora --lora-backend triton --moe-runner-backend triton` 配合 per-expert LoRA adapter,在任何 MoE 模型上都无法运行:第一次 forward 即稳定触发 `CUDA error: an illegal memory access was encountered`。

## 根因

正文披露两个相关 bug:

1. `LoRAMemoryPool` 的 per-expert buffer 按 `[num_loras, num_global_experts, ...]`(如 128 个全局专家)分配;而 EP 下 `StandardDispatcher` 在 MoE kernel 之前就把 `topk_ids` 从全局 ID 重映射为本地 ID(`[0, num_local_experts)` 加 `-1` 哨兵),用本地 ID 索引按全局专家数分配的 buffer 会读到错误专家的权重,或与 bug 2 叠加触发非法访存。
2. `moe_lora_align_block_size` 的 `_count_and_sort_expert_tokens` 只挡 `expert_id >= num_experts`,`-1` 哨兵通过守卫,造成 `cumsum_buffer` 负偏移的 OOB 写,以及用 garbage `rank_post_pad` 写 `sorted_token_ids`;同文件另外两处调用点已有 `< 0` 守卫,仅此处遗漏。

## 修复方案

- `python/sglang/srt/lora/mem_pool.py`:池初始化时捕获 `(moe_ep_size, moe_ep_rank)` 与 `moe_use_local_expert_ids`(仅当 EP>1 且 runner 会重映射为本地 ID 时为真;FlashInfer CUTLASS/CuteDSL/TRTLLM-routed 保持全局键),per-expert buffer 按 `num_local_experts` 分配(`get_lora_A_shape`/`get_lora_B_shape`);MoE 模块按 `moe_tp_size` 而非外层 `tp_size` 切分;新增 `_iter_local_expert_weights` 统一过滤并重映射全局→本地专家 ID(dict 与 3D tensor 两种输入),未拥有的槽位置零。
- `python/sglang/jit_kernel/csrc/lora/moe_lora_align_kernel.cu`:守卫补上 `expert_id < 0`。

```diff
-    if (expert_id >= num_experts) {
+    if (expert_id < 0 || expert_id >= num_experts) {
       continue;
     }
```

- 新增 `test/registered/unit/lora/test_mem_pool_ep_unit.py`(22 例,纯 CPU)。

注:pack 中 diff 在 40000 字符处截断,以上修复描述基于正文 Changes 部分与可见 diff;截断部分未读取。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:EP + per-expert MoE LoRA 加载与 kernel 路径。纯 TP(moe_ep_size==1)与保持全局 ID 的 runner 行为不变(正文以 TP-only LoRA 单测与 `--tp 2` 端到端回归验证);`--tp 4 --ep 4` 无 LoRA 基线仍可用;非 LoRA 路径未动。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
