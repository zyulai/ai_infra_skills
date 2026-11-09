# PR #28378: [AMD] Fix Always mask padded topk_ids on HIP to prevent garbage MoE routing (DeepSeek-R1-MXFP4 accuracy regression)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28378](https://github.com/sgl-project/sglang/pull/28378) |
| 作者 | kangwangamd |
| 合并日期 | 2026-06-18 |
| labels | amd, run-ci, bypass-fastfail |
| 热度 | 👍 0 · 💬 7 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
HIP 上 padded topk_ids 掩码被误移入 EPLB 守卫导致垃圾专家 ID 摧毁 MoE 精度,改为无条件掩码填 0。

## 问题现象
`test_deepseek_r1_mxfp4_8gpu.py`(`stage-c-test-large-8-gpu-amd-mi35x`)失败,出现灾难性 GSM8K 精度下降:`AssertionError: 0.1379833206974981 not greater than 0.94`。即在 MI35x、TP=8 的 `amd/DeepSeek-R1-MXFP4-Preview` 上,精度从 >0.94 崩到约 0.09–0.14。

## 根因
PR 正文披露:commit `63df86f5`(PR #28188,"Skip eplb bookkeeping and topk remap when EPLB is not in use on mori-ep / HIP")在 `_post_process_topk_ids` 中把 padded 区域掩码意外移进了 `_eplb_remap_enabled()` 守卫内部。对不使用 EPLB 的模型(常见情况,含 DeepSeek-R1-MXFP4),`_eplb_remap_enabled()` 返回 False,`_mask_topk_ids_padded_region` 永远不会执行,`topk_ids` 的 padded 槽位保留内核产生的垃圾/越界专家 ID。虽然路由权重已被清零,但 aiter MoE / MXFP4 indexer 内核在加权和之前直接读 `topk_ids` 索引专家权重表,垃圾 ID 破坏计算并摧毁精度。

## 修复方案
改动位于 `python/sglang/srt/layers/moe/topk.py` 的 `_post_process_topk_ids`(HIP 分支),净改动 +9/−8:
- `_mask_topk_ids_padded_region(topk_ids, num_token_non_padded, fill_value=0)` 改为无条件调用,保证 padded token 总拿到合法在界专家 ID(填 0 而非 -1,因 aiter 无法处理 -1);
- 仅保留 logical→physical remap(`topk_ids_logical_to_physical`)在 `_eplb_remap_enabled()` 守卫内;
- `_zero_topk_weights_padded_region` 保持无条件调用不变,padded token 对 hidden state 的贡献仍为零。

关键 diff 摘录:

```diff
     elif _is_hip:
+        _mask_topk_ids_padded_region(topk_ids, num_token_non_padded, fill_value=0)
         # The logical->physical remap is only meaningful when a real
         # expert-location mapping exists. With a trivial placement and EPLB off
-        # the map is identity, and indexing it here is both unnecessary and not
-        ...
         if _eplb_remap_enabled():
-            _mask_topk_ids_padded_region(topk_ids, num_token_non_padded, fill_value=0)
             topk_ids = topk_ids_logical_to_physical(
```

## 影响与回归风险
- 影响窗口:由正文可知,问题自 #28188(commit `63df86f5`)引入至本 PR 合并前;受影响的是 HIP/AMD 上不启用 EPLB 的 MoE 模型。
- 回归面:仅 HIP 分支的 topk 后处理路径。掩码恢复为 #28188 之前的无条件行为,padded 区域填 0;EPLB remap 逻辑仍在守卫内,启用 EPLB 的路径不受影响。正文确认无性能影响(`test_bs_1_speed` 152.01 token/s 通过),并在 MI355X 上单测验证:padded topk_ids 被正确钳到 0、真实 token 不受影响。

## 关联
- [PR #28188](https://github.com/sgl-project/sglang/pull/28188)(引入回归的 commit `63df86f5`)
- [PR #28410](https://github.com/sgl-project/sglang/pull/28410)(正文提及:MTP 测试中 `avg_spec_accept_length=1.45 (<2.04)` 的失败与本 PR 无关,系 #28043 引入的 `predict.to(torch.int64)` plan-stream 跨流竞态,由 #28410 修复)
- CI run [27598431399](https://github.com/sgl-project/sglang/actions/runs/27598431399)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
