# PR #34443: [Bugfix][DSA] Fix num_splits "(b+1)" crash on prefill-CP speculative decode

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34443](https://github.com/sgl-project/sglang/pull/34443) |
| 作者 | NVShreyas |
| 合并日期 | 2026-08-12 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
prefill-CP + EAGLE/MTP 投机解码时 DSA 元数据被过度 cp_align 填充,`fwd_kvcache_mla` 断言崩溃;按 CP-v2 状态对齐填充逻辑。

## 问题现象
运行 DSA 模型(GLM-5.x / DeepSeek-V3.2),参数组合 `--dsa-prefill-backend flashmla_kv --enable-prefill-cp --cp-strategy interleave` 加 EAGLE/MTP 投机解码,在 warmup 阶段即确定性崩溃:
```
RuntimeError: num_splits must have shape (b+1)
```
这是 `torch.ops.sgl_kernel.fwd_kvcache_mla` 内部的 `TORCH_CHECK`,断言 `num_splits.shape[0] == q.shape[0] + 1`。正文验证:GLM-5-FP8, tp8/ep8,`--attention-backend dsa --dsa-prefill-backend flashmla_kv --enable-prefill-cp --cp-strategy interleave --moe-a2a-backend megamoe` + EAGLE,修复前约 175 秒时必崩,修复后正常服务并处理请求。

## 根因
正文披露:`cal_padded_tokens`(`python/sglang/srt/layers/attention/dsa/utils.py`)负责定长 FlashMLA 的 `num_splits` 元数据(经 `pad_dsa_cache_seqlens` → `_compute_flashmla_metadata`),必须与给 `q` 定长的 `ForwardBatch.prepare_mlp_sync_batch` 镜像一致。但 `prepare_mlp_sync_batch` 仅在 **CP-v2 关闭时** 应用 `cp_align_size`,而 `cal_padded_tokens` **无条件** 应用。在投机 forward(`TARGET_VERIFY`/`DRAFT_EXTEND_V2`)下,`enable_cp_v2()` 为 True 而 `is_cp_v2_active(forward_batch)` 为 False(这些模式不是 `context_parallel_extend`),于是 `cal_padded_tokens` 越过 CP-v2 早退分支继续执行:`q` 只 pad 到 `attn_tp_size`,元数据却被无条件 cp_align 到 `attn_cp_size` 的倍数,造成过度填充。当 `num_draft_tokens * batch_size` 不是 `attn_cp_size` 的倍数时,`len(dsa_cache_seqlens_int32)`(= `num_splits.shape[0] - 1`)超过 `q.shape[0]`,断言触发。正文称此为 #30642 中 `attn_tp` 对齐修复的 `attn_cp` 类似问题。

## 修复方案
`python/sglang/srt/layers/attention/dsa/utils.py` 的 `cal_padded_tokens`:给 `cp_align` 循环加 `if not enable_cp_v2()` 守卫,使 DSA 元数据填充与 `prepare_mlp_sync_batch` 完全一致:
```diff
-    cp_align_size = get_cp_padding_align_size()
-    for i in range(sync_group_size):
-        global_num_tokens[i] = ceil_align(global_num_tokens[i], cp_align_size)
+    if not enable_cp_v2():
+        cp_align_size = get_cp_padding_align_size()
+        for i in range(sync_group_size):
+            global_num_tokens[i] = ceil_align(global_num_tokens[i], cp_align_size)
```
正文说明无数值影响——只是移除没有对应 `q` 行的幽灵元数据行(填充项置零、输出被丢弃)。各路径行为:Legacy CP(`enable_cp_v2()` 为 False)不变;CP-v2 下的普通 prefill EXTEND 走 `is_cp_v2_active` 早退,不变;只有崩溃的 `TARGET_VERIFY`/`DRAFT_EXTEND_V2` 投机 forward 改变:元数据长度现在等于 `q.shape[0]`。

## 影响与回归风险
- 影响窗口:正文未说明(关联 issue #30296 所报故障窗口)。
- 回归面:仅 DSA 的 `cal_padded_tokens` 元数据长度计算;`enable_cp_v2()` 开启时的投机 forward 元数据变短(q 对齐),其余路径(legacy CP、CP-v2 普通 EXTEND)逐字节不变。若下游有代码隐式依赖旧的 cp_align 过度填充长度,需重新核对(正文未提及此类依赖)。

## 关联
- https://github.com/sgl-project/sglang/issues/30296(修复的 issue)
- https://github.com/sgl-project/sglang/pull/30642(attn_tp 对齐修复,#34443 为其 attn_cp 类似修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
