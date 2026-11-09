# PR #19415: [BUG]: fix prevent illegal memory access in Mamba SSM tracking during EAGLE speculative verification

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19415](https://github.com/sgl-project/sglang/pull/19415) |
| 作者 | ConnorLi96 |
| 合并日期 | 2026-03-05 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
EAGLE 验证阶段沿用按完整 prefill 长度构建的过期 Mamba track mask,越界索引仅含 draft_token_num 行的 h,触发 `cudaErrorIllegalAddress`;prepare_for_verify 时显式清空 mask 与 seqlens。

## 问题现象
使用 `--mamba-scheduler-strategy extra_buffer` 搭配 EAGLE 投机解码时,TARGET_VERIFY 阶段发生非法 GPU 内存访问。正文说明该错误"在高负载下出现、无法稳定复现",表现为 scheduler 异常退出,栈底为:

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

出错位置在 `hybrid_linear_attn_backend.py:974` 的 `forward_batch.mamba_track_mask.any()`。复现环境:Qwen3.5-397B-A17B-FP4、`--tensor-parallel-size 4`、`--speculative-algorithm EAGLE`(`--speculative-num-steps 3`、`--speculative-eagle-topk 1`、`--speculative-num-draft-tokens 4`)、`--mamba-scheduler-strategy extra_buffer`、B200 硬件。

## 根因
正文披露根因链:
1. `prepare_for_extend()` 按完整 prefill 序列长度(如 1000 tokens)设置全部三个 Mamba 跟踪字段;
2. `prepare_for_decode()` 在投机解码下提前返回,三个字段保持不动;
3. `prepare_for_verify()` 正确把 `mamba_track_indices` 更新为 ping-pong slot 索引,但把 `mamba_track_mask` 与 `mamba_track_seqlens` 留成了过期值;
4. TARGET_VERIFY 期间 `_track_mamba_state_extend()` 用过期的 `mamba_track_mask` 去索引只有 `draft_token_num` 行(典型 5-7 行)的 h,过期索引(如 512、576、…)远超边界,触发 `cudaErrorIllegalAddress` 并导致段错误。

## 修复方案
在两个投机验证准备函数中显式清空过期的跟踪字段——`eagle_info.py` 的 `prepare_for_verify()` 与 `eagle_info_v2.py` 的 `prepare_for_v2_verify()` 均新增:

```diff
+            batch.mamba_track_mask = None
+            batch.mamba_track_seqlens = None
```

正文说明:TARGET_VERIFY 阶段不需要 prefix-cache 的 Mamba 状态跟踪,因此直接置空即可。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:`python/sglang/srt/speculative/eagle_info.py` 与 `eagle_info_v2.py` 中两处 verify 准备路径。影响 EAGLE(含 v2)投机解码 + Mamba/线性注意力混合模型(extra_buffer 调度策略)的 TARGET_VERIFY 前向;置空后这些字段不再流入 `_track_mamba_state_extend`,依赖这两个字段在 verify 阶段有值的路径会改走 None 分支(由 diff 推断)。非 EAGLE 或非 Mamba 模型路径不受影响。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
