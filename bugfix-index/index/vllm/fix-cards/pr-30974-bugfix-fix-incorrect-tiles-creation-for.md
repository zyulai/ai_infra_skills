# PR #30974: [Bugfix] Fix incorrect tiles creation for mm prefix triton attention

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#30974](https://github.com/vllm-project/vllm/pull/30974) |
| 作者 | Isotr0py |
| 合并日期 | 2025-12-19 |
| labels | ready |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
mm prefix 场景下 triton attention 的 tile 选择不完整、hidden_states 未收敛,改为对双向 mask 覆盖全部有效 tile。

## 问题现象
gemma3 等使用 mm prefix(图像双向注意力)的模型走 triton unified attention 时输出不正确。作者用 `python examples/offline_inference/vision_language.py -m gemma3 --num-prompts 1` 对比:Main 分支的 attn out 数值(如第一层 0.146484375)与 FlexAttention 参考实现(0.14453125)不一致,图像描述文本明显劣化(描述被截断、细节丢失);本 PR 修复后 attn out 与 FlexAttention 一致,图像描述恢复完整("Cherry Blossoms / Tokyo Tower" 两点结构)。

## 根因
正文披露:#30386 的实现仍然不正确,因为 tile 依然是「随意选择」的(casually selected),导致部分有效 tile 未参与计算、hidden_states 不能完全收敛("the tiles are still selected casually, so the hidden_states is not fully converged");本 PR 确保所有有效 tile 被选入计算。
由 diff 推断的具体机制:2d kernel 中 `max_seq_prefix_len` 此前一律 `tl.minimum(max_seq_prefix_len, seq_len)`(为 causal 尾部 q_block padding 而收紧),但 mm prefix 的图像双向注意力需要完整范围(包含 q_block padding)才能保证 doc mask 正确;另外 sliding window 的 tile 裁剪对双向 mask 同样不成立(由 diff 推断)。

## 修复方案
改 `vllm/attention/ops/triton_unified_attention.py` 的 `kernel_unified_attention_2d`:
```python
if USE_MM_PREFIX:
    # image bidirectional attention ranges require a full range
    # including q_block padding to make sure doc mask is correct
    max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
else:
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
```
并把 sliding window 的 tile 裁剪条件由 `if SLIDING_WINDOW > 0:` 改为 `if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:`(注释标注 TODO(Isotr0py): sliding window pruning with image bidirectional mask,即暂不支持)。非 mm prefix 路径保持原行为不变。

## 影响与回归风险
- 影响窗口:自 #30386 引入 mm prefix 支持起(2025-12-19 修复),使用 triton unified attention 的 mm prefix 模型(gemma3 等)
- 回归面:仅 2d kernel;USE_MM_PREFIX 时计算的 tile 数可能变多,存在一定性能开销;sliding window + mm prefix 组合不再做 tile 裁剪(正确性优先);普通 causal / sliding window 路径的 tile 选择逻辑未变。

## 关联
- [#30386](https://github.com/vllm-project/vllm/pull/30386)(被修正的前序实现)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
