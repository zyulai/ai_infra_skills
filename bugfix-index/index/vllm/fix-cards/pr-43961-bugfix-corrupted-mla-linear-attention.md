# PR #43961: [Bugfix] Corrupted MLA + linear attention

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43961](https://github.com/vllm-project/vllm/pull/43961) |
| 作者 | gau-nernst |
| 合并日期 | 2026-05-29 |
| labels | bug, ready, v1 |
| 热度 | 👍 1 · 💬 6 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
Kimi Linear(MLA+线性注意力)KV cache 满后输出乱码;把 MLAAttentionSpec 纳入 new_block_ids 记录列表修复。

## 问题现象
Kimi Linear 模型在 KV cache 被填满后开始输出垃圾(garbage)。作者用多轮 gsm8k 填满 KV cache 复现:main 上 epoch 0 得分 0.9075,epoch 1 降到 0.7096,epoch 2 起崩至 0.0136 / 0.0061 / 0.0144;本 PR 后五个 epoch 稳定在 0.8886-0.8969。正文说明该问题此前未暴露,是因为 Kimi Linear 是唯一的 MLA + Linear attention 模型。另注意:问题只影响 FlashMLA 与 FlashInfer MLA 后端,Triton MLA 不受影响。

## 根因
正文披露:该问题与 #35219 相关——该 PR 为 Qwen 修过同类问题,但 Kimi Linear 用的是 MLA,未被那个修复覆盖。正文进一步指出了出错的后端机制:FlashMLA 和 FlashInfer MLA 不对尾部 KV cache 页做掩码,依赖乘 `* 0` 把无效位置清零,而这一性质仅对非 nan、非 inf 的输入成立;缓存满后无效页内容不再保证为零,输出随之被污染。至于 #35219 修的 `new_block_ids` 记录机制为何漏掉 MLA(即 MLAAttentionSpec 需要与 FullAttentionSpec 一样把新分配块 id 记入该列表),作者自述对 KV cache 管理代码不熟,未在正文给出完整解释,属(由 diff 推断):修复即把 `MLAAttentionSpec` 加进两处 spec 类型判断,使其走与 full attention 相同的新块 id 记录路径。

## 修复方案
改 `vllm/v1/core/single_type_kv_cache_manager.py` 两处(`allocate_new_computed_blocks` 与 `allocate_new_blocks`),把 spec 类型判断从 `(FullAttentionSpec, TQFullAttentionSpec)` 扩为包含 `MLAAttentionSpec`:

```diff
-            if type(self.kv_cache_spec) in (FullAttentionSpec, TQFullAttentionSpec):
+            if type(self.kv_cache_spec) in (
+                FullAttentionSpec,
+                TQFullAttentionSpec,
+                MLAAttentionSpec,
+            ):
                 self.new_block_ids.extend(b.block_id for b in allocated_blocks)
```

即 MLA 层新分配(含 prefill 复算)的块 id 现在同样被记录。验证为上表 gsm8k 多 epoch 对比,修复后不再随 cache 填满而退化。

## 影响与回归风险
- 影响窗口:正文未给出版本号;按指认,自 #35219 修复 Qwen 同类问题后 MLA 路径仍缺失的版本起,至本修复前。仅影响 KV cache 填满后的长时运行。
- 回归面:按 diff 只影响 MLAAttentionSpec 类型的单类型 KV cache 管理器的 `new_block_ids` 记录行为,纯管理簿记改动;FlashMLA/FlashInfer/Triton 等 attention 后端本身未动。Triton MLA 因自带输入掩码,行为不受此 bug 影响,但簿记变更同样对其生效,需留意事件/下游消费方对 MLA 块 id 的新增上报。

## 关联
- [#35219](https://github.com/vllm-project/vllm/pull/35219)(Qwen 侧同类修复,本 PR 补齐 MLA 覆盖)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
