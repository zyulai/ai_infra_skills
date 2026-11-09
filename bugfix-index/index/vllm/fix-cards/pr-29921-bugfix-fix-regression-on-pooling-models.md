# PR #29921: [Bugfix] Fix regression on pooling models from PR#29621

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#29921](https://github.com/vllm-project/vllm/pull/29921) |
| 作者 | ywang96 |
| 合并日期 | 2025-12-03 |
| labels | ready, multi-modality |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#29621 后 EmbeddingItems.get 返回 MediaWithBytes 包装对象而非张量,encoder-decoder 多模态示例失败;加 _unwrap 解包修复。

## 问题现象
CI 示例测试失败(buildkite vllm/ci 构建 41568),复现命令为 `python3 offline_inference/encoder_decoder_multimodal.py --model-type whisper --seed 0`(标题指明受影响的是 pooling 类多模态模型)。正文未贴出具体报错信息。

## 根因
正文未披露具体机制,仅说明这是 PR#29621 引入的回归。(由 diff 推断)#29621 之后,`EmbeddingItems.data` 中的元素可能是 `MediaWithBytes[torch.Tensor]` 包装对象而非裸 `torch.Tensor`;`get(index)` 直接返回该包装对象,下游按张量消费时失败。修复在 `get()` 中检测 `MediaWithBytes` 并解包出 `item.media`,印证了"数据被包装、调用方未解包"这一机制。

## 修复方案
`vllm/multimodal/parse.py` 的 `EmbeddingItems` 类新增 `_unwrap` 辅助方法,并让 `get()` 经其返回:

```diff
+    def _unwrap(
+        self, item: torch.Tensor | MediaWithBytes[torch.Tensor]
+    ) -> torch.Tensor:
+        """Extract media from wrapper if present."""
+        return item.media if isinstance(item, MediaWithBytes) else item
+
     def get_count(self) -> int:
         return len(self.data)

     def get(self, index: int) -> torch.Tensor:
-        return self.data[index]
+        return self._unwrap(self.data[index])
```

## 影响与回归风险
- 影响窗口:#29621 合并之后至本 PR 之前,使用 `EmbeddingItems` 的多模态 pooling / encoder-decoder 模型(如 whisper 示例)受影响。
- 回归面:仅 `EmbeddingItems.get()` 一条路径;`get_count()`、`get_processor_data()` 不变。若 `data` 中本就是裸张量,`_unwrap` 原样返回,无行为变化;若为包装对象,则恢复到 #29621 之前"返回张量"的契约。`_unwrap` 同时写入了 `get` 的类型签名(`torch.Tensor | MediaWithBytes[torch.Tensor]`),后续维护者可直接看出两种可能形态。

## 关联
- PR #29621(标题所指的回归来源)
- buildkite 构建失败链接:https://buildkite.com/vllm/ci/builds/41568#019adfe2-6075-4032-aea5-8b8b48bc5a5a

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
