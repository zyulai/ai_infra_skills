# PR #7887: fix: modality length mismatch with image_data

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7887](https://github.com/sgl-project/sglang/pull/7887) |
| 作者 | Yangruipis |
| 合并日期 | 2025-07-15 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

未处理的 image_data 输入使 modalities 与 image_data 长度不等;else 分支补 append None 保持一致。

## 问题现象

当用户传入未被现有分支处理的 `image_data` 输入时,`GenerateReqInput` 中 `len(self.modalities)` 与 `len(self.image_data)` 不相等(正文 Motivation)。正文未描述具体的报错信息、复现命令与影响面。

## 根因

正文未披露根因。由 diff 可描述机制:`io_struct.py` 的 `_normalize_image_data` 中,只有当 `image_data[i]` 命中前面的分支(单个 image、`len > 1` 记为 multi-images)时才会 `self.modalities.append(...)`;不满足任何分支的情况直接落空,该元素没有对应的 modality 记录,随后两列表按 `parallel_sample_num` 扩展,长度不一致被放大(由 diff 推断)。

## 修复方案

改 `python/sglang/srt/managers/io_struct.py` 的 `_normalize_image_data`,补一个 else 分支:

```diff
                     elif len(self.image_data[i]) > 1:
                         self.modalities.append("multi-images")
+                    else:
+                        # Ensure len(self.modalities) == len(self.image_data)
+                        self.modalities.append(None)
                 # Expand parallel_sample_num
                 self.image_data = self.image_data * self.parallel_sample_num
                 self.modalities = self.modalities * self.parallel_sample_num
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:多模态请求的 `modalities` 归一化路径。行为变化:未匹配到已知模态分支的元素现在会得到 `None` 占位,`modalities` 与 `image_data` 长度恒等;下游如何处理 `None` 值本 PR 正文与可见 diff 未涉及。命中原有分支(image、multi-images)的行为不变。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
