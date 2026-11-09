# PR #31957: fix(vlm): reject Moss vision metadata mismatches

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#31957](https://github.com/sgl-project/sglang/pull/31957) |
| 作者 | mickqian |
| 合并日期 | 2026-07-28 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
Moss-VL 视觉元数据与 image token 数不一致时被静默截断仍接受请求;改为 fail-closed 抛 ValueError。

## 问题现象
正文未描述线上/运行期现象,属于防御性修复。旧行为:`_compute_vision_position_ids` 在元数据与 token 计数不一致时用 `min(...)` 静默截断两侧,任一侧缺失时提前返回——两条路径都可能在接受请求的同时构造出部分或空的 vision position metadata。修复后这类 mismatch 会被拒绝。

## 根因
正文 Root cause 披露:`_compute_vision_position_ids` 此前取 metadata 计数与 token 计数中的较小者;若任一侧缺失则提前返回。两条路径都可能在接受请求的同时静默构造出部分或空的 vision position metadata。处理器现在 fail-closed,抛出带具体计数的 `ValueError`;计数相等的合法输入保持原来的位置计算不变。

## 修复方案
`python/sglang/srt/multimodal/processors/moss_vl.py` 的 `_compute_vision_position_ids`:
- `frame_count != image_token_count` 时抛 `ValueError("Moss-VL vision metadata must map one-to-one to image tokens: found X frame(s) and Y token(s)")`;
- `frame_count == 0` 时保留原提前返回路径(rope_deltas 计算);
- 删除按 `num_matches = min(...)` 截断的逻辑,`flat_eff_h/flat_w/flat_vis_starts` 与 `target_indices` 直接用完整数组构造。

```diff
+        frame_count = len(flat_eff_h)
+        image_token_count = len(image_token_indices)
+        if frame_count != image_token_count:
+            raise ValueError(
+                "Moss-VL vision metadata must map one-to-one to image tokens: "
+                f"found {frame_count} frame(s) and {image_token_count} token(s)"
+            )
```

- 新增 `test/registered/unit/models/test_moss_vl_processor.py`,覆盖 metadata 多、token 多、任一侧为空三种 mismatch 及一个合法匹配用例。

验证:NVIDIA GB300、CPU/mocked unit path `5 passed in 8.96s`;改动文件通过 `ruff check` 与 `git diff --check`。

## 影响与回归风险
- 回归面:Moss-VL 多模态处理路径。此前能被 `min(...)` 静默容忍的不一致输入现在会直接抛 `ValueError` 拒绝请求,依赖旧截断行为的调用方会失败(由 diff 推断);合法等计数输入的位置计算不变(正文)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
