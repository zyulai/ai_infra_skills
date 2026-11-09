# PR #42760: [Fix] model_loader: Avoid unsafe pickle fallback when loading PyTorch checkpoints

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#42760](https://github.com/sgl-project/sglang/pull/42760) |
| 作者 | nguyendvk |
| 合并日期 | 2026-10-27 |
| labels | deepseek, run-ci, diffusion |
| 热度 | 👍 0 · 💬 32 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论

加载 .pt 权重时对 legacy tar 格式回退到 `weights_only=False` 的 pickle 反序列化,恶意 checkpoint 可执行任意代码;移除回退并统一 `weights_only=True`。

## 问题现象

正文未描述运行期故障现象,本 PR 属于安全加固类:`torch.load(weights_only=False)` 用 pickle 反序列化,一个被构造的 checkpoint 在加载时即可执行代码。触发条件是加载这类不可信/恶意权重文件。暴露点有两个:`_load_pt_file` 对 legacy tar 格式的 .pt checkpoint 会先尝试 `weights_only=True`,失败后回退到 `weights_only=False`;此外 `deepseek_ocr.py`(两处)、`roberta.py`、`rife_interpolator.py` 在加载模型文件时调用 `torch.load` 未传 `weights_only=True`。影响面为所有走这些加载路径的用户,与 #35653(RIFE)属同类问题。

## 根因

正文披露:`torch.load(weights_only=False)` 以 pickle 反序列化,构造的 checkpoint 可在加载时执行代码;`_load_pt_file` 为兼容 legacy tar checkpoint(无法用 `weights_only=True` 加载)保留了该不安全回退,另有若干 loader 直接调用 `torch.load` 时缺少 `weights_only=True`。

## 修复方案

- 移除 `python/sglang/srt/model_loader/weight_utils.py` 与 `python/sglang/multimodal_gen/runtime/loader/weight_utils.py` 中 `_load_pt_file` 的 `weights_only=False` 回退,统一为:

```diff
-    try:
-        return torch.load(bin_file, map_location="cpu", weights_only=True)
-    except RuntimeError as e:
-        if "legacy .tar format" in str(e):
-            logger.warning(
-                "Loading %s with weights_only=False (legacy tar format)",
-                os.path.basename(bin_file),
-            )
-            return torch.load(bin_file, map_location="cpu", weights_only=False)
-        raise
+    """Load a PyTorch checkpoint file with weights_only=True (no pickle execution)."""
+    return torch.load(bin_file, map_location="cpu", weights_only=True)
```

- `deepseek_ocr.py`(两处)、`roberta.py`、`rife_interpolator.py` 的 `torch.load` 补上 `map_location` 与 `weights_only=True`。
- 新增单测:张量 state dict 仍可正常加载;legacy tar checkpoint 中夹带的 pickle payload 不会被执行(断言 marker 文件未创建)。
- `srt/debug_utils` 保持不变:它读取的是开发者自己写的 tensor dump,不是模型权重。

## 影响与回归风险

- 影响窗口:正文未说明。
- 回归面:legacy tar 格式的 .pt checkpoint 将不再被兼容,现在会直接以 torch 自身的报错失败,需要离线转换为 safetensors 或 zip 格式的 `.pt`;其余改动仅涉及 checkpoint 反序列化路径,不触碰模型 forward 代码(正文注明 Accuracy/Speed 测试不适用)。

## 关联

- #35653(RIFE,同类问题)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
