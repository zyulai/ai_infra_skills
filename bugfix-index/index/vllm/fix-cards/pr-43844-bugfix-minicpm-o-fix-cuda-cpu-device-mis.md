# PR #43844: [Bugfix][MiniCPM-o] Fix cuda/cpu device mismatch in Resampler2_5 pos_embed

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43844](https://github.com/vllm-project/vllm/pull/43844) |
| 作者 | parthash0804 |
| 合并日期 | 2026-06-09 |
| labels | bug, nvidia, ready, verified |
| 热度 | 👍 1 · 💬 4 |
| 分类 | Sampler / Detokenization |
| bug 类型 | correctness |

## 一句话结论
MiniCPM-o 的 Resampler2_5 把仍驻留 CPU 的非持久 pos_embed buffer 直接与 GPU 上的隐状态相加导致崩溃,一行 `.to(device=device, dtype=dtype)` 修复。

## 问题现象
MiniCPM-o 处理图像时引擎初始化即崩溃(exit code 1):
```
File "vllm/model_executor/models/minicpmv.py", line 233, in forward
    x + pos_embed,
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
...
RuntimeError: Engine core initialization failed. See root cause above.
```
复现:用 vLLM 内置 synthetic multimodal benchmark,random-mm bucket `(800, 1024, 1)`(单张 800x1024 图像,resampler 网格在 70x70 以内);修复后 benchmark 正常完成(exit 0)。

## 根因
正文披露:`pos_embed` 注册为 non-persistent buffer,常规的权重加载设备搬运不会移动它,除非有代码显式迁移,否则一直留在 CPU。是否暴露由"patch 网格尺寸 vs 缓存 pos_embed 尺寸(max_size,默认 70x70)"决定:`_adjust_pos_cache` 只在目标网格超出缓存尺寸时才在输入设备上重建 pos_embed;网格不超过 70x70 的图像会原样使用 CPU buffer,暴露设备不匹配——很大的图像反而会碰巧在 GPU 上重建 buffer,掩盖该 bug。

## 修复方案
改 `vllm/model_executor/models/minicpmv.py` 的 `Resampler2_5.forward`,取 pos_embed 切片时同时迁移设备:
```python
-                self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype)
+                self.pos_embed[:tgt_h, :tgt_w, :]
+                .reshape((tgt_h * tgt_w, -1))
+                .to(device=device, dtype=dtype)
```
与已正确的 `Resampler4_5.forward` 对齐;无 API 或行为变化。

## 影响与回归风险
- 回归面:仅 MiniCPM-o/MiniCPM-V 系列模型(`minicpmv.py`)Resampler2_5 的 forward;每次取 pos_embed 切片多一次 `.to(device)`,若 buffer 已在目标设备则为 no-op。仅影响 pos_embed 尚未重建到 GPU 的场景(网格 ≤ 默认 70x70)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
