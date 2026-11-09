# PR #30789: [ROCm] [Bugfix] Fix torch sdpa hallucination

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#30789](https://github.com/vllm-project/vllm/pull/30789) |
| 作者 | tjtanaa |
| 合并日期 | 2025-12-16 |
| labels | rocm, ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
ROCm 上 torch SDPA 因缺少 contiguous 处理令视觉模型幻觉,补回该逻辑修复。

## 问题现象
ROCm 平台上,基于 torch SDPA 的视觉语言模型出现明显幻觉:用 `examples/offline_inference/vision_language.py` 对东京晴空树(樱花环绕)图片做问答,qwen3_vl 修复前输出描述为 "LED light strip"/"LED panel" 等错误内容;qwen2_5_vl 修复前输出 "roller coaster"、"dancers or performers" 等完全无关内容。修复后两个模型均正确识别为 Tokyo Skytree + 樱花。影响面为 ROCm 上走 torch SDPA attention 后端的 VL 模型推理。

## 根因
正文披露:PR #30125 把 PR #27744 中针对 ROCm 的必要处理(contiguous 逻辑)删掉了,导致 ROCm 上该后端出现幻觉。即一次重构误删了 ROCm 必需的张量连续化步骤,属于回归。

## 修复方案
在 `vllm/attention/ops/vit_attn_wrappers.py` 的 `torch_sdpa_wrapper` 中重新加上 ROCm 平台的 contiguous 处理:新增 `from vllm.platforms import current_platform`,在计算前判断 `current_platform.is_rocm()`,对 q/k/v 各做 `.contiguous()`:

```diff
+    # Never remove the contiguous logic for ROCm
+    # Without it, hallucinations occur with the backend
+    if current_platform.is_rocm():
+        q = q.contiguous()
+        k = k.contiguous()
+        v = v.contiguous()
+
     outputs = []
```

## 影响与回归风险
- 影响窗口:自 #30125 删除该逻辑起、至本 PR 合并止(两者提交号正文未给出,仅给 PR 号)。
- 回归面:仅 `torch_sdpa_wrapper` 一处;且 contiguous 仅在 ROCm 平台分支执行,CUDA 路径不受影响。非连续张量在 ROCm 上会额外产生一次拷贝,理论上有轻微性能开销。

## 关联
- https://github.com/vllm-project/vllm/pull/30125
- https://github.com/vllm-project/vllm/pull/27744

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
