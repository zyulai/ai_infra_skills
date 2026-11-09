# PR #17916: [Diffusion][MOVA] fix: resolve library mismatch in scheduler and update dit offload method name

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#17916](https://github.com/sgl-project/sglang/pull/17916) |
| 作者 | CloudRipple |
| 合并日期 | 2026-01-29 |
| labels | run-ci, diffusion |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Overlap / Scheduler |
| bug 类型 | correctness |

## 一句话结论
MOVA scheduler 组件加载时库标识未映射到 diffusers 造成库不匹配,且 dit 调用了已更名的 offload 方法;补映射并把调用改为 `prepare_for_next_req`。

## 问题现象
正文未描述具体现象(无报错日志或复现说明),仅 Motivation 说明修复目标:`component_loader.py` 的 `for_module_type` 在 `module_type` 为 `"scheduler"`、backend 为 `"mova.diffusion.schedulers.flow_match_pair"` 时需要正确映射到 `"diffusers"` 以保证一致的模块加载;标题另指出 dit offload 方法名需要更新。

## 根因
正文披露:`for_module_type` 中,scheduler 模块在 `transformers_or_diffusers` 为 `"mova.diffusion.schedulers.flow_match_pair"` 时未被重映射为 `"diffusers"`,与加载器其余逻辑(library 匹配)不兼容。第二个改动(标题中的 dit offload 方法名)正文未单独披露原因;由 diff 推断:mova.py 中对 `OffloadableDiTMixin` 调用的 `prepare_for_next_denoise()` 方法已被更名为 `prepare_for_next_req()`,旧调用点未同步更新。

## 修复方案
1. `python/sglang/multimodal_gen/runtime/loader/component_loader.py` `for_module_type`:在既有 video_dit 等模块强制 `diffusers` 的分支后新增——`module_type == "scheduler"` 且 `transformers_or_diffusers == "mova.diffusion.schedulers.flow_match_pair"` 时,将 `transformers_or_diffusers` 重映射为 `"diffusers"`;
2. `python/sglang/multimodal_gen/runtime/models/model_stages/mova.py` `forward`:遍历 `video_dit / video_dit_2 / audio_dit`,对 `OffloadableDiTMixin` 的调用由 `dit.prepare_for_next_denoise()` 改为 `dit.prepare_for_next_req()`。

```diff
+        if (
+            module_type == "scheduler"
+            and transformers_or_diffusers == "mova.diffusion.schedulers.flow_match_pair"
+        ):
+            transformers_or_diffusers = "diffusers"
```

## 影响与回归风险
- 影响窗口:正文未说明;影响 multimodal_gen 加载 MOVA 模型 scheduler 组件及 MOVA 推理 denoise 迭代时的 dit offload 路径。
- 回归面:组件加载器的 library 映射判断(仅命中 MOVA flow_match_pair scheduler 这一特例,其他模块映射不变);MOVA forward 中 offload 准备调用——若 `prepare_for_next_req` 在其他 OffloadableDiT 实现上语义与旧方法不同,行为随之改变(正文未说明)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
