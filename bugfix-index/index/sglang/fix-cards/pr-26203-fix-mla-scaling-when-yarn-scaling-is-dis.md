# PR #26203: Fix MLA scaling when YARN scaling is disabled

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#26203](https://github.com/sgl-project/sglang/pull/26203) |
| 作者 | BBuf |
| 合并日期 | 2026-06-11 |
| labels | run-ci, bypass-fastfail, run-ci-extra |
| 热度 | 👍 1 · 💬 104 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
YARN 关闭时 MLA 仍被再次套用 mscale 缩放,导致 Mistral/Pixtral 类 MLA 模型精度下降;现在直接返回 base scale。

## 问题现象
原生 Mistral 配置下,MLA attention 的缩放被过度施加(over-scaled):当配置把 `apply_scale=false` 映射为 `apply_yarn_scaling=False` 时,旧的 mscale 计算路径仍会再乘一次 YARN mscale,产出错误的缩放值 `0.194969494733`(正确应为 base scale `0.088388347648`)。该错误缩放最终导致源 PR 中描述的 B200 NVFP4 精度下降。

## 根因
PR 正文披露:原生 Mistral 配置可能把 `apply_scale=false` 映射为 `apply_yarn_scaling=False`;此时 MLA attention 缩放应保留 base scale,而不应再施加 YARN mscale。此前的 `compute_mla_mscale_scaling` 没有检查这两个开关,只要进了 YARN 分支就会重复缩放,造成 Mistral/Pixtral 风格 MLA 配置的 attention 被过度缩放。

## 修复方案
在 `python/sglang/srt/configs/model_config.py` 的 `compute_mla_mscale_scaling` 函数开头加一个早退分支:`apply_yarn_scaling` 或 `apply_scale` 任一被显式关闭时直接返回 `base_scaling`,不再计算 YARN mscale。关键 diff:

```diff
+    if not rope_scaling.get("apply_yarn_scaling", True) or not rope_scaling.get(
+        "apply_scale", True
+    ):
+        return base_scaling
+
     mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
```

同时新增回归测试 `test/registered/unit/configs/test_model_config_scaling.py`,覆盖三个场景:YARN 关闭时返回 base scale、默认开启时仍施加缩放(大于 base scale)、原生 Mistral `apply_scale=False` 时返回 base scale。

验证:B200 容器内 `pytest -q test/registered/unit/configs/test_model_config_scaling.py` 2 passed;`_remap_mistral_yarn_args` 冒烟确认 `apply_scale=False` 映射到 `apply_yarn_scaling=False` 后,修复路径返回 base scale `0.088388347648`。

## 影响与回归风险
- 回归面:该函数被 DeepSeek、BailingMoe、SarvamMLA 及类似 MLA 模型使用(docstring 注明)。改动只影响 `apply_yarn_scaling` 或 `apply_scale` 被显式置 false 的配置——这些配置从「错误地再缩放一次」变为「保留 base scale」;默认行为(两个开关缺省为 True)不受影响。若某模型此前依赖旧路径在开关关闭时仍缩放(属于依赖 bug 的场景),数值会发生变化。

## 关联
- 源 PR:BBuf/sglang#4(正文注明本 PR 提取其最小精度修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
