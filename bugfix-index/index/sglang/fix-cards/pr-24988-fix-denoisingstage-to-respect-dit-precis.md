# PR #24988: Fix DenoisingStage to respect dit_precision config instead of hardcoded bfloat16

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24988](https://github.com/sgl-project/sglang/pull/24988) |
| 作者 | niehen6174 |
| 合并日期 | 2026-05-15 |
| labels | run-ci, diffusion |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
`DenoisingStage` 把 `target_dtype` 硬编码为 bfloat16,无视用户配置的 `dit_precision`(fp16/fp32 被强制截断,精度损失、生成质量退化);改为查 `PRECISION_TO_TYPE[dit_precision]`,与其他 stage 对齐。

## 问题现象
正文披露:多模态生成(multimodal_gen)流水线的 `DenoisingStage` 中,即使显式设置 `dit_precision` 为 `fp16` 或 `fp32`,去噪阶段仍把张量强制 cast 成 `bfloat16`,造成精度损失(如 fp32 → bfloat16 截断),可能劣化生成质量——对以 fp16 或 fp32 训练/微调的模型尤其明显。`dit_precision` 为 `bfloat16`(常见默认)时行为与之前完全一致,因此该 bug 只影响配置为 fp16/fp32 的用户。

## 根因
正文披露:`DenoisingStage._prepare_denoising_loop` 里 `target_dtype = torch.bfloat16` 是硬编码常量,没有读取 `pipeline_config.dit_precision`;而其他 stage(如 `helios_denoising`、`mova`)都正确地用 `PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]` 决定目标 dtype,属于该 stage 与其余流水线不一致的遗漏。

## 修复方案
改 `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py` 的 `_prepare_denoising_loop`,一行替换:

```diff
-        target_dtype = torch.bfloat16
+        target_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
```

后续 `autocast_enabled = (target_dtype != torch.float32) and not server_args.disable_autocast` 逻辑不变,fp32 配置时自动不启用 autocast。

## 影响与回归风险
- 回归面:仅 `DenoisingStage` 的 dtype 选择。`dit_precision=bfloat16`(默认)时查表结果仍是 bfloat16,行为不变;`fp16`/`fp32` 配置下从此真正生效——fp16 时 autocast 开启但目标 dtype 变为 fp16,fp32 时 autocast 关闭。作者说明这是「一行配置查表替换硬编码常量」,无性能影响;潜在回归面是此前一直跑 bfloat16 的 fp16/fp32 配置用户,修复后输出数值会变化(属预期修正)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
