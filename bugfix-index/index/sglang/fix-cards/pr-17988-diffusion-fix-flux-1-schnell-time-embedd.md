# PR #17988: [Diffusion] Fix FLUX.1-schnell time embedding argument mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#17988](https://github.com/sgl-project/sglang/pull/17988) |
| 作者 | BBuf |
| 合并日期 | 2026-01-31 |
| labels | run-ci, diffusion |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
FLUX.1-schnell 不支持 guidance embeds 却被传入 guidance 参数,调用 time embedding 报 TypeError,现按 config.guidance_embeds 双条件判断修复。

## 问题现象
用 `sglang.multimodal_gen.runtime.launch_server` 启动 FLUX.1-schnell(带 `--attention-backend fa3 --pin-cpu-memory --warmup --enable-torch-compile`),客户端以 guidance_scale=0.0、4 步请求 `/v1/images/generations` 生成 1024x1024 图像时,DenoisingStage 执行约 4.0582 ms 后报错:

```
[DenoisingStage] Error during execution after 4.0582 ms: CombinedTimestepTextProjEmbeddings.forward() takes 3 positional arguments but 4 were given
```

调用栈落在 `python/sglang/multimodal_gen/runtime/models/dits/flux.py:531` 的 `self.time_text_embed(timestep, guidance, pooled_projections)`。

## 根因
正文披露:
- FLUX.1-schnell 的 config 为 `guidance_embeds=False`,因此使用只接受 2 个参数(timestep、pooled_projections)的 `CombinedTimestepTextProjEmbeddings`。
- 原代码只用 `if guidance is None` 判断是否传 guidance,没有校验模型本身是否支持 guidance embeddings;即使 guidance 为 0.0 也会按 3 参数调用,触发 TypeError。

## 修复方案
改 `python/sglang/multimodal_gen/runtime/models/dits/flux.py` 的 `forward`:只有 `self.config.guidance_embeds` 为真且 `guidance is not None` 时才把 guidance 传给 `time_text_embed`,否则只传 timestep 与 pooled_projections:

```diff
-        temb = (
-            self.time_text_embed(timestep, pooled_projections)
-            if guidance is None
-            else self.time_text_embed(timestep, guidance, pooled_projections)
-        )
+        # Only pass guidance to time_text_embed if the model supports it
+        if self.config.guidance_embeds and guidance is not None:
+            temb = self.time_text_embed(timestep, guidance, pooled_projections)
+        else:
+            temb = self.time_text_embed(timestep, pooled_projections)
```

修复后正文附了成功生成图片的截图。

## 影响与回归风险
- 影响窗口:正文说明问题存在于 main 分支(复现命令标注 "In main");未给出具体版本范围。
- 回归面:`flux.py` 中 FLUX 系列模型的 time embedding 调用路径。对 `guidance_embeds=True` 的模型(如 FLUX.1-dev)行为不变;对 `guidance_embeds=False` 的模型(如 FLUX.1-schnell),即使请求带 guidance_scale 也不再把它传入 embedding,输出会忽略 guidance(这本来就是此类模型的正确语义)。

## 关联
- [FLUX.1-schnell config.json](https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/transformer/config.json#L5)(`guidance_embeds=False` 依据)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
