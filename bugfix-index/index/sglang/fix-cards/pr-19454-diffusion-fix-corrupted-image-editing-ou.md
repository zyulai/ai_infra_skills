# PR #19454: [Diffusion] Fix corrupted image editing outputs in Multi-GPU SP mode for FLUX.2-klein models

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19454](https://github.com/sgl-project/sglang/pull/19454) |
| 作者 | ruihanglix |
| 合并日期 | 2026-03-04 |
| labels | run-ci, diffusion |
| 热度 | 👍 2 · 💬 5 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

多 GPU SP 模式图像编辑输出完全损坏:image_latent 未分片、RoPE 分片错位、文本 token 被复制 SP 倍,三处一并修复。

## 问题现象

以 `--num-gpus 8` 服务 `FLUX.2-klein-9B` 时,图像编辑接口(`/v1/images/edits`)输出完全损坏(灰色噪声/彩色斑块),而文生图正常;服务端不报任何错误,损坏完全静默。作者在 8× NVIDIA H200、`--num-gpus 8 --sp-degree 8`(与 bug 报告相同配置)复现:修前吉他编辑结果「长出两个琴颈」等结构性伪影,修后 8 卡输出与 1 卡参考质量一致。

## 根因

正文披露 SP(Sequence Parallelism)路径上三个相互独立的 bug:

1. **`image_latent` 未按 SP 分片**。`_preprocess_sp_latents` 只对噪声 latent 做分片,条件图 latent(`batch.image_latent`)保持全尺寸;去噪循环里 `torch.cat([latents, batch.image_latent], dim=1)` 时形状不匹配:`[B, 128, D] + [B, 1024, D]` 而非 `[B, 128, D] + [B, 128, D]`。
2. **RoPE 位置分片错误**。`get_freqs_cis` 先把 `latent_ids`(1024)与 `condition_image_latent_ids`(1024)拼成 2048 长序列再分片;rank 0 收到位置 0–255(全是噪声位置),但其本地 token 实际是 noise[0:128] + image[0:128],图像 token 拿到了噪声位置编码。
3. **文本 token 在 Ulysses all-to-all 中被复制 SP 倍**。FLUX joint attention 里文本 token 被各 SP rank 复制后仍与图像 token 一起过 `USPAttention` 的 all-to-all,聚合序列含 `txt_len × SP` 个文本 token;SP=8 时每个文本 key 出现 8 次,文本注意力权重膨胀约 8 倍,挤压输入图像保真度,产生结构性伪影。

文生图受影响较小,因为 bug 1、2 不适用(`batch.image_latent` 为 `None`),bug 3 只强化文本条件、没有输入图像信号可被挤压。

## 修复方案

4 个文件、+94/-29 行:

1. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py` — `_preprocess_sp_latents` 中 `image_latent` 存在即分片,不再限定 `WanI2V480PConfig`:

```diff
-        if (
-            isinstance(server_args.pipeline_config, WanI2V480PConfig)
-            and batch.image_latent is not None
-        ):
-            batch.image_latent, _ = server_args.pipeline_config.shard_latents_for_sp(
-                batch, batch.image_latent
-            )
+        if batch.image_latent is not None:
+            batch.image_latent, _ = server_args.pipeline_config.shard_latents_for_sp(
+                batch, batch.image_latent
+            )
```

2. `python/sglang/multimodal_gen/configs/pipeline_configs/flux.py` — `get_freqs_cis` 改为噪声 token 与条件图 token **各自独立**分片 RoPE 后再拼接,而不是先拼 position IDs 再分片。

3. `python/sglang/multimodal_gen/runtime/layers/attention/layer.py` — `USPAttention` 新增 `_forward_with_replicated_prefix`:复制前缀(文本 token)不进 all-to-all,只本地切到同一 head 分片 `H_local` 再与 gathered 图像序列拼接,文本恰好出现一次;输出侧 all-to-all 回图像后缀、all-gather 文本前缀。每层 3 次通信原语(2×all-to-all + 1×all-gather),非单卡回退。

4. `python/sglang/multimodal_gen/runtime/models/dits/flux_2.py` — `Flux2Attention`/`Flux2ParallelSelfAttention`/`Flux2SingleTransformerBlock`/`Flux2Transformer2DModel` 透传 `num_replicated_prefix`(即 `txt_len`)。

另附带新增 `flux_2_klein_ti2i_2_gpus` 性能基线与测试用例、修复 `test_server_utils.py` 中 `Path` 局部变量小问题。

## 影响与回归风险

- 回归面:所有多模态生成管道的 `_preprocess_sp_latents` 路径(`image_latent` 由仅 Wan 特例改为统一分片)、FLUX.2 的 RoPE 计算与所有经 `USPAttention` 的 SP 注意力前向;去噪阶段性能不变(纯正确性修复,无计算图/内核改动)。附带改动触及 multimodal_gen 测试基线与用例配置。

## 关联

- Fixes #19449

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
