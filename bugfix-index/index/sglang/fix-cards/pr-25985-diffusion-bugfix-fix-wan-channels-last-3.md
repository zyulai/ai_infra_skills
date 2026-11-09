# PR #25985: [diffusion][bugfix] Fix Wan channels_last_3d VAE decode corruption

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#25985](https://github.com/sgl-project/sglang/pull/25985) |
| 作者 | IPostYellow |
| 合并日期 | 2026-05-22 |
| labels | run-ci, diffusion |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
Wan VAE 的 Conv3d 权重以 channels_last_3d 格式存储时输入未匹配该格式导致解码画面花屏;修复为卷积前把输入转成与权重相同的 memory format,输出再恢复 contiguous。

## 问题现象
Wan VAE 解码时,当 Conv3d 权重以 `channels_last_3d` memory format 存储,生成的视频出现视觉损坏/花屏。在当前 main 上,以 8 GPU 服务 Wan2.2 I2V 且关闭所有 offload 时可复现(正文附了修复前/后视频对比与完整复现命令:`sglang serve --model-path Wan2.2-I2V-A14B-Diffusers --dit-cpu-offload false ... --vae-cpu-offload false --pin-cpu-memory false --num-gpus 8`,再 POST `/v1/videos`)。

## 根因
正文未披露具体机制(正文 Motivation 只说明做法:让 Wan causal Conv3d 的输入在跑卷积前匹配 Conv3d 权重的 memory format,之后把输出恢复为 contiguous 以保持下游 tensor 布局不变)。(由 diff 推断)PyTorch 的 Conv3d 在输入与权重 memory format 不一致(权重 channels_last_3d、输入 contiguous)时不会得到正确的卷积输出,Wan VAE 解码路径上恰好出现这一组合,导致图像数据错乱。

## 修复方案
1. `python/sglang/multimodal_gen/runtime/models/vaes/parallel/wan_common_utils.py`:新增 `_conv3d_weight_is_channels_last_3d(weight)`(5 维且 `is_contiguous(memory_format=torch.channels_last_3d)`)与 `match_conv3d_input_format(x, weight)`(5 维输入匹配权重格式转 channels_last_3d);在 Wan 因果卷积 forward 中调用:

```python
+        x = match_conv3d_input_format(x, self.weight)
         return super().forward(x)
```

2. `wan_dist_utils.py`:并行 decode 的 `split_for_parallel_decode`/`ParallelResidualBlock` forward 中同样在 `super().forward(x_padded)` 前做 `match_conv3d_input_format`;新增 `_maybe_contiguous_for_sp_gather(x)`,在 `gather_and_trim_height` 与 attention 的 `all_gather(dim=-2)` 前把 channels_last_3d 张量转回普通 contiguous,避免 SP gather 路径布局问题。
3. `python/sglang/multimodal_gen/test/test_utils.py`:更新 CI 一致性 ground-truth 数据 revision(94eab4f→5b728ad)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:影响 Wan VAE 的因果卷积与 SP 并行解码路径:权重为 channels_last_3d 时输入会多一次格式转换(有额外拷贝开销),权重为普通 contiguous 时 `match_conv3d_input_format` 原样返回;`all_gather` 前对 5 维 channels_last_3d 张量多一次 `.contiguous()`。测试数据 revision 更新会影响引用该 GT 的 diffusion 一致性测试。

## 关联
正文未列出 issue/PR 链接(附修复前后视频对比链接 user-attachments assets)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
