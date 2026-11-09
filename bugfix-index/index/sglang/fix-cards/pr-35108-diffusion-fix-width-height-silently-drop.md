# PR #35108: [diffusion] Fix width/height silently dropped on multipart /v1/videos requests

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#35108](https://github.com/sgl-project/sglang/pull/35108) |
| 作者 | zetyquickly |
| 合并日期 | 2026-10-06 |
| labels | run-ci, diffusion |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
multipart `/v1/videos` 请求的 `width`/`height` 被静默丢弃、管线默认值生效;为其补上显式 `Form(None)` 参数并接入 `form_value()` 即修复。

## 问题现象
不限定模型:任何带显式 `width`/`height` 的 multipart `/v1/videos` 请求都会命中此问题。作者在对接 LTX-2.5 做 image-to-video 客户端(见 #35107)时暴露:请求 `width=704&height=512` 实际产出 `960x544` 视频(即 LTX25SamplingParams 硬编码默认值),经对实际输出的 `ffprobe` 确认。JSON body 路径不受影响(修复前后均正常)。

## 根因
PR 正文披露:`create_video()` 为 `seconds`、`size`、`fps`、`num_frames`、`seed` 等声明了显式 `Form(...)` 参数,但没有为 `width`/`height` 声明。由于这两个名字已经是 `VideoGenerationsRequest` 上已声明的字段,通用的「额外表单字段」fallback 会显式排除已声明字段,因此 multipart 请求里的 `width`/`height` 永远不会被读取——管线的 `SamplingParams` 子类默认值静默胜出。

## 修复方案
在 `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py` 的 `create_video()` 中为 `width`/`height` 增加显式 `Form(None)` 参数,并按 `fps`/`num_frames` 已有的 `form_value()` 优先级链路透传进 `VideoGenerationsRequest`:

```diff
     fps: Optional[int] = Form(None),
     num_frames: Optional[int] = Form(None),
+    width: Optional[int] = Form(None),
+    height: Optional[int] = Form(None),
     seed: Optional[int] = Form(None),
...
         fps_val = form_value("fps", fps)
         num_frames_val = form_value("num_frames", num_frames)
+        width_val = form_value("width", width)
+        height_val = form_value("height", height)
...
             fps=fps_val,
             num_frames=num_frames_val,
+            width=width_val,
+            height=height_val,
```

## 影响与回归风险
- 回归面:diff 仅改动 `video_api.py` 的 multipart 处理路径,新增两个可选 Form 参数并透传。JSON body 路径不经过该函数的表单解析,正文已复测确认修复前后行为一致。行为变化:multipart 请求中显式给出的 `width`/`height` 从「被忽略」变为「被采纳并覆盖管线默认值」。

## 修复验证
- 修复前:multipart 请求 `width=704&height=512` 产出 `960x544` 视频(`ffprobe` 确认)。
- 修复后:同一请求产出 `704x512`(`ffprobe` 确认)。
- JSON body 路径修复前后复测均不受影响。

## 关联
- [#35107](https://github.com/sgl-project/sglang/pull/35107)(构建 image-to-video 客户端时暴露此问题)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
