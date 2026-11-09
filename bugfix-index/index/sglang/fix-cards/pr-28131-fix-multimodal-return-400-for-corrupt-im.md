# PR #28131: fix(multimodal): return 400 for corrupt image inputs

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28131](https://github.com/sgl-project/sglang/pull/28131) |
| 作者 | Sunt-ing |
| 合并日期 | 2026-09-25 |
| labels | run-ci |
| 热度 | 👍 2 · 💬 8 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
损坏/截断图片字节经 PIL 懒加载被包成 RuntimeError 返回 500;强制解码并翻译成 ValueError,让服务端按 400 返回。

## 问题现象
OpenAI 兼容 VLM 请求中携带损坏、截断或非图片字节时,本属客户端输入错误,却会被服务端当作内部错误返回 HTTP 500。作者用 Qwen2.5-VL-3B-Instruct 在 A800-80GB 上 E2E 复现:非图片字节、截断 PNG、截断 JPEG、垃圾字节、双图之一损坏等输入在基线上均返回 500;修复后均返回 400(如 `Failed to load image: cannot identify image file`、`Failed to load image: image file is truncated`、`Failed to load image: Truncated File Read`);12 并发(6 有效 + 6 损坏)从 6x200 + 6x500 变为 6x200 + 6x400。同一矩阵还在 fast path 的 `Qwen2.5-VL-3B`、`InternVL2-1B`、`Gemma-3-4B`、`GLM-4.1V-9B`、`Pixtral-12B` 与 legacy path 的 `MiniCPM-V-2_6` 上复现,`/health` 在损坏输入风暴后仍为 200。

## 根因
正文披露:`_load_image` 返回懒加载的 `Image.open(...)` 结果,既不强制解码也不翻译 PIL 解码失败;当该失败传播到 multimodal loader 时,fast 与 legacy 两条聚合路径把它包装成 `RuntimeError`,而服务端把 `RuntimeError` 视为内部错误(500)。#27451 已把畸形 base64 在 `_load_single_item` 保持在 400,本 PR 补上剩余的损坏图片字节路径。

## 修复方案
- `python/sglang/srt/utils/common.py`:`_load_image` 经 `_fully_load_pil_image` 强制 `image.load()` 解码,并把 `OSError` / `SyntaxError` 翻译为 `ValueError`:
```python
except (OSError, SyntaxError) as e:
    raise ValueError(f"Could not decode image: {e}") from e
```
(正文还列出 `base_processor.py` 的改动:让 `ValueError` 在 `fast_load_mm_data` 与 `legacy_load_mm_data` 中先于泛化的 `Exception -> RuntimeError` 包装传播;该文件改动未包含在本 pack 的 diff 摘录中。)
- 新增 CPU 注册单测 `TestLoadImage::test_corrupt_image_bytes_raise_value_error`,覆盖有效、垃圾、空、截断 PNG、坏 PNG chunk 五类输入。
- 服务端已把 `ValueError` 映射为 400;强制解码不增加 happy path 的额外解码(有效图片反正会在预处理阶段解码),只是把失败点提前到更重的路径之前。GPU nvJPEG fast path 保持正确:缺尾部 marker 的截断 JPEG 不进 `decode_jpeg`;marker 有效但内容损坏的 JPEG 使 `decode_jpeg` 抛错、回退 CPU 路径,修复后返回 400。

## 影响与回归风险
- 影响窗口:正文未说明(仅提及 E2E 在 `sglang==0.5.12.post1` 验证环境运行,再移植到 `upstream/main`)。
- 回归面:PIL 图片加载与多模态数据聚合的异常处理路径(fast 与 legacy 两条 loader 路径)。有效请求不受影响(上表 200 不变);风险在于把原本可能属于服务端问题的解码失败也归为 400。

## 关联
- [SGLang #27451](https://github.com/sgl-project/sglang/pull/27451)(已把畸形 base64 保持为 400)
- [#26995](https://github.com/sgl-project/sglang/pull/26995)(eager decode 提吞吐,但未翻译/保留 PIL 解码错误)
- [vllm-project/vllm#38253](https://github.com/vllm-project/vllm/pull/38253)(行为对齐对象)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
