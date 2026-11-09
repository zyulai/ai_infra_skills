# PR #38253: [Bugfix][Frontend] Return 400 for corrupt/truncated image inputs instead of 500

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38253](https://github.com/vllm-project/vllm/pull/38253) |
| 作者 | aliialsaeedii |
| 合并日期 | 2026-03-30 |
| labels | bug, multi-modality, ready |
| 热度 | 👍 2 · 💬 0 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论

截断/损坏的 base64 图片在解码时 PIL 抛 OSError 未被捕获,误返回 500;现捕获后转 ValueError 走 400。

## 问题现象

客户端发送 base64 编码、语法上合法但数据截断或损坏的图片时,vLLM 返回 HTTP 500 Internal Server Error。作者给出复现方法:把一张好图 `good.jpeg` 去掉末尾 128 字节生成 `truncated.jpeg`,base64 编码后通过 `/v1/chat/completions`(模型 meta-llama/Llama-4-Scout-17B-16E-Instruct)发 `image_url` 请求;修复前返回 `{"message":"Truncated File Read","type":"InternalServerError","code":500}`,修复后返回 `{"message":"Failed to load image: Truncated File Read","type":"BadRequestError","code":400}`。人工测试矩阵显示截断的 JPEG/PNG/WebP 均从 500 变 400,而有效 JPEG/PNG/RGBA、malformed base64、随机非图片字节等原本就正确的用例保持不变。

## 根因

正文明确披露:PIL 在图片解码时抛出 `OSError`(如 "Truncated File Read"),该异常没有被捕获,一路穿透到默认错误处理路径,被归为服务器内部错误返回 500。而按 API 语义,4xx 应表示客户端错误(输入非法)、5xx 表示服务器错误,这类坏输入被错误分类了。

## 修复方案

改 `vllm/multimodal/media/image.py` 的 `ImageMediaIO`:

1. `load_bytes`:把 `Image.open`、`image.load()`(补上完整解码)、`_convert_image_mode` 包进 try,捕获 `(OSError, Image.UnidentifiedImageError)` 后统一抛 `ValueError(f"Failed to load image: {e}") from e`,让既有错误处理将其映射为 HTTP 400:

```diff
+        try:
+            image = Image.open(BytesIO(data))
+            image.load()
+            image = self._convert_image_mode(image)
+        except (OSError, Image.UnidentifiedImageError) as e:
+            raise ValueError(f"Failed to load image: {e}") from e
+        return MediaWithBytes(image, data)
```

2. `load_file`:原来单独读文件再 `Image.open`,现直接复用 `load_bytes(filepath.read_bytes())`,消除重复且继承同样的异常处理。

另在 `tests/multimodal/media/test_image.py` 新增 `test_image_media_io_load_bytes` / `test_image_media_io_load_file`,覆盖合法图、乱码字节、PNG 头截断、真实文件半截、空字节等用例,断言均抛 "Failed to load image" 的 ValueError。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:按 diff,`load_bytes` 现在会立即 `image.load()` 完整解码(此前惰性加载,坏数据可能延后才失败),合法图片行为不变;`load_file` 逻辑等价合并。所有走 `ImageMediaIO` 的多模态图片输入路径(含 base64 与文件)受影响:损坏输入从 500 改为 400,若上游有依赖 500 语义的监控/重试逻辑需相应调整。`load_base64` 经 `b64decode(validate=True)` 抛的 base64 解析错误不在本次捕获范围,行为不变。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
