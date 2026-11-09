# PR #62704: [Bugfix][Multimodal] Return 400 for structurally-corrupted PNG images

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#62704](https://github.com/vllm-project/vllm/pull/62704) |
| 作者 | reda-maizate |
| 合并日期 | 2026-10-22 |
| labels | bug, ready, multi-modality |
| 热度 | 👍 0 · 💬 11 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
结构损坏的 PNG 逃过异常捕获致 500;把 `SyntaxError` 加进 except 元组,统一转 400。

## 问题现象
请求中携带结构损坏(损坏的 chunk 结构)的 PNG 时,聊天补全接口返回 **HTTP 500 `InternalServerError`**(带原始 Pillow 报错信息),例如 `500 InternalServerError: broken PNG file (chunk b'\x00\x00IE')`,而正确的客户端错误应是 HTTP 400。影响面:所有向多模态接口提交损坏 PNG 的请求;有效图片与已被 `OSError` / `UnidentifiedImageError` 分支处理的非法输入行为不变(仍返回 400)。作者用真实世界的 79 字节损坏 PNG(IDAT CRC 与 IEND 之间的杂散字节使 chunk 流错位)作复现,并在真实部署上做了 A/B 验证:`vllm/vllm-openai:nightly`(`df3bd094a`),单卡 H100,`google/gemma-4-26b-A4B-it`,修复前 500、修复后 400,有效 64×64 PNG 两者均 200,随机非图片字节两者均 400。

## 根因
PR 正文披露:`ImageMediaIO.load_bytes` 会把解码失败包装成面向客户端的 `ValueError`(`"Failed to load image: ..."`),但其 except 子句只捕获 `(OSError, Image.UnidentifiedImageError)`。Pillow 对损坏的 chunk 结构抛出的是普通 `SyntaxError`,两者都不是,于是异常未经包装直接逃逸。请求级错误映射器(`vllm/entrypoints/serve/exception_handling/error_response.py`)把 `ValueError` 提升为 `BadRequestError`(400),而未映射的异常映射为 `InternalServerError`(500)——这正是客户端观察到的 500。

## 修复方案
在 `vllm/multimodal/media/image.py` 的 `load_bytes` 中,把 `SyntaxError` 加入现有 except 元组,一行扩展既有错误包装模式:

```python
# Pillow raises SyntaxError (not OSError) for broken PNG chunks.
except (OSError, Image.UnidentifiedImageError, SyntaxError) as e:
    raise ValueError(f"Failed to load image: {e}") from e
```

正文论证:该 try 块内只调用 Pillow 与普通属性读取,不可能出现真正的 Python `SyntaxError`,因此放宽捕获不会掩盖无关失败。同时在 `tests/multimodal/media/test_image.py` 新增 `test_image_broken_png_rejected`,使用真实复现字节断言抛出 `ValueError` 且 `__cause__` 为 `SyntaxError`(20 passed)。

## 影响与回归风险
- 影响窗口:正文未写明版本范围;验证基线为 `vllm/vllm-openai:nightly`(`df3bd094a`)。
- 回归面:仅改动 `vllm/multimodal/media/image.py` 的异常捕获元组与新增一个测试文件用例。对有效图片零行为变化;对已被 `OSError` / `UnidentifiedImageError` 覆盖的非法输入保持原 400 路径;唯一行为变化是损坏 PNG 从 500 变 400。与 #60308 的 `DecompressionBombError` 改动可平凡组合。

## 关联
- Fixes #62703
- Related: #60303
- Related: #60308

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
