# PR #63433: [Bugfix][Multimodal] Return 400 for corrupt AVIF and BLP images

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#63433](https://github.com/vllm-project/vllm/pull/63433) |
| 作者 | vadiklyutiy |
| 合并日期 | 2026-10-27 |
| labels | bug, ready, multi-modality |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
损坏 AVIF/BLP 图片请求误返回 500/501:`ImageMediaIO.load_bytes` 的异常捕获表缺 `RuntimeError`,补上后统一归一为 400。

## 问题现象
- 带损坏 AVIF 图片的 chat 请求返回 HTTP 500(`RuntimeError: Failed to decode image: Missing or empty image item`),带坏 BLP 数据的请求返回 HTTP 501(`BLPFormatError: Unknown BLP compression 0`),而正确行为应是客户端错误 400。影响面:所有传入无法解码的 AVIF/BLP(及触发同族异常)图片的请求;有效 PNG/JPEG/WEBP/AVIF 与 vLLM 自身的 pixel limit、decoded-size budget `ValueError` 不受影响。验证环境:CPU only、upstream `main` `debcf2284`、Pillow 13.0.0、Python 3.12;HTTP 状态验证通过将输入交给 `ImageMediaIO.load_bytes` 并用真实 `create_error_response` 完成,见正文对照表(垃圾字节/截断 PNG 前后均 400,DecompressionBombError 场景由 #60308 修)。

## 根因
正文披露:`ImageMediaIO.load_bytes` 只把 `OSError`、`UnidentifiedImageError`、`SyntaxError` 三类异常包装成 `ValueError("Failed to load image: ...")`(该 ValueError 上层转为 400)。但 Pillow 的 AVIF 解码器(`_avif.AvifDecoder`,在 `Image.open()` 中)抛 `RuntimeError`,BLP 插件抛 `BLPFormatError`(`NotImplementedError` 子类),两者都绕过了该 `except`,于是以未处理异常形态变成 500/501。正文 Details 还说明:Pillow 没有统一的「坏图像数据」基类,`RuntimeError` 同时覆盖 `NotImplementedError`(插件不支持特性,如 BLPFormatError)与 `RecursionError`(深层嵌套输入),这些都源自客户端输入,属 400;`MemoryError` 仍不捕获、仍 500;超像素上限的 `DecompressionBombError` 同因但由 #60308 处理,本 PR 不动。

## 修复方案
`vllm/multimodal/media/image.py::load_bytes` 的 except 元组加入 `RuntimeError`(≤1 行实质改动):
```python
except (
    OSError,
    Image.UnidentifiedImageError,
    SyntaxError,
    RuntimeError,
) as e:
    raise ValueError(f"Failed to load image: {e}") from e
```
并在 `tests/multimodal/media/test_image.py` 新增参数化测试 `test_image_decoder_error_rejected`(AVIF 无 primary item 的 base64 样本 + compression 0 的 BLP2 头),断言抛出 `Failed to load image` ValueError 且 `__cause__` 类型正确;补丁后 22 passed,旧代码下这两个新用例分别以 RuntimeError/BLPFormatError 失败。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:`ImageMediaIO.load_bytes` 对所有图片格式的异常映射:原本逃逸为 500/501 的 `RuntimeError`/`NotImplementedError`/`RecursionError` 解码失败现在统一为 400 `Failed to load image`;有效图片加载路径与 vLLM 自身的 pixel limit / decoded-size budget `ValueError` 不变;`MemoryError` 仍为 500。正文与 #62704、#60308 修改的是同一个 except 元组,后合并者需一行 rebase。

## 关联
- [#62703](https://github.com/vllm-project/vllm/issues/62703)(灵感来源场景)
- [#62704](https://github.com/vllm-project/vllm/pull/62704)(同元组的上一处修复)
- [#60308](https://github.com/vllm-project/vllm/pull/60308)(DecompressionBombError 场景修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
