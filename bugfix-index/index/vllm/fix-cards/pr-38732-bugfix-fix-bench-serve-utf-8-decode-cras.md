# PR #38732: [Bugfix] Fix bench_serve UTF-8 decode crash on split multi-byte chars

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38732](https://github.com/vllm-project/vllm/pull/38732) |
| 作者 | he-yufeng |
| 合并日期 | 2026-04-16 |
| labels | bug, performance, ready |
| 热度 | 👍 3 · 💬 2 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
bench_serve 流式响应逐块 decode 在多字节字符跨 chunk 时崩溃;改用 codecs.IncrementalDecoder 缓冲续解。

## 问题现象
`StreamedResponseHandler.add_chunk()` 直接 `chunk_bytes.decode("utf-8")`,当多字节 UTF-8 字符(如中文)被 HTTP chunk 边界分割时崩溃:
```
UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 65527-65528: unexpected end of data
```
报告者在 H100x8 的 bench_serve 中约 0.4% 的请求(4/1000)命中。

## 根因
正文披露:一次性 `decode("utf-8")` 无法处理被 chunk 边界截断的不完整多字节序列。

## 修复方案
`vllm/benchmarks/lib/endpoint_request_func.py`:把一次性 decode 换成 `codecs.IncrementalDecoder`,跨 `add_chunk()` 调用缓冲不完整字节序列,等下一个 chunk 补全字符后再解码——这是 Python 流式字节转字符串的标准做法:
```python
def __init__(self):
    self.buffer = ""
    self._decoder = codecs.getincrementaldecoder("utf-8")()

def add_chunk(self, chunk_bytes: bytes) -> list[str]:
    """Add a chunk of bytes to the buffer and return any complete
    messages."""
    chunk_str = self._decoder.decode(chunk_bytes)
```
作者验证:崩溃 traceback 与 `endpoint_request_func.py:32` 匹配;并用人为拆分的中文字节本地验证 `IncrementalDecoder` 能正确处理跨块多字节序列;`ruff check` 与 `ruff format --check` 通过。

## 影响与回归风险
- 回归面:仅 benchmark 工具的流式响应处理(bench_serve 等使用 `endpoint_request_func.py` 的路径),不影响服务端推理;解码语义对完整合法 UTF-8 流不变。

## 关联
Fixes #38717

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
