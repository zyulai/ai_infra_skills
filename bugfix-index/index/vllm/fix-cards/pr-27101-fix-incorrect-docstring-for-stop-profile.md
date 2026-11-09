# PR #27101: Fix incorrect docstring for stop_profile() method

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#27101](https://github.com/vllm-project/vllm/pull/27101) |
| 作者 | hyongtao-code |
| 合并日期 | 2025-10-17 |
| labels | (none) |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
`vllm/engine/protocol.py` 中 `stop_profile()` 的 docstring 误写成 "Start profiling the engine",与语义相反,更正为 "Stop profiling the engine"。

## 问题现象
正文未描述具体现象。问题在于引擎协议 `vllm/engine/protocol.py` 里抽象方法 `stop_profile()` 的 docstring 写的是 "Start profiling the engine",与该方法「停止 profiling」的语义正好相反;该协议是各引擎实现(LLMEngine 等抽象接口)的公共定义,阅读源码或据此生成的 API 文档时,读者会得到与实际行为相反的说明,造成误导。仅文档字符串问题,不影响任何运行时行为。

## 根因
正文未披露根因。(由 diff 推断)`stop_profile()` 紧跟在同文件的 `start_profile()` 抽象方法之后,其 docstring 疑似从 `start_profile()` 复制后未加修改,导致文案停留在 "Start profiling the engine"。

## 修复方案
改动仅 `vllm/engine/protocol.py` 中 `stop_profile()` 的一行 docstring:

```diff
     @abstractmethod
     async def stop_profile(self) -> None:
-        """Start profiling the engine"""
+        """Stop profiling the engine"""
```

## 影响与回归风险
- 影响窗口:正文未提及具体版本/提交范围,不写。
- 回归面:纯注释(docstring)改动,不改变任何函数签名或逻辑,运行时回归风险为零;唯一影响是 API 文档/IDE 提示中该方法的描述文案。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
