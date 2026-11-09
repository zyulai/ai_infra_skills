# PR #47253: [XPU] Fix PP accuracy on XPU device

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47253](https://github.com/vllm-project/vllm/pull/47253) |
| 作者 | yisustc |
| 合并日期 | 2026-07-07 |
| labels | intel-gpu, ready, v1 |
| 热度 | 👍 2 · 💬 4 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

XPU 上流水线并行 PP 广播采样结果时,广播流未等主流写完 int64 结果就发送,产生竞态导致输出不准;在广播前对主流做 synchronize 修复。

## 问题现象

正文仅写「Fix the accuracy issue on XPU device」,未描述具体现象;可确定的是 XPU 设备上启用流水线并行(PP)时出现精度/正确性问题,修复针对 `vllm/v1/worker/gpu/pp_utils.py` 中采样结果广播的代码路径。

## 根因

正文未披露根因。(由 diff 推断)`broadcast` 函数把 `sampled_token_ids`(int64)放到独立的 `broadcast_stream` 上做 `torch.distributed.broadcast`,流程是 `broadcast_stream.wait_stream(self.main_stream)` 后即广播。在 XPU 平台上,`wait_stream` 的语义可能不足以保证主流已完成写入 `sampled_token_ids`,广播读到未写完/过期的数据,PP 各 rank 收到错误的采样结果。修复在广播前显式调用 `self.main_stream.synchronize()`(仅 XPU),强制等待主流写完后再广播。

## 修复方案

修改 `vllm/v1/worker/gpu/pp_utils.py`:导入 `current_platform`,在 `broadcast` 中断言 `sampled_token_ids.dtype == torch.int64` 之后、进入 `torch.cuda.stream(self.broadcast_stream)` 之前,针对 XPU 插入主流同步。关键 diff:

```diff
         assert sampled_token_ids.dtype == torch.int64
+
+        if current_platform.is_xpu():
+            self.main_stream.synchronize()
+
         with torch.cuda.stream(self.broadcast_stream):
             self.broadcast_stream.wait_stream(self.main_stream)
             torch.distributed.broadcast(
```

仅 XPU 平台走新同步,其他平台行为不变。

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:改动只在 XPU 的 PP 广播路径插入一次主流同步,属于用性能换正确性(同步会带来一定延迟);NVIDIA/AMD 等其他平台代码路径完全不变。若 XPU 驱动后续修复 wait_stream 语义,该同步可移除。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
