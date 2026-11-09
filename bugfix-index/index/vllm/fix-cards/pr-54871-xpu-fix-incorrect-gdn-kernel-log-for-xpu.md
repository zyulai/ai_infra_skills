# PR #54871: [XPU] fix incorrect gdn kernel log for XPU path

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54871](https://github.com/vllm-project/vllm/pull/54871) |
| 作者 | yma11 |
| 合并日期 | 2026-09-20 |
| labels | intel-gpu |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论

XPU 平台缺少独立分支,GDN decode kernel 日志输出错误内容;补上 XPU 分支并标记 `gdn_decode_kernel = "XPU"` 修复。

## 问题现象

XPU 平台上 GDN 内核选择日志内容不正确。正文仅以修复前后日志截图(before/after 链接)展示差异,未用文字描述具体错误内容;Test Plan 与 Test Result 小节为空,正文自述为「minor code change」。

## 根因

正文未披露根因分析。由 diff 可见两处暗示的机制(由 diff 推断):`_log_gdn_backend_decision` 没有 XPU 平台分支,会继续执行面向 CUDA 的 flashinfer/triton 选择字典逻辑,打印错误决策;`__init__` 的 kernel 选择链只处理 cuda/cpu 分支,XPU 平台未给 `self.gdn_decode_kernel` 赋值,导致 `logger.info_once("GDN decode kernel: %s", ...)` 输出错误值。

## 修复方案

修改 `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` 两处:`_log_gdn_backend_decision` 中在进入 CUDA 选择逻辑前加 `elif current_platform.is_xpu(): return`;`__init__` 选择链加 `elif current_platform.is_xpu(): self.gdn_decode_kernel = "XPU"`。

```diff
         elif current_platform.is_cpu():
             self.gdn_decode_kernel = "CPU"
+        elif current_platform.is_xpu():
+            self.gdn_decode_kernel = "XPU"
 
         self.enable_fused_gdn_decode = self.gdn_decode_kernel == "cuda"
         logger.info_once("GDN decode kernel: %s", self.gdn_decode_kernel)
```

## 影响与回归风险

- 影响窗口:正文未说明。
- 回归面:仅 `qwen_gdn_linear_attn.py` 的日志函数与 XPU 平台的 `gdn_decode_kernel` 标记;`enable_fused_gdn_decode` 仍按 `== "cuda"` 判定,XPU 分支不改变 CUDA/CPU 行为(由 diff 推断)。正文定性为修正 XPU 平台 GDN 内核日志的 minor 改动。

## 关联

正文未列出(仅附 before/after 截图链接)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
