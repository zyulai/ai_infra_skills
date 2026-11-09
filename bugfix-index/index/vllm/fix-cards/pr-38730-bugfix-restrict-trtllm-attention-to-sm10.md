# PR #38730: [Bugfix] Restrict TRTLLM attention to SM100, fixing GB300 (SM103) hang

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38730](https://github.com/vllm-project/vllm/pull/38730) |
| 作者 | stecasta |
| 合并日期 | 2026-04-01 |
| labels | bug, documentation, nvidia, ready |
| 热度 | 👍 1 · 💬 5 |
| 分类 | Attention / Kernel |
| bug 类型 | hang |

## 一句话结论

FlashInfer 0.6.7 的 TRTLLM attention 不兼容 SM103,GB300 大 batch 推理挂死;把支持判定收紧到精确 SM100 修复。

## 问题现象

GB300(SM103,CC 10.3)上**所有模型**在较大 batch size 推理时无限期挂起:GPU 显示 99% SM 利用率、0% 显存带宽。这是 FlashInfer 从 0.6.6 升级到 0.6.7 引入的回归——TRTLLM attention kernel 不再向前兼容 SM103。GB200(SM100)不受影响。测试计划验证:Qwen3-8B-FP8 与 Qwen2.5-7B BF16 吞吐基准在 GB300 上由挂死变为通过,且 GB200/B200 仍使用 TRTLLM attention 无回归。

## 根因

PR 正文披露:根因是 FlashInfer 0.6.6→0.6.7 升级后 TRTLLM attention kernel 与 SM103 不向前兼容(fix 在 FlashInfer 侧进行中)。vLLM 侧的触发机制:`vllm/utils/flashinfer.py` 的 `supports_trtllm_attention()` 用 `is_device_capability_family(100)` 判定,覆盖整个 CC 10.x 家族,导致 SM103 的 GB300 也被选中走 TRTLLM attention backend,从而挂死。

## 修复方案

改 `vllm/utils/flashinfer.py`:`supports_trtllm_attention()` 从 `is_device_capability_family(100)`(10.x 家族)收紧为 `is_device_capability(100)`(精确 CC 10.0);SM103 自动回退到 FlashInfer 默认 attention backend(工作正常):

```diff
-    # Requires SM100 and NVIDIA artifactory to be accessible to download cubins
-    return (
-        current_platform.is_device_capability_family(100) and has_nvidia_artifactory()
-    )
+    # TRTLLM attention is currently only validated on SM100 (CC 10.0).
+    # SM103 (GB300) hangs with FlashInfer >= 0.6.7.
+    return current_platform.is_device_capability(100) and has_nvidia_artifactory()
```

同步更新 `docs/design/attention_backends.md` 表格(TRTLLM CC 列 10.x → 10.0)与 `tools/pre_commit/generate_attention_backend_docs.py` 的文档生成器(支持 `is_device_capability(N)` → "M.m" 的解析)。

## 影响与回归风险

- 影响窗口:回归由 FlashInfer 0.6.7 升级引入(正文口径);本修复影响所有 CC 10.x 中非精确 10.0 的 GPU(如 SM103)的后端选择。
- 回归面:TRTLLM attention 的启用判定与相关文档;SM100 设备不受影响仍走 TRTLLM,SM103 等 family 内其他型号回退默认 backend(吞吐/性能特征可能与之前不同,但正确性恢复)。

## 关联

- https://github.com/flashinfer-ai/flashinfer/issues/2939
- Fixes https://github.com/vllm-project/vllm/issues/38729

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
