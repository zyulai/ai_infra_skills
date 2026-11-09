# PR #33788: Fix inference mode mismatch in FlashInfer warmup

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33788](https://github.com/sgl-project/sglang/pull/33788) |
| 作者 | nvpohanh |
| 合并日期 | 2026-08-06 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
FlashInfer warmup 用 inference_mode 创建了 workspace 推理张量,后续被正式服务原地更新报错;移除该上下文修复。

## 问题现象
正文以 Summary 形式描述:FlashInfer autotune 与共享的 dummy warmup forward 处于 `torch.inference_mode()` 中,为惰性分配(lazily allocated)的 workspace 创建了 inference tensor;随后 eager serving 与 CUDA graph capture 阶段会对这些 workspace 做原地(in-place)更新,与 inference tensor 的约束冲突,出现 inference mode mismatch 错误(对应 issue #33470)。正文未附具体报错日志。

## 根因
正文披露:warmup 路径(`base_runner.py` 的共享 dummy warmup forward 与 `flashinfer_autotune.py` 的 autotune context)在 `torch.inference_mode()` 下运行,而惰性分配的 workspace 张量此时被创建为 inference tensor;这些张量在之后的 eager 推理与 CUDA graph capture 中被原地写入,PyTorch 禁止在 inference mode 之外原地更新 inference tensor,从而报错。修复思路是不引入 `torch.no_grad()` 之类的替代上下文,直接移除 `inference_mode`。

## 修复方案
1. `python/sglang/srt/model_executor/runner/base_runner.py`:共享 dummy warmup `run_once()` 的上下文由 `with torch.inference_mode(), run_ctx or empty_context():` 改为 `with run_ctx or empty_context():`,即移除 `torch.inference_mode()` 且不添加显式 `torch.no_grad()`。
2. `python/sglang/srt/model_executor/runner/flashinfer_autotune.py`:`flashinfer_autotune_context()` 中 `with torch.inference_mode(), autotune(...)` 改为 `with autotune(...)`,同样仅移除 `inference_mode`。

```diff
-            with torch.inference_mode(), run_ctx or empty_context():
+            with run_ctx or empty_context():
                 run_once()
```

## 影响与回归风险
- 影响窗口:引入 `inference_mode` 的 warmup 重构之后的版本(正文未给出具体提交/版本号,不写具体范围)。
- 回归面:模型启动 warmup 与 FlashInfer autotune 阶段不再处于 inference mode,梯度追踪状态可能变化(autotune 的 dummy 前向理论上不反传,主要影响内存/耗时极小);所有依赖 warmup 期创建的 workspace 的后续路径(eager serving、CUDA graph capture)行为恢复为普通张量;非 FlashInfer 后端若共用 `base_runner` 的 warmup 也会一并脱离 inference mode。

## 关联
- Closes #33470:https://github.com/sgl-project/sglang/issues/33470
- 其余 issue/PR:正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
