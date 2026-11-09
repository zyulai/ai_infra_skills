# PR #10032: Fix RMSNorm API CALL mismatch issue.

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10032](https://github.com/sgl-project/sglang/pull/10032) |
| 作者 | sogalin |
| 合并日期 | 2025-09-06 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
vllm 升级后 `fused_add_rms_norm` 新增两个参数,HIP 路径按 vllm 版本分支传参修复。

## 问题现象
随着 vllm 更新,SGLang 中出现 API 不匹配("With updating of the vllm, we see API mismatch in SGLang")。正文未描述具体报错信息;按改动位置推断(由 diff 推断),问题出现在 HIP(ROCm)环境下、residual 非空时调用 `fused_add_rms_norm` 的 `forward_hip` 路径,调用参数与新版本 vllm 的函数签名对不上。

## 根因
正文披露:vllm 更新后其 `vllm::fused_add_rms_norm()` 引入了两个新参数("Two new arguments were introduced into the vllm::fused_add_rms_norm()"),SGLang 中的调用方式需要相应修订。

## 修复方案
改动在 `python/sglang/srt/layers/layernorm.py` 的 `forward_hip`:

1. 引入 `packaging.version.Version`,在 `_is_hip` 分支 import vllm 并记录 `_vllm_version = Version(vllm.__version__)`。
2. residual 非空时按 vllm 版本分支调用:vllm < 0.9 保持旧的 4 参数调用(x、residual、weight、variance_epsilon 原地更新);vllm >= 0.9 改为新签名,显式分配输出张量:

```diff
-            fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
-            return x, residual
+            if _vllm_version < Version("0.9"):
+                fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
+                return x, residual
+            else:
+                residual_out = torch.empty_like(x)
+                output = torch.empty_like(x)
+                fused_add_rms_norm(
+                    output,
+                    x,
+                    residual_out,
+                    residual,
+                    self.weight.data,
+                    self.variance_epsilon,
+                )
+                return output, residual_out
```

aiter 分支(`SGLANG_USE_AITER` / `rmsnorm2d_fwd_with_add`)不受影响。

## 影响与回归风险
- 影响窗口:vllm >= 0.9 的 HIP 环境(旧 vllm < 0.9 走原路径,行为不变)。
- 回归面:仅影响 HIP 下 RMSNorm forward 的 residual 融合路径;新路径多分配两个 `empty_like` 输出张量,可能带来轻微显存/性能开销。作者用 Qwen3-30B-A3B-Thinking-2507、tp 8、aiter attention backend 跑了 gsm8k 2000 题评测:Accuracy 0.849、Invalid 0.000、Latency 54.786s、Output throughput 5336.137 token/s。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
