# PR #13727: Revert "Fix RMSNorm API CALL mismatch issue. (#10032)"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#13727](https://github.com/sgl-project/sglang/pull/13727) |
| 作者 | ErsongWang |
| 合并日期 | 2025-11-22 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论

#10032 按 vLLM 0.9 的新 6 参签名调 `fused_add_rms_norm`,但该签名只存在于 ROCm/vllm 355_wip 开发分支,revert 回 4 参调用。

## 问题现象

HIP/ROCm 环境下,使用 rocm.Dockerfile 构建的 vllm 调用 `fused_add_rms_norm` 会失败:rocm.Dockerfile 使用的 vllm 仍是 4 参数版本的 `_custom_ops.py` 实现(正文引用了 [ROCm/vllm 9f6b92d 的 _custom_ops.py](https://github.com/ROCm/vllm/blob/9f6b92db47c3444b7a7d67451ba0c3a2d6af4c2c/vllm/_custom_ops.py)),而 #10032 引入的代码在检测到 vllm 版本 >= 0.9 时改用新 6 参签名调用。触发条件:HIP 环境 + vllm 版本 >= 0.9 且该 vllm 来自不含新签名的分支。影响面为 ROCm 上 RMSNorm + residual 的前向路径。

## 根因

正文披露:`vllm::fused_add_rms_norm()` 的新参数签名只在 ROCm/vllm 的 355_wip 开发分支中引入,vllm-project/vllm 与 ROCm/vllm 的 main 分支都不存在;而 #10032 的版本判断逻辑假设 vllm >= 0.9 就具备新签名,对已发布分支的 vllm 做出错误调用。因此整体 revert #10032,恢复 4 参调用。

## 修复方案

revert commit 8d114f254b0ebff665b31bb814cc30cf8aee1c66,改 `python/sglang/srt/layers/layernorm.py`:

- 删除 `import vllm` 与 `_vllm_version = Version(vllm.__version__)` 的版本分支判断;
- `forward_hip` 中 residual 非空时恢复为统一的 4 参调用,不再按版本走 6 参路径:

```diff
-            if _vllm_version < Version("0.9"):
-                fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
-                return x, residual
-            else:
-                residual_out = torch.empty_like(x)
-                output = torch.empty_like(x)
-                fused_add_rms_norm(
-                    output,
-                    x,
-                    residual_out,
-                    residual,
-                    self.weight.data,
-                    self.variance_epsilon,
-                )
-                return output, residual_out
+            fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
+            return x, residual
```

## 影响与回归风险

- 影响窗口:自 #10032 合并引入版本分支调用起,至本 revert。
- 回归面:仅 HIP/ROCm 下 `layernorm.py` 的 `forward_hip` RMSNorm(+residual)路径。恢复 4 参调用后,在 355_wip 等真正具备新签名(新实现)的 vllm 构建上,行为回到旧签名约定;若后续要支持新签名,需等新签名进入 main 分支再重新引入。

## 关联

- #10032(被 revert 的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
