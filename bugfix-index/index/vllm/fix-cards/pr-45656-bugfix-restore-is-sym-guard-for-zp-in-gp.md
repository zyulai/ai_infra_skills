# PR #45656: [Bugfix] Restore is_sym guard for zp in GPTQ/CT MoE to fix symmetric quant regression

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45656](https://github.com/vllm-project/vllm/pull/45656) |
| 作者 | yuwenzho |
| 合并日期 | 2026-06-18 |
| labels | bug, ready, verified |
| 热度 | 👍 0 · 💬 0 |
| 分类 | MoE |
| bug 类型 | regression |

## 一句话结论
#43409 移除对称量化的 zp 守卫,使 GPU 对称 GPTQ MoE 向 Marlin kernel 传无效 qzeros 得到错误结果;恢复守卫并加 CPU 例外。

## 问题现象
在 NVIDIA GPU 上,对称 GPTQ MoE 量化(如 Autoround,以及 `is_sym=True` 的标准 GPTQ)输出错误结果。触发场景:对称量化模型在 GPU 后端运行时,无意义的 `qzeros` 张量被传入 Marlin kernel。

## 根因
正文披露:回归由 #43409 引入(见其 issuecomment-4700406550)。#43409 为让 CPU 能为对称模型传入合成的 zero points,移除了 `AutoGPTQMoEMethod.get_fused_moe_quant_config` 中 `if not self.quant_config.is_sym else None` 的守卫;副作用是对称 GPU 量化开始把无意义的 `qzeros` 张量传给 Marlin kernel,产生错误结果。同一模式也存在于 `CompressedTensorsWNA16MarlinMoEMethod.process_weights_after_loading`:缺少 `if not self.symmetric:` 守卫,非 CPU 对称后端存在错误注册 zp 的风险。

## 修复方案
按正文所述在两个文件恢复 `is_sym` 守卫并加显式 CPU 例外(CPU 的 `fused_experts_cpu` 即使对称量化也需要 zero points):

1. `vllm/model_executor/layers/quantization/auto_gptq.py` 的 `get_fused_moe_quant_config`:

```diff
+        # CPU fused_experts_cpu requires zero points even for symmetric quant
+        use_zp = (
+            not self.quant_config.is_sym
+            or self.wna16_moe_backend == WNA16MoEBackend.CPU
+        )
         return gptq_marlin_moe_quant_config(
             ...
-            w1_zp=getattr(layer, "w13_qzeros", None),
-            w2_zp=getattr(layer, "w2_qzeros", None),
+            w1_zp=getattr(layer, "w13_qzeros", None) if use_zp else None,
+            w2_zp=getattr(layer, "w2_qzeros", None) if use_zp else None,
```

2. `.../compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py` 的 `process_weights_after_loading`:`if w13_qzeros is not None:` 改为 `if not self.symmetric or self.wna16_backend == WNA16MoEBackend.CPU:`,并把 w2 零点注册并入同一分支。

测试:CPU 上 GPTQ W4A16 INT4 MoE(Qwen/Qwen3-30B-A3B-GPTQ-Int4)与 compressed-tensor W4A16 INT4 MoE(RedHatAI/Qwen3-30B-A3B-quantized.w4a16)均通过。

## 影响与回归风险
- 影响窗口:正文未给出具体版本号;回归自 #43409 移除守卫起,至本修复合并前。
- 回归面:GPTQ Marlin MoE(`auto_gptq.py`)与 compressed-tensors WNA16 Marlin MoE 的量化配置构造/权重后处理路径;CPU 对称量化路径依赖 `WNA16MoEBackend.CPU` 判定仍能拿到合成的 zero points,若该判定有遗漏,CPU 对称 MoE 可能重新受影响(由 diff 推断)。

## 关联
- #43409(正文指明的回归引入 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
