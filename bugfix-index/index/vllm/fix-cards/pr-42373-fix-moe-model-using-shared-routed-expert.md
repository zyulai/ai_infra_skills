# PR #42373: fix: MoE model using shared routed experts crashes on AMD GPUs

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42373](https://github.com/vllm-project/vllm/pull/42373) |
| 作者 | weizhoublue |
| 合并日期 | 2026-05-25 |
| labels | rocm, ready, deepseek |
| 热度 | 👍 1 · 💬 3 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
目录重组后两处懒加载仍引用旧模块路径,ROCm+AITER 的共享路由专家 MoE 启动即 ModuleNotFoundError,改 import 路径修复。

## 问题现象
在 ROCm 平台且启用 AITER fused MoE 时,加载带共享路由专家(shared routed expert)路径的 MoE 模型(如 DeepSeek-V2/V3、Mixtral)会在模型加载阶段崩溃:抛出 `ModuleNotFoundError`,因为 `import vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe` 指向的模块已不存在,模型初始化直接失败且无 workaround。正文概括为「stale import crashes every ROCm+AITER MoE model at startup」。

## 根因
正文明确披露:PR #41979 重组了 `fused_moe/` 目录,把专家实现移入 `experts/` 子目录,并将 `rocm_aiter_fused_moe.py` 重命名为 `experts/rocm_aiter_moe.py`,但漏改了 `router/aiter_shared_routed_fused_moe_router.py` 中的两处懒加载 import,它们仍引用旧路径 `vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe`,模块已不存在,触发导入失败。

## 修复方案
只改 `vllm/model_executor/layers/fused_moe/router/aiter_shared_routed_fused_moe_router.py` 中 `_compute_routing` 里的两处懒加载 import,把模块路径从旧名 `...fused_moe.rocm_aiter_fused_moe` 改为新名 `...fused_moe.experts.rocm_aiter_moe`;导入的符号(`aiter_topK_meta_data`、`inject_shared_expert_weights`)与使用逻辑不变。

关键 diff 摘录:
```diff
-        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
+        from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
             aiter_topK_meta_data,
         )
...
-            from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
+            from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
                 inject_shared_expert_weights,
             )
```

## 影响与回归风险
- 影响窗口:自引入 #41979 目录重组(该 PR 合并后)至本修复之间的版本,ROCm+AITER 且启用共享路由专家路径的 MoE 模型启动即崩。
- 回归面:纯 import 路径修正,仅影响 `aiter_shared_routed_fused_moe_router.py` 的两条懒加载分支;NVIDIA(CUDA)路径与其他 MoE 实现不受影响。若 `experts/rocm_aiter_moe.py` 中符号名再有变动才会再次出错,风险很低。

## 关联
[#41979](https://github.com/vllm-project/vllm/pull/41979)(正文指出的引入回归的重组 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
