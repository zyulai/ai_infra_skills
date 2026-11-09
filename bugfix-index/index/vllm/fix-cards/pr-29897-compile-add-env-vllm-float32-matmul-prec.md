# PR #29897: [Compile] Add env `VLLM_FLOAT32_MATMUL_PRECISION` to fix torch warning `TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled`

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#29897](https://github.com/vllm-project/vllm/pull/29897) |
| 作者 | yewentao256 |
| 合并日期 | 2025-12-09 |
| labels | ready, v1 |
| 热度 | 👍 1 · 💬 2 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论

vLLM 未显式设置 float32 matmul 精度,torch/inductor 反复弹出 TF32 未启用警告;新增环境变量默认锁定 `highest`。

## 问题现象

在 torch inductor 编译阶段反复出现 UserWarning(正文照录):

`/home/wentao/pytorch/torch/_inductor/compile_fx.py:321: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting torch.set_float32_matmul_precision('high') for better performance.`

即每次触发编译都会刷屏警告,提示 TF32 tensor core 可用但未启用。触发条件是使用 torch inductor 编译路径(compile_fx);正文未描述其他功能性症状,警告本身不影响结果正确性,只影响日志整洁并可掩盖真正的问题。

## 根因

正文未长篇披露根因,仅说明修复方式是 "Now we explicitly set `torch.set_float32_matmul_precision('highest')` by default"。结合警告文本可推断:vLLM 此前从未显式调用 `torch.set_float32_matmul_precision`,float32 矩阵乘精度处于 torch 默认状态,inductor 在编译时检测到硬件支持 TF32 但未启用,便按其内部逻辑发出该 UserWarning(由 diff 推断)。

## 修复方案

1. `vllm/envs.py`:新增环境变量 `VLLM_FLOAT32_MATMUL_PRECISION`,取值为 `["highest", "high", "medium"]`(镜像 `torch.set_float32_matmul_precision` 的选项,大小写不敏感),默认 `highest`。
2. `vllm/v1/worker/gpu_worker.py`:worker 初始化时读取该环境变量并调用 `torch.set_float32_matmul_precision(precision)`,即默认显式设为 `highest`(全 FP32 精度),使警告条件不再出现:

```diff
+        # configure float32 matmul precision according to vLLM env.
+        precision = envs.VLLM_FLOAT32_MATMUL_PRECISION
+        torch.set_float32_matmul_precision(precision)
```

3. `tests/v1/e2e/test_async_scheduling.py`:在 `VLLM_ATTENTION_BACKEND=FLEX_ATTENTION` 的用例中同时设置 `VLLM_FLOAT32_MATMUL_PRECISION=highest`,注释为 "lock matmul precision to full FP32",避免该测试出现精度误差。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:按 diff,所有 v1 GPU worker 初始化时都会执行 `set_float32_matmul_precision('highest')`,作用于该 worker 内全部 float32 matmul——数值上与 torch 默认的 `highest` 一致,不改变默认计算结果;若用户此前依赖在 worker 进程内自行开启 TF32(`high`)提速,worker 初始化会将其覆盖为 `highest`,需改用新环境变量。环境变量解析走 `env_with_choices`,传入非法值会报错;`medium` 选项仅透传给 torch,实际语义由 torch 版本决定。仅改 v1 worker,未触及 CPU/TPU 等其他 worker 路径。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
