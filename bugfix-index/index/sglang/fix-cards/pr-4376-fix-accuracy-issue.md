# PR #4376: fix accuracy issue

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#4376](https://github.com/sgl-project/sglang/pull/4376) |
| 作者 | zhyncs |
| 合并日期 | 2025-03-13 |
| labels | (none) |
| 热度 | 👍 4 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
sgl-kernel FP8 per-token 量化内核精度相关修复:除法改乘倒数、部分架构启用 `-use_fast_math`,并调整 scale 形状与测试容差。

## 问题现象
正文未描述具体现象:Motivation 只贴出 `sgl-kernel/tests` 中 FP8 量化相关测试(`test_per_tensor_quant_fp8.py` 9 passed、`test_per_token_group_quant_fp8.py` 240 passed、`test_per_token_quant_fp8.py` 9 passed)的通过日志,`Modifications` 一节为空。

## 根因
正文未披露根因,也未说明「accuracy issue」的具体内容。从 diff 描述改动所暗示的机制(由 diff 推断):`per_token_quant_fp8_kernel` 把逐元素除以 `block_max` 改为先算倒数 `scale_val = 1.0f / block_max` 再做乘法,并在 setup.py 中对非 sm100/sm90 分支追加 `-use_fast_math` 编译选项;同时测试中 scale 由 `(input.size(0), 1)` 形状改为 1-D、量化后 `reshape(-1, 1)` 对齐,scale 的 assert 容差从 `atol=1e-5` 放宽到 `atol=1e-3`,参数矩阵从 5x5 组合缩减为 3x3。正文未解释这些改动与标题所称精度问题的确切关系。

## 修复方案
- `sgl-kernel/csrc/gemm/per_token_quant_fp8.cu`:量化循环改用乘以倒数:
```diff
+  const float scale_val = 1.0f / block_max;
...
-      float val = fmaxf(fminf(static_cast<float>(input_vec[j]) / block_max, FP8_E4M3_MAX), -FP8_E4M3_MAX);
+      float val = fmaxf(fminf(static_cast<float>(input_vec[j]) * scale_val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
```
- `sgl-kernel/setup.py`:在两个 nvcc 架构分支的 `else` 路径追加 `-use_fast_math`。
- `sgl-kernel/tests/test_per_token_quant_fp8.py` 与 `benchmark/bench_per_token_quant_fp8.py`:scale 形状改为 1-D(测试中 `reshape(-1, 1)` 返回),基准脚本补 scale/output diff 计算;测试容差与参数矩阵调整如上。
- 验证:正文所附日志为改动后的测试通过结果(9 + 240 + 9 passed)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:sgl-kernel 的 FP8 per-token 量化内核数值路径(除法变乘法,快速数学编译下舍入行为可能变化,测试容差相应放宽到 atol=1e-3)、sgl-kernel 构建选项(非 sm100/sm90 架构全局启用 `-use_fast_math`,可能影响该条件下其它内核的数值行为)、量化 API 的 scale 张量形状约定(1-D)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
