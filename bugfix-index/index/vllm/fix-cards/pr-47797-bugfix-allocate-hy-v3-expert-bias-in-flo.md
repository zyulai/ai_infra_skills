# PR #47797: [Bugfix] Allocate HY V3 expert_bias in float32 to prevent silent downcasting

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47797](https://github.com/vllm-project/vllm/pull/47797) |
| 作者 | aoright |
| 合并日期 | 2026-07-08 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 1 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
Hunyuan-V3 的路由偏置 `expert_bias` 按服务 dtype(如 bf16)分配,checkpoint 里的 float32 bias 加载时被 `copy_` 静默降精度,边界 token 路由翻转;改为显式 float32 分配。

## 问题现象
Hunyuan-V3(`hy_v3` / `hy_v3_mtp`)MoE 模型执行中,router 选择偏置 `expert_bias` 以服务 dtype 而非 `torch.float32` 分配。在默认 torch dtype 为 `torch.bfloat16`/`torch.float16` 时该参数为 16 位精度;checkpoint 加载时,权重文件中真实的 float32 bias 张量被 `param.data.copy_()` 静默降为 16 位。由于该 bias 直接参与 top-k 路由决策,降精度会翻转处于决策边界上的 token 的路由选择,影响 MoE 输出正确性。对应 issue #47777。

## 根因
正文披露:`HYV3MoEFused` 中 `self.expert_bias = nn.Parameter(torch.empty(config.num_experts))` 未指定 dtype,参数继承默认 torch dtype(bf16/fp16);`copy_()` 加载时只做类型转换不报错,造成静默 downcast。对照:deepseek_v2.py 与 Hugging Face 的 `_keep_in_fp32_modules_strict` 都把该类 bias 保留在 fp32,hy_v3 缺了这层处理。

## 修复方案
`vllm/model_executor/models/hy_v3.py` 一处:显式固定 dtype:

```diff
-        self.expert_bias = nn.Parameter(torch.empty(config.num_experts))
+        self.expert_bias = nn.Parameter(
+            torch.empty(config.num_experts, dtype=torch.float32)
+        )
```

另新增单元测试 `tests/models/test_hy_v3_expert_bias.py`,验证分配 dtype 与 `copy_` 行为;`pytest --noconftest tests/models/test_hy_v3_expert_bias.py` 结果 1 passed。

## 影响与回归风险
- 影响窗口:正文未提及具体版本/提交范围,不写。
- 回归面:仅 `hy_v3.py` 的 `expert_bias` 参数分配;参数变为 float32 后,checkpoint 中的 fp32 bias 可无损加载,`scoring_func="sigmoid"`、`e_score_correction_bias = self.expert_bias` 引用路径的类型随参数本身改变(路由计算中该 bias 以 fp32 参与)。对非 bf16/fp16 默认 dtype 的部署,行为与之前一致(本就是 fp32 语义)。

## 关联
- closes https://github.com/vllm-project/vllm/issues/47777

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
