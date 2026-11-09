# PR #51411: [Bugfix][Quantization] Fix INT8 W8A8 MoE crash in TritonExperts

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51411](https://github.com/vllm-project/vllm/pull/51411) |
| 作者 | djramic |
| 合并日期 | 2026-08-08 |
| labels | bug, quantization |
| 热度 | 👍 2 · 💬 6 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
TritonExperts 不认 torch.int8 输入,INT8 W8A8 MoE 引擎初始化即断言崩溃;加入 int8 并用 BF16 计算修复。

## 问题现象
INT8 W8A8 MoE 模型在处理量化激活时崩溃:`tests/quantization/test_quark.py::test_quark_int8_w8a8_moe[1]` 失败,报 `RuntimeError: Engine core initialization failed`,栈指向 `fused_moe/experts/triton_moe.py` 第 249 行 `apply` 中的 `assert hidden_states.dtype in [...]` 抛 `AssertionError`。修复后该测试 1 passed。

## 根因
正文披露:TritonExperts 未把 `torch.int8` 识别为合法输入 dtype,INT8 W8A8 MoE 模型处理量化激活时直接命中输入类型断言而崩溃。

## 修复方案
`vllm/model_executor/layers/fused_moe/experts/triton_moe.py` 的 `apply`:把 `torch.int8` 加入受支持的激活类型列表,并让 int8 与 fp8 一样以 BF16 作为计算类型,使 INT8 激活能正确通过 Triton MoE 路径:

```python
elif (
    hidden_states.dtype == torch.float8_e4m3fn
    or hidden_states.dtype == torch.float8_e4m3fnuz
    or hidden_states.dtype == torch.int8
):
    compute_type = tl.bfloat16
```

## 影响与回归风险
- 影响窗口:正文未写明版本范围。
- 回归面:仅 Triton MoE experts 的 `apply` 输入类型分支;非 int8 输入走原分支不变。INT8 激活现在以 bf16 精度参与计算(与 fp8 路径同款 compute type),量化精度表现依赖该假设,正文未附数值精度对比。验证仅 `test_quark_int8_w8a8_moe` 一项;正文 checklist 未勾选,无 e2e 或性能数据。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
