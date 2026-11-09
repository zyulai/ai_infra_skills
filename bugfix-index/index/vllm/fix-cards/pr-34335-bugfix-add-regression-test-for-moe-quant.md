# PR #34335: [Bugfix] Add regression test for MoE quant_config under torch.compile

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34335](https://github.com/vllm-project/vllm/pull/34335) |
| 作者 | mgehre-amd |
| 合并日期 | 2026-02-20 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 4 |
| 分类 | MoE |
| bug 类型 | regression |

## 一句话结论
torch.compile 下 MoE 的 quant_config 初始化副作用不被 Dynamo 重放,W4A16 模型报 Hidden size mismatch;修复经 #34371 落地,本 PR 补回归测试。

## 问题现象
MoE Refactor(#32344)之后,W4A16 模型在 torch.compile 下失败:`AssertionError: Hidden size mismatch 2048 != 1024`。复现:`vllm serve RedHatAI/Qwen3-30B-A3B-Instruct-2507.w4a16 --max-model-len 1024 --gpu-memory-utilization 0.85`,不加 `--enforce-eager` 即命中断言;加 `--enforce-eager` 则正常,因为 `forward_native` 以 eager 方式运行、属性修改能实际生效。影响面:W4A16 量化(如 AWQ 4-bit)的 MoE 模型 + torch.compile(非 eager)组合。

## 根因
正文披露:`ensure_moe_quant_config_init()` 在 `FusedMoE.forward_native()` 中被调用;torch.compile 激活时 `forward_native` 被 Dynamo 追踪,但设置 `self.quant_method.moe_quant_config` 这一属性修改副作用不会在运行时重放,导致运行时 `DefaultMoERunner.forward_impl` 在 `moe_forward` custom op 内执行时 `moe_quant_config` 仍为 `None`。对 W4A16 量化 MoE,这使 `use_int4_w4a16` 为 `False` 而非 `True`,断言 `hidden_states.size(1) == w1.size(2)` 失败——4-bit 打包权重的维度只有预期的一半。

## 修复方案
本 PR 只添加回归测试,修复本身经 #34371 单独落地:在 `DefaultMoERunner.forward_impl()` 开头调用 `layer.ensure_moe_quant_config_init()`,该函数在 `moe_forward` custom op 内运行、不被 Dynamo 编译。新增测试 `tests/quantization/test_compressed_tensors.py::test_w4a16_moe_torch_compile`:加载微型 W4A16 MoE 模型 `nm-testing/tinysmokeqwen3moe-W4A16-first-only-CTstable`,`enforce_eager=False`、`cudagraph_mode=NONE`,验证推理成功。已验证该测试在无修复时失败(报 Hidden size mismatch)、有修复时通过;`tests/kernels/moe/test_moe.py` 537 通过、1 个无关 OOM 跳过。

```python
+def test_w4a16_moe_torch_compile(vllm_runner):
+    model_path = "nm-testing/tinysmokeqwen3moe-W4A16-first-only-CTstable"
+    with vllm_runner(
+        model_path,
+        enforce_eager=False,
+        max_model_len=256,
+        compilation_config={"cudagraph_mode": "NONE"},
+    ) as llm:
+        output = llm.generate_greedy("Hi", max_tokens=1)
+        assert output
```

## 影响与回归风险
- 影响窗口:正文说明问题由 #32344(MoE Refactor)引入,修复经 #34371 落地。
- 回归面:本 PR 仅新增测试文件用例,不改运行时代码,无运行时回归风险;测试覆盖 torch.compile + W4A16 MoE 的组合路径。

## 关联
- #34371(实际修复所在 PR)
- #32344(引入问题的 MoE Refactor)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
