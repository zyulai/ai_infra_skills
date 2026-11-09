# PR #36412: Fix/resupport nongated fused moe triton

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @shaunkotek |
| 合并 | 2026-03-09 |
| 分类 | MoE |
| Label | documentation, performance, rocm, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, qwen, cpu, kv-connector, nvidia |
| Diff | diffstat 缺失(pack 中为 +None/-None, None files);diff 仅 2 个文件、共 4 处小改 |
| 链接 | https://github.com/vllm-project/vllm/pull/36412 |

## 一句话结论

重新放开 Triton fused MoE 对非门控(nongated,无 gate×up 相乘)激活的支持,使 SILU/GELU/ReLU2 的 no-mul 变体走 Triton 融合 MoE kernel。

## 动机

Triton fused MoE 后端此前声明不支持 `no_act_and_mul`(非门控激活,即只有激活、没有 SwiGLU 式的乘法分支),导致使用这类激活(MoEActivation 的 *_NO_MUL 变体)的模型无法选用 Triton fused MoE 路径,只能回退其他实现。kernel 实际具备该能力,只是能力开关被置 False,需要恢复。

## 优化点

改动是能力声明的恢复,不触碰 kernel 本体:在 `fused_moe.py` 与 `fused_batched_moe.py` 两个 Triton 后端类中,`_supports_no_act_and_mul()` 由 `return False` 改为 `return True`;同时 `_supports_activation()` 的支持列表加入 `MoEActivation.SILU_NO_MUL`、`MoEActivation.GELU_NO_MUL`、`MoEActivation.RELU2_NO_MUL` 三种非门控激活。选择逻辑据此重新允许 nongated fused MoE 走 Triton kernel。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/fused_moe.py: _supports_no_act_and_mul 改 True;_supports_activation 加 3 个 *_NO_MUL 激活
vllm/model_executor/layers/fused_moe/fused_batched_moe.py: _supports_no_act_and_mul 改 True
```

```python
@staticmethod
def _supports_no_act_and_mul() -> bool:
    return True
```

```python
MoEActivation.SILU_NO_MUL,
MoEActivation.GELU_NO_MUL,
MoEActivation.RELU2_NO_MUL,
```

## 性能声明

无。PR 正文未给出任何 benchmark 数字,Test Result 仅一句"models run properly, tests pass"。正文未注明测试的模型、硬件或 workload。

## 验证与回归风险

验证方式按 Test Plan 为:在 FP8 与 FP16 上加载 Triton kernel 确认正常运行,并运行既有测试确认无回归;PR 未列出具体测试文件。风险面:该开关同时影响普通 fused MoE 与 batched fused MoE 两条路径,凡 `*_NO_MUL` 激活的 MoE 模型此后会命中 Triton kernel,数值正确性依赖既有 kernel 实现(本 PR 未改 kernel 代码);对使用门控激活的模型无影响。

## 关联

无(pack 素材中未给出同区域 PR 信息)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
