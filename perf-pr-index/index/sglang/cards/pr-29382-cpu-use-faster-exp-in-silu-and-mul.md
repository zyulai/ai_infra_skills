# PR #29382: [CPU] use faster exp in silu_and_mul

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mingfeima |
| 合并 | 2026-06-29 |
| 分类 | Other (tier: major) |
| Label | sgl-kernel, cpu, run-ci |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 仅含 sgl-kernel/csrc/cpu/activation.cpp 一处 1 行替换 |
| 链接 | https://github.com/sgl-project/sglang/pull/29382 |

## 一句话结论

把 CPU `silu_and_mul_cpu` 向量路径的 `exp()` 换成 PyTorch 向量库的 `exp_u20()` 近似指数,在 32 核 gen6 Xeon 上拿到 6%–29% 提速。

## 动机

CPU 推理中 SiLU/gated activation 是 MoE/MLP 的热点算子,`silu_and_mul_cpu` 向量路径此前调用 `exp()`,底层走 PyTorch 的 sleef u10 实现(精度高但慢)。sgl-kernel 其他 CPU kernel 已在使用更快的向量指数近似,该 kernel 没有跟上;推理 hot path(softmax、SiLU、gated activations、Mamba/FLA recurrences)对 `exp` 精度要求不高,可以用 ULP 误差更大的近似换取吞吐。

## 优化点

改动本身是单行替换:向量 lambda 中 `x.neg().exp()` 改为 `x.neg().exp_u20()`。`exp_u20` 是 PyTorch `at::vec::Vectorized<float>` 提供的向量化指数近似,误差为 20 ULP(相对 IEEE-754 float 语义),比 sleef u10 快,对推理精度足够。标量 fallback 路径(`std::exp`)保持不变,即只优化批量的 SIMD 向量路径,小批尾部的标量计算不受影响。

## 关键改动

```text
sgl-kernel/csrc/cpu/activation.cpp: silu_and_mul_cpu 的 Vec lambda 中 exp() → exp_u20(),标量路径不变
```

```cpp
[](float x) { return x / (1.f + std::exp(-x)); },
[](Vec x) { return x / (Vec(1.f) + x.neg().exp_u20()); });
```

## 性能声明

测试环境:32 核 gen6 Xeon(PR 注明)。结果(作者正文表格):

| Shape | Dtype | Before | After | Speedup |
| --- | --- | --- | --- | --- |
| (1, 18432) | bf16 | 10.311 us | 9.103 us | 1.133x |
| (17, 18432) | bf16 | 29.097 us | 22.425 us | 1.298x |
| (1000, 18432) | bf16 | 462.275 us | 392.141 us | 1.179x |
| (1, 18432) | fp16 | 12.866 us | 11.206 us | 1.148x |
| (17, 18432) | fp16 | 27.284 us | 25.828 us | 1.056x |
| (1000, 18432) | fp16 | 518.754 us | 401.485 us | 1.292x |

总体 **6% 到 29% speedup**。未给出端到端模型 tokens/s 数字。

## 验证与回归风险

作者运行 `python test/registered/cpu/test_activation.py` 做正确性验证。精度回归风险:exp 误差从 sleef u10 放宽到 20 ULP,单层激活影响很小,但 PR 正文只做了 kernel 级单测,未附端到端模型精度评估(如 gsm8k)结果。CI 状态注:PR 页面最新 PR Test(Base/Extra)显示为失败(:x:)——pack 中 CI States 记录如此,合并是否在修复后通过未注明。影响范围限于 CPU 后端的 silu_and_mul,不影响 GPU 路径。

## 关联

同区域:sgl-kernel CPU 路径其他算子(softmax、Mamba/FLA recurrences)也在正文提及使用同类近似;pack 未列出具体关联 PR 编号,关联暂记"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
