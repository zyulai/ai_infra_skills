# PR #31776: [Bugfix][CI/Build] Fix failing pooling models test due to Triton kernel accuracy diff

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#31776](https://github.com/vllm-project/vllm/pull/31776) |
| 作者 | Isotr0py |
| 合并日期 | 2026-01-06 |
| labels | ready |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
Triton kernel 相比旧 FlexAttention 后端有约 0.001 精度差,ModernBERT pooling 测试在 CI 失败;放宽测试容差修复。

## 问题现象
CI 中 pooling 模型测试失败(源自 [#31406 的评论](https://github.com/vllm-project/vllm/pull/31406#issuecomment-3713125841)),失败用例为 `test_modernbert_models[float-disham993/electrical-ner-ModernBERT-base]`(`tests/models/language/pooling/test_token_classification.py`)。正文说明:Triton kernel 与之前的 FlexAttention 后端相比存在约 0.001 的轻微精度误差,超出原断言容差。作者本地无法复现("Can't reproduce locally"),希望通过放宽容差让 CI 变绿。

## 根因
正文披露:Triton kernel 相比此前的 FlexAttention 后端有约 0.001 的轻微精度(~0.001)误差,与原测试 `atol=1e-2` 的断言在 CI 环境下冲突。未披露更深层的数值来源分析。

## 修复方案
仅改测试文件 `tests/models/language/pooling/test_token_classification.py`,把 `torch.allclose` 断言换成容差显式、更宽松的 `torch.testing.assert_close`:

```python
-        assert torch.allclose(hf_output, vllm_output, atol=1e-2)
+        torch.testing.assert_close(hf_output, vllm_output, atol=1.2e-2, rtol=1e-3)
```

atol 从 1e-2 放宽到 1.2e-2,并引入 rtol=1e-3。生产代码无改动。

## 影响与回归风险
- 影响窗口:正文未写明版本范围。
- 回归面:纯测试改动,不影响任何运行时路径;风险在于测试容差被放宽,可能掩盖 ModernBERT pooler 在 Triton 后端下更大的数值偏差(正文承认本地未能复现失败、无法进一步定位)。测试计划为单测命令重跑,正文未贴出修复后的 CI 结果。

## 关联
- [#31406(评论)](https://github.com/vllm-project/vllm/pull/31406#issuecomment-3713125841)(CI 失败出处)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
