# PR #32384: [Bugfix] Fix xgrammar dtype mismatch on macOS CPU inference

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32384](https://github.com/vllm-project/vllm/pull/32384) |
| 作者 | karanb192 |
| 合并日期 | 2026-03-14 |
| labels | bug, ready, structured-output, v1 |
| 热度 | 👍 11 · 💬 12 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
macOS CPU 上 xgrammar 要求 float32 logits 而推理用 float16/bfloat16,structured output 报错;加 dtype 转换修复。

## 问题现象
macOS CPU 推理下使用 xgrammar 的 guided decoding 失败。复现脚本(来自 issue):`LLM(model="Qwen/Qwen3-0.6B", max_model_len=512)`,配 `StructuredOutputsParams(json=base_schema)`、`temperature=0.0` 的 `SamplingParams` 生成。修复前报 `ValueError: logits must be of type float32`;修复后成功生成 structured output。

## 根因
正文披露:xgrammar 的 CPU kernel 要求 `float32` logits,而 macOS CPU 推理使用 `float16`/`bfloat16`,应用 bitmask 时 dtype 不匹配。

## 修复方案
`vllm/v1/structured_output/utils.py` 的 `apply_grammar_bitmask()`:当 logits 在 CPU 且非 float32 时,先转 float32 应用 bitmask,再拷回原 dtype:

```python
if logits.device.type == "cpu" and logits.dtype != torch.float32:
    # Convert to float32, apply bitmask, then convert back
    logits_float32 = logits.to(torch.float32)
    xgr.apply_token_bitmask_inplace(
        logits_float32, grammar_bitmask, indices=index_tensor
    )
    # Copy the modified values back to the original tensor
    logits.copy_(logits_float32.to(logits.dtype))
else:
    xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)
```

修法来自 @Inokinoki 在 #31901 中的建议。

## 影响与回归风险
- 影响窗口:正文未提及具体受影响版本范围。
- 回归面:仅 `apply_grammar_bitmask` 新增的「CPU 且非 float32」分支;GPU 及已是 float32 的 CPU 路径走原 else 分支不变。CPU 非 fp32 场景多一次 float32 往返拷贝开销。

## 关联
- Fixes [#31901](https://github.com/vllm-project/vllm/issues/31901)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
