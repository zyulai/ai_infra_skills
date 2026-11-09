# PR #42169: [Bugfix] Fix DeepSeek v4 topk numerical issue for unaligned max-model-len

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42169](https://github.com/vllm-project/vllm/pull/42169) |
| 作者 | wzhao18 |
| 合并日期 | 2026-05-10 |
| labels | bug, deepseek, ready |
| 热度 | 👍 2 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
max-model-len 未按 256 对齐时 DeepSeek v4 topk stride 算错导致数值异常;改用 logits.stride(0) 修复。

## 问题现象
DeepSeek v4 在某些 `--max-model-len` 取值(如 900000)下出现数值问题:同一问题采 100 个回复,生成 token 数分布相对不传该参数时有明显偏移。GPQA e2e(main, TP, `--max-model-len 900000`):pass@1 从 89.39% 降到 81.82%,no_answer 0.00%→5.05%,avg_tokens 11754→25955,gen_seconds 3361→12300;DEP 配置下分布则基本正常。

## 根因
正文披露:indexer 的 persistent topk kernel 用 `const int64_t stride = logits.size(1);` 计算行跨度;而 DeepGEMM 的 `fp8_fp4_paged_mqa_logits` 会把 `max_model_len` 维度 pad 到 256 对齐,再把视图 slice 回 `max_model_len`。当 `max_model_len` 非自然对齐时,得到的 logits 不连续(NonContiguous),`size(1)` 不再等于真实行跨度,stride 计算错误。

## 修复方案
`csrc/topk.cu` 两处(`launch_persistent_topk` 与 `persistent_topk`)把逻辑宽度换成真实行跨度:

```cuda
-  const int64_t stride = logits.size(1);
+  const int64_t stride = logits.stride(0);
```

## 影响与回归风险
- 影响窗口:正文未写明版本范围(触发条件为 `--max-model-len` 非 256 自然对齐 + DeepGEMM fp8_fp4 paged mqa logits 路径)。
- 回归面:`csrc/topk.cu` persistent topk kernel 的 stride 语义从"逻辑列数"改为"实际行跨度",该 kernel 的所有调用方都受影响;对连续 logits 两种取值一致。验证:patch 后 token 数分布恢复正常;GPQA 上 TP 88.89%、DEP 87.88%(max-model-len 900000),接近无 max-model-len 基线。正文测试计划仅两项(token 分布、e2e 精度),未列专门单元测试。

## 关联
- [DeepGEMM `fp8_fp4_paged_mqa_logits`](https://github.com/deepseek-ai/DeepGEMM/blob/891d57b4db1071624b5c8fa0d1e51cb317fa709f/csrc/apis/attention.hpp#L361)(调查引用)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
