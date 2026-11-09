# PR #51824: [Bugfix] vLLM crashes at startup when DeepEP v2 is used with `--enforce-eager` wiht TRTLLM Bf16

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51824](https://github.com/vllm-project/vllm/pull/51824) |
| 作者 | SageMoore |
| 合并日期 | 2026-08-20 |
| labels | bug, ready, nvidia |
| 热度 | 👍 2 · 💬 8 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
DeepEP V2 以 expanded 布局返回 topk_ids 时 TRTLLM Bf16 MoE 后端取错 top_k,启动即崩;改为直接取 `topk_ids.size(1)`。

## 问题现象
执行如下命令时 vLLM 启动即崩溃:
```
vllm serve deepseek-ai/DeepSeek-V2-Lite \
--data-parallel-size 2 \
--enable-expert-parallel \
--all2all-backend deepep_v2 \
--enforce-eager
```
报错:
```
expert_indices.size(1) == args->top_k (1 vs. 6) : expert_indices dim1 must match top_k.
```

## 根因
正文披露:DeepEP V2 以"expanded"布局返回 topk_ids 时,`TrtLlmBf16ExpertsModular` 没有正确提取 `top_k` 值,把错误的 top_k 传给 kernel,与 expert_indices 的实际维度(1 vs. 6)不匹配,导致断言失败崩溃。

## 修复方案
改 `vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py` 的 `apply()`:不再使用 `self.topk`,改为按实际输入张量取值:
```python
-            top_k=self.topk,
+            top_k=topk_ids.size(1),
```
同时扩充 `tests/kernels/moe/test_deepep_v2_moe.py` 单测:参数化覆盖 `flashinfer_trtllm`、`flashinfer_cutlass`、`trtllm_fp8` 三种 experts 后端 × expanded/非 expanded 布局 × cudagraph 开关,抽出 `_make_experts()` 复用权重构造逻辑;`tests/kernels/moe/parallel_utils.py` 的 DeepEP V2 构造加 `allow_hybrid_mode=False`。

## 影响与回归风险
- 回归面:仅 `TrtLlmBf16ExpertsModular.apply()` 中传给 TRTLLM kernel 的 `top_k` 来源;DeepEP V2 expanded 布局下由"错误常量"变为"实际列数",非 expanded 布局与其它后端路径不受影响(测试为新增参数化,不改产品逻辑)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
