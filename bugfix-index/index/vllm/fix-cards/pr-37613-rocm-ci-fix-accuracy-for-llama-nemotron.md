# PR #37613: [ROCm][CI] Fix accuracy for llama-nemotron-vl pooling tests

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37613](https://github.com/vllm-project/vllm/pull/37613) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-03-20 |
| labels | rocm, ready, multi-modality, llama |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
ROCm 上 HF 与 vLLM attention backend 不同,致 llama-nemotron-vl pooling 测试小幅精度差;补 ROCm 引擎参数并放宽 ROCm 容差到 0.022。

## 问题现象
ROCm CI(mi250_1 的「Multi-Modal Models (Extended Pooling)」)中,llama-nemotron-vl 的 pooling/rerank 测试因 HF 与 vLLM 输出存在小幅精度差而不匹配;为 #34839 的 follow-up,动机为 Buildkite AMD CI 失败(正文附链接)。

## 根因
正文披露:ROCm 上 HF 与 vLLM 使用不同的 attention backend,带来略高的数值方差(diff 注释原文:"ROCm shows slightly higher numerical variance dues to different attention backend between vLLM and HF")。

## 修复方案
`tests/models/multimodal/pooling/test_llama_nemotron_vl.py`:
1. embedding 与 reranker 两处 VllmRunner 构造追加 `**ROCM_ENGINE_KWARGS`;
2. reranker 的分数比较容差按平台区分,ROCm 为 0.022、其余平台保持 0.02:

```diff
+    rel_tol = 0.022 if current_platform.is_rocm() else 0.02
     for i, (hf_score, vllm_score) in enumerate(zip(hf_scores, vllm_scores)):
-        assert hf_score == pytest.approx(vllm_score, rel=0.02), (
+        assert hf_score == pytest.approx(vllm_score, rel=rel_tol), (
```

## 影响与回归风险
- 影响窗口:正文未说明(纯 CI 测试改动)。
- 回归面:仅测试文件:ROCm 上该模型 embedding/rerank 测试的引擎参数与比较容差;生产代码路径不受影响。容差从 0.02 放宽到 0.022 后,ROCm 上同类的小幅数值漂移更不易触发测试失败。

## 关联
- #34839(follow-up 对象)
- Buildkite AMD CI 失败链接(motivation,正文给出)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
