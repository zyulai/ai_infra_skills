# PR #58216: [Perf] DiffusionGemma: constrained reads over the request's logprob_token_ids

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mmastrac |
| 合并 | 2026-09-24 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | documentation, performance, ready |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 主体见 `vllm/model_executor/models/diffusion_gemma.py`、`vllm/utils/diffusion.py` 及示例/测试 |
| 链接 | https://github.com/vllm-project/vllm/pull/58216 |

## 一句话结论

DiffusionGemma 新增 `diffusion_constrained` 读路径,把 unembedding/采样/自条件限制在请求给定的 token 集合上,单次 read 约省 25% GPU 时间。

## 动机

请求已传 `logprob_token_ids` 时,答案集合其实是已知的,但此前 unembedding 仍对全部 262k 词表做 GEMM,采样、熵与自条件也在全词表空间进行。对 constrained diffusion(结构化答案)来说,这些全词表计算是纯粹浪费。该 PR 是 #57250 的 followup,目标是让引擎只在这个集合上做计算。

## 优化点

新增 `diffusion_constrained` sampling 参数:当一步内所有 decode slot 共享同一 allowed 集合时,`compute_logits` 只对 tied embedding 中 K 个 gathered 行做 `[rows, K]` 线性层,Gumbel 采样、argmax、熵与收敛判断都在 K 空间完成,再把选中的索引经 `allowed` 映射回真实 token id。自条件 soft embed 的乘法同样只用 K 行 embedding。当同一 batch 内 slot 的集合不一致时,退化为全词表路径,但用 `_mask_rows_to_allowed` 把 constrained 请求的 logit 行按各自集合 mask(mask 值取有限大负数 -1e20,避免 0 * -inf 产生 NaN,保证熵有限)。renoise 仍从全词表采样,因为模型训练时用的是随机词表噪声,集合内噪声会产生级联乱码。约束集要求 TP=1。

## 关键改动

```text
vllm/model_executor/models/diffusion_gemma.py: compute_logits K 空间 GEMM 快路径、
  _mask_rows_to_allowed 行级 mask、states.constrained/batch_allowed/step_allowed
  状态与 logprobs 重归一化输出
vllm/utils/diffusion.py: 校验 diffusion_constrained 必须配 logprob_token_ids
examples/features/structured_diffusion/structured_server.py: 服务端默认开启,
  新增 --no-constrained 开关
tests/v1/sample/test_diffusion_gemma_reads.py: batch_allowed 与 mask 行为测试
tests/test_sampling_params.py: 参数校验测试
```

关键片段(`compute_logits` 快路径):

```python
allowed = states.step_allowed
if allowed is not None:
    logits = torch.nn.functional.linear(
        hidden_states, self.lm_head.weight[allowed]
    ).float()
```

## 性能声明

PR 正文:实验约快 25%(整体);单次 read 约 1/4 GPU 时间减少。详细基准:GB10 本地、warm engine、每 arm 新 prompt、两轮取范围、single-slot reads、one step、canvas 48。yes/no 题:c1 延迟中位 105-106 ms → 95-96 ms,answers/s 9.5 → 10.5;c32 延迟中位 551-561 ms → 420 ms,p90 575-621 → 430-440 ms,answers/s 56.7-56.9 → 76.0-76.2。4 选 1 题:c1 延迟 98-100 ms → 88-92 ms,answers/s 10.0-10.2 → 10.9-11.4;c32 延迟中位 502-530 → 354-382 ms,answers/s 60.0-60.1 → 81.1-82.8。作者总结:c1 约 10% 提速,c32 多 34-38% answers/s、延迟约少四分之一,因为批处理步骤中全词表计算占比更大。

## 验证与回归风险

新增测试覆盖参数校验(缺 logprob_token_ids 报错、类型必须 bool)、`batch_allowed` 混合集合回退逻辑、`_mask_rows_to_allowed` 只 mask constrained 行、无约束行时 no-op 且不改动原张量。风险点:mask 值与 softmax 等价性依赖 -1e20 与温度 clamp 的数值边界;非共享集合 batch 走全词表 + mask 副本,每步多一次全词表拷贝;功能仅限 TP=1。

## 关联

前序:#57250(DiffusionGemma 结构化 diffusion 基础)。同属 Model-specific Fast Path 的 DiffusionGemma 系列改动。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
