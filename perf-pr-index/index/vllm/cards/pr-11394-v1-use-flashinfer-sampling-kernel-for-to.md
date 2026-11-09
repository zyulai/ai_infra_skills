# PR #11394: [V1] Use FlashInfer Sampling Kernel for Top-P & Top-K Sampling

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @WoosukKwon |
| 合并 | 2024-12-27 |
| 分类 | Attention / Kernel（tier: major） |
| Label | ready |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 显示改动 6 个文件：新增 `vllm/v1/sample/ops/topk_topp_sampler.py`、`vllm/v1/sample/ops/penalties.py`、`vllm/v1/sample/ops/__init__.py`，修改 `vllm/v1/sample/sampler.py`、`vllm/envs.py`、`tests/v1/sample/test_sampler.py` |
| 链接 | https://github.com/vllm-project/vllm/pull/11394 |

## 一句话结论

V1 sampler 引入 FlashInfer 的 top-p/top-k 拒绝采样 kernel，免去对整个 logits 排序，ShareGPT 下 top-p 吞吐 40.20→47.15 reqs/s，基本抹平约 20% 的排序开销。

## 动机

PyTorch 原生 top-p/top-k 实现需要对整个 logits 张量做全量排序，batch 大时该张量非常可观，排序 op 在高吞吐场景下约占 20% 的性能开销。FlashInfer 提供基于拒绝采样（rejection sampling）的 top-p/top-k kernel，不需要排序。当时 V1 sampler 的 `_apply_top_k_top_p` 正是纯 PyTorch sort 版本，且带 TODO 注释等待自定义 kernel。

## 优化点

新建 `TopKTopPSampler` 模块：CUDA 上若 FlashInfer 可用且未显式禁用则走 `forward_cuda`——先 softmax 成 probs，无需 top-k/top-p 时仍用 `random_sample`（避免 CPU-GPU 同步），否则调 `flashinfer.sampling` 的 `top_p_sampling_from_probs` / `top_k_sampling_from_probs` / `top_k_top_p_sampling_from_probs`（最多 32 轮均匀随机数的拒绝采样，deterministic=True）；若 `success` 不全为真则回退为先 renorm 再 `sampling_from_probs`。环境变量 `VLLM_USE_FLASHINFER_SAMPLER` 改为三态 Optional[bool]：默认 None 在 V0 解释为 False、在 V1 解释为 True（即 V1 + FlashInfer 可用时默认启用，V0 行为不变，保持向后兼容）。同时对 V1 Sampler 结构重构：penalty 逻辑抽到 `ops/penalties.py`；top-k logprobs 改用 penalty/temperature 之前的原始 logits 计算（与 V0 不同），且先算 logprobs 再做可能 in-place 修改 logits 的操作，避免 clone。

## 关键改动

```text
vllm/v1/sample/ops/topk_topp_sampler.py: TopKTopPSampler，FlashInfer 拒绝采样 top-k/top-p，含 renorm 回退
vllm/v1/sample/ops/penalties.py: min_token/presence/frequency/repetition penalty 抽取
vllm/v1/sample/sampler.py: 流程重排（logprobs 先行）、greedy/random 分派、接入 TopKTopPSampler
vllm/envs.py: VLLM_USE_FLASHINFER_SAMPLER 变三态，V1 默认启用
```

```python
if no_top_k:
    next_token_ids, success = flashinfer.sampling.top_p_sampling_from_probs(
        probs, uniform_samples, p, deterministic=True)
elif no_top_p:
    next_token_ids, success = flashinfer.sampling.top_k_sampling_from_probs(
        probs, uniform_samples, k, deterministic=True)
```

## 性能声明

ShareGPT benchmark（PR 正文未注明模型/硬件等条件）：main 无 top-p/top-k 47.40 reqs/s；main 开 top-p 40.20 reqs/s；本 PR 开 top-p 47.15 reqs/s——即 top-p 采样开销从约 20% 降到接近零（与无 top-p 相比差 0.5%）。除此之外无其他量化数据。

## 验证与回归风险

测试：`tests/v1/sample/test_sampler.py` 改为直接对 `apply_penalties` 的输出 logits 断言（min_tokens/penalty 测试不再依赖 SamplerOutput 的排序 logprobs），默认 `max_num_logprobs` 改为 0。风险面：`flashinfer_sample` 内含 CPU-GPU 同步（代码注释要求放在 forward 末尾以减小开销）；FlashInfer 输出与 `random_sample` 仅"统计等价"，逐 token 结果可能不同；采样行为依赖 FlashInfer 可用性，不可用时 warning 并回退 PyTorch 原生路径；ROCm/非 CUDA 平台走 `forward_native`。

## 关联

本 PR 是 V1 sampler 采样路径的起点；索引内后续演进：#61996（小 batch split top-p 流水线优化，作用在 Triton 版 `apply_top_k_top_p` 上）、#37225（apply_top_k_top_p top-k 搜索优化）。PR 正文未列关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
