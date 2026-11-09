# PR #18608: [V1][Sampler] Improve performance of FlashInfer sampling by sampling logits instead of probs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @lgeiger |
| 合并 | 2025-05-26 |
| 分类 | Attention / Kernel \| tier: major |
| Label | ready, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/18608 |

## 一句话结论

FlashInfer 采样改从 logits 而非 probs 走 top-k/top-p，L40s 微基准 +9.3%、gemma-3-12b 吞吐 +1.75%。

## 动机

`top_k_top_p_sampling_from_probs` 路径调用 `(softmax) -> top_k_renorm_probs -> top_p_sampling_from_probs`，而 `top_k_top_p_sampling_from_logits` 调用 `top_k_mask_logits -> softmax -> top_p_sampling_from_probs`，正文称后者更快。V1 sampler 原先在 `forward_cuda` 里无条件先对 logits 做 softmax 再传入 FlashInfer，本 PR 用替换采样入口来消除这条较慢路径。

## 优化点

把 FlashInfer 采样入口从 probs 换成 logits：`forward_cuda` 去掉无条件的 `logits.softmax(dim=-1, dtype=torch.float32)`，仅在 `k`/`p` 均为 None、走 `random_sample` 的分支里才计算 probs；`flashinfer_sample` 的参数由 `probs` 改为 `logits`，top-p only 与 top-k only 分支各自在内部做 softmax，而 top-k 与 top-p 同时存在时直接调用 `top_k_top_p_sampling_from_logits(logits, k, p, deterministic=True)`。函数 docstring 同步改为 "Sample from the logits using FlashInfer"，pack 中仅改动 `topk_topp_sampler.py` 一个源码文件。

## 关键改动

```text
vllm/v1/sample/ops/topk_topp_sampler.py: forward_cuda 去掉无条件 softmax，仅 random_sample 分支计算 probs
vllm/v1/sample/ops/topk_topp_sampler.py: flashinfer_sample 改接收 logits，top-k + top-p 分支调用 top_k_top_p_sampling_from_logits
```

## 性能声明

微基准：L40s GPU、jitted FlashInfer、CUDA 12.8，正文称"In a quick micro benchmark on an L40s GPU I am seeing a 9.3 % speedup with this PR and jitted flashinfer using Cuda 12.8."；复现脚本参数为 batch_size 40、num_classes 262208、dtype float、k=64、p=0.95、`deterministic=True`、5000 iters（正文未给微基准绝对时间数值）。端到端：`vllm serve google/gemma-3-12b-it --disable-log-requests` 加 `benchmarks/benchmark_serving.py --backend openai-chat --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat --hf-split train --num-prompts 1000`，正文称"1.75 % improvement in throughput"：Baseline 984 successful、duration 187.19 s、Request throughput 5.26 req/s、Output token throughput 619.43 tok/s、Total 1128.87 tok/s；This PR 984、184.04 s、5.35 req/s、630.47 tok/s、1148.62 tok/s。PR 未注明端到端的精度（dtype）与硬件条件。

## 验证与回归风险

PR 正文未注明测试计划、测试结果或回归风险。diff 仅改 `vllm/v1/sample/ops/topk_topp_sampler.py`；`flashinfer_sample` 的 docstring 注明其输出不必与 `random_sample` 逐项一致、只保证统计等价（statistically equivalent），且 FlashInfer 0.2.3+ 不支持 per-request generators，带 generators 时回退到 `forward_native`。正文还说明 `k`、`p` 全为 None 时优先走 `random_sample`（无需排序、避免 CPU-GPU 同步），有 `generators` 时因 FlashInfer 0.2.3+ 不支持 per-request generators 而回退 `forward_native`。两组 serving 结果的 successful requests 均为 984，TTFT、TPOT、ITL 的均值与分位数差异不大，吞吐提升主要来自 duration 缩短。正文未列出受影响模型或采样参数组合。

## 关联

无。正文未给出可关联的 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
