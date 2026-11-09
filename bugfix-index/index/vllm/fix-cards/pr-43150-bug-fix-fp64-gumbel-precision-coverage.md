# PR #43150: [BUG] Fix FP64 Gumbel precision coverage

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43150](https://github.com/vllm-project/vllm/pull/43150) |
| 作者 | tianyu-z |
| 合并日期 | 2026-06-05 |
| labels | bug, ready, speculative-decoding, v1 |
| 热度 | 👍 2 · 💬 6 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
`--use-fp64-gumbel` 只覆盖显式 Triton Gumbel 采样,V1 采样与 spec decode 的指数竞赛路径仍用 fp32 噪声;将 fp64 贯穿全部等价路径。

## 问题现象
开启 `--use-fp64-gumbel` 后,V1 top-k/top-p 采样、spec decode recovered-token 采样、LLM draft proposer 采样仍走 `q.exponential_(); probs / q; argmax` 的 fp32 指数噪声路径。正文论证:该写法数学上等价于 Gumbel-max(`argmax(log(probs) - log(q))`),而 CUDA 上 fp32 随机数无法表示 fp64 能表示的极小下尾事件;普通单 token 自回归采样影响通常很小,但宽分布/大量并行 categorical 竞赛时缺失的下尾会变得可观测,并系统性抹掉稀有 token 的命中。H100 + PyTorch 2.9.1+cu126 实测:2 亿次 fp32 指数采样 min 恰为 2^-24、无一低于该阈值,float64 则有 8 个低于阈值;many-tail race(trials=100000、tail_tokens=262144、gap=20.5)期望尾命中 32.76,fp32 得 0 次,float64 得 32 次。

## 根因
正文披露:fp32 指数采样在 CUDA 上继承 24 位均匀分布的下尾截断(约 2^-24);由于指数竞赛写法与 Gumbel-max 数学等价,此前只修显式 Gumbel 采样器的 fp64 开关并未覆盖这些等价路径。

## 修复方案
把 `use_fp64_gumbel` 贯穿各路径:`Sampler`/`TopKTopPSampler.__init__` 新增参数并从 `ModelConfig` 接线(`gpu_model_runner.py`);`topk_topp_sampler.py` 新增 `empty_exponential_noise_like`(按需 float64)与 `sample_with_exponential_noise`(dtype 不一致时 `q.reciprocal_()` 后乘 probs),`random_sample` 与 forward_cpu 走 fp64,请求 fp64 时 forward_cuda/forward_hip 回落 native;`rejection_sampler.py` 的 `sample_recovered_tokens` 按 flag 用 float64 噪声并给 Triton kernel 加 `USE_FP64_GUMBEL` constexpr(累加器 `max_val` dtype 随之切换);`llm_base_proposer.compute_probs_and_sample_next_token` 同样接入(注意此处改用 `probs.clone()` 以免污染后续 rejection sampling 用的 draft_probs)。默认 `use_fp64_gumbel=False` 的 fp32 快速路径不变,并新增 `tools/gumbel_precision/prove_exponential_race_precision.py` 证明脚本与针对性回归测试。

## 影响与回归风险
- 影响窗口:正文未给出具体版本范围(flag 存在但未覆盖这些路径的期间)。
- 回归面:V1 采样(top-k/top-p 的 native/CUDA/HIP/CPU 路径)、rejection sampler recovered-token 采样、draft proposer 采样;FlashInfer/aiter 在请求 fp64 时改走 native 回退;默认关闭时行为不变;开启 fp64 会显著降低吞吐(正文提及)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
