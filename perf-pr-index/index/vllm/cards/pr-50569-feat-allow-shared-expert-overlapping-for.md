# PR #50569: feat: allow shared expert overlapping for FlashInfer one-sided all-to-all

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jdebache |
| 合并 | 2026-08-11 |
| 分类 | MoE (tier: major) |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);diff 可见 1 个文件(shared_experts.py) |
| 链接 | https://github.com/vllm-project/vllm/pull/50569 |

## 一句话结论

放开 FlashInfer one-sided all2all 的共享专家重叠,DeepSeek-V2-Lite 上 TPOT 改善约 5-7%。

## 动机

PR 正文 Purpose:此前 PR #28377 因在 `deepep_low_latency` 上观察到正确性问题,在 EPLB 场景下对所有非 `allgather_reducescatter` 后端一律禁用 shared expert overlap。本 PR 在验证 `flashinfer_nvlink_one_sided` 开启重叠能产出正确结果后,把它从该限制中豁免。

## 优化点

把 `SharedExperts._disable_shared_experts_overlap` 中的判断,由「EPLB 且后端 != `allgather_reducescatter` 就禁用」改为维护一个安全后端白名单 `_EPLB_OVERLAP_SAFE_BACKENDS = ("allgather_reducescatter", "flashinfer_nvlink_one_sided")`,仅当 EPLB 启用且后端不在白名单内(或使用 `use_fi_nvl_two_sided_kernels`)时才禁用重叠。注释也从 non-default backend 改为 non-safe backend。改动只涉及调度开关,不改变 kernel 计算本身。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/runner/shared_experts.py: 引入 _EPLB_OVERLAP_SAFE_BACKENDS 白名单,把 flashinfer_nvlink_one_sided 加入允许重叠的后端
```

关键机制代码:

```python
        # Both these comm backends have been shown to be safe for shared expert overlap.
        _EPLB_OVERLAP_SAFE_BACKENDS = (
            "allgather_reducescatter",
            "flashinfer_nvlink_one_sided",
        )
...
        return (
            parallel_config.enable_eplb
            and parallel_config.all2all_backend not in _EPLB_OVERLAP_SAFE_BACKENDS
        ) or parallel_config.use_fi_nvl_two_sided_kernels
```

## 性能声明

PR 正文 Test Result(DeepSeek-V2-Lite,TP=1 / DP=4 / EP=4,EPLB 开启,`--all2all-backend flashinfer_nvlink_one_sided`,4× NVIDIA GB300):性能用 `vllm bench serve --dataset-name random --input-len 1000 --output-len 1000 --max-concurrency 1 --num-warmups 3 --num-prompts 30 --backend openai`。带重叠(本 PR) vs 不带重叠(baseline):输出吞吐 227.96 vs 223.80 tok/s,mean TPOT 4.15 vs 4.39 ms,median TPOT 3.18 vs 3.42 ms,正文称 TPOT 改善约 5-7%。精度(GSM8K 200 题):0.330 vs 0.350,作者说明差异在 200 题采样噪声内、无正确性回归。

## 验证与回归风险

Test Plan:在 4× GB300 上用 `vllm serve deepseek-ai/DeepSeek-V2-Lite ... --enable-eplb --all2all-backend flashinfer_nvlink_one_sided` 起服务,跑 `tests/evals/gsm8k/gsm8k_eval.py --num-questions 200` 精度检查与 `vllm bench serve` 性能检查。影响路径:EPLB + `flashinfer_nvlink_one_sided` 后端的 shared expert 重叠调度;正文声明精度差异在采样噪声内,其余后端(尤其 `deepep_low_latency`)仍保持禁用,未改动通信 kernel 本身。

## 关联

前序 PR #28377(正文所引,原先在 EPLB 下禁用非 allgather_reducescatter 后端的共享专家重叠)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
