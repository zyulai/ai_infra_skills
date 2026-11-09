# PR #30302: [AMD] [MORI-EP] Skip LocalExpertCount kernel in decode graph when not recording

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @rbrugaro-amd |
| 合并 | 2026-07-07 |
| 分类 | MoE(tier: major) |
| Label | 无 |
| Diff | pack diffstat 缺失;PR 正文注明单文件 moriep.py +19/−2 |
| 链接 | https://github.com/sgl-project/sglang/pull/30302 |

## 一句话结论

跳过 decode CUDA graph 中无人消费的 mori LocalExpertCount kernel,decode TPOT/输出吞吐提升约 1–2%。

## 动机

mori EP token dispatcher 的 `_should_record_expert_distribution()` 此前只要 CUDA stream 正在 capturing 就返回 True。这导致 mori 的 `LocalExpertCountKernel`(及其 buffer memset)被烤进 decode CUDA graph,并在每次重放中执行——包括根本没有配置 expert-distribution recorder 的普通 serving。此时产出的 `local_expert_count` 没有任何消费者(recorder hook 未激活),属于 decode 热路径上的纯死工作。影响所有经 mori EP all-to-all 后端、开启 CUDA graph 的 MoE 模型。

## 优化点

把 capture 时刻的门控从"是否在 capturing"改为"是否实际配置了 recorder(非 Noop)":没有配置 recorder(常见 serving 路径)时,capturing 期间直接跳过 count kernel;已配置非 Noop recorder 时,即使尚未 `start_record()` 仍在 capture 期烤入机制,保证 `start_record()` 在 capture 之后调用仍能生效(实际 gather 仍由 recorder 自身的 `_on_hook` 按 recording/capturing 门控);正在 recording 时行为不变。由于两条 recording 路径(configured-but-not-yet-recording 与 actively-recording)完全未动,不存在 expert-distribution 记录功能回归。

## 关键改动

```text
python/sglang/srt/layers/moe/token_dispatcher/moriep.py: 重写 _should_record_expert_distribution() 的 capture 期门控
```

```python
def _should_record_expert_distribution() -> bool:
    recorder = get_global_expert_distribution_recorder()
    if recorder.recording:
        return True
    if torch.get_device_module().is_current_stream_capturing():
        return not isinstance(recorder, _ExpertDistributionRecorderNoop)
    return False
```

## 性能声明

8×MI355X,DeepSeek-R1 MXFP4,mori EP(tp8/ep8/dp8、DP attention、aiter、full decode CUDA graph,`--cuda-graph-max-bs-decode 32`),bench_serving random 输入 256/输出 1024,seed 42:

| concurrency | Median TPOT (ms) baseline→patched | Output tput (tok/s) baseline→patched |
|---|---|---|
| 16 | 19.78 → 19.56(−1.1%) | 793.79 → 802.86(+1.1%) |
| 32 | 21.20 → 20.98(−1.0%) | 1473.77 → 1490.12(+1.1%) |
| 64 | 23.25 → 22.85(−1.7%) | 2631.25 → 2683.34(+2.0%) |

Median ITL 同向变化(−1.1%…−1.2%)。PR 注明 decode batch 上限为 32,更大 decode batch 下收益上限可能更高。

## 验证与回归风险

patch 后无 recorder 配置:gsm8k(5-shot、200 questions、parallel 128)accuracy 0.965、invalid 0.000;EPLB `stat_approx` 记录流程(`expert_distribution_recorder_mode=stat_approx` + start/dump)仍产出非空 per-layer expert counts,gsm8k 0.975 不变。风险面集中在 expert-distribution recorder/EPLB 路径与 mori EP 后端;若存在 Noop 之外的 recorder 类型判定差异,需确认 capture 期烤入行为不受影响。

## 关联

无(pack 未给出同区域其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
