# PR #50018: [Perf] model_overrides: Default GLM-5.3-Flash to an agentic Mamba pool split

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @alphabetc1 |
| 合并 | 2026-11-12 |
| 分类 | Overlap / Scheduler(tier: major) |
| Label | documentation, run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/50018 |

## 一句话结论

为 `Glm5NextForConditionalGeneration` 新增模型级默认:`--mamba-full-memory-ratio 0.25` + `extra_buffer_lazy` cache 策略,把几乎闲置的 KDA state pool 让给 KV pool——agentic 多轮 trace 回放中 prefix-cache hit rate 0.056→0.767(叠加 #49886 达 0.887),输出吞吐 3.4x(叠加 #49886 达 4.6x、超过 FlashStates)。

## 动机

GLM-5.3-Flash 默认以 `--mamba-full-memory-ratio` 0.9 把 weight 后的内存分成 KDA state pool 与 KV pool。agentic 多轮流量下 KV pool 低于会话工作集:每个会话在下一轮之前就被逐出,prefix-cache hit rate 崩到 0.056,而 state pool 大部分时间闲置。

## 优化点

新增 `model_overrides/glm5_next.py` 的 `Glm5NextForConditionalGeneration` override,每个默认都让位于显式值:

- `--mamba-full-memory-ratio 0.25`:更小的 state pool,其余给 KV。
- `--mamba-radix-cache-strategy extra_buffer_lazy`(在 `auto` 会选 `extra_buffer` 且 lazy 变体受支持时——无 PD disaggregation、overlap scheduler 开启、无 pipeline parallelism):每请求一个 tracking slot 而非两个,更小的 pool 也能容纳足够请求。
- 后续 #50041 把 cookbook 的 High Throughput recipes(以及 `TestGLM53FlashB200HighThroughput`)固定回 `--mamba-full-memory-ratio 0.9`:大量短并发请求时更小的 state pool 会限制 running requests(B200 DP-attention CI cell 在 0.25 时降到每 rank 17)。

## 关键改动

```text
python/sglang/srt/arg_groups/model_overrides/glm5_next.py(新增): Glm5NextForConditionalGeneration 的 mamba_full_memory_ratio 0.25 与 extra_buffer_lazy 默认(条件:extra_buffer 受支持、无 PD、overlap 开、pp_size 1)
python/sglang/srt/arg_groups/model_overrides/__init__.py: 导入 glm5_next
test/registered/unit/test_model_overrides.py: 新增 TestGlm5NextAgenticMambaPool(默认与显式值/不支持/PD/overlap off/PP 用例)
```

```python
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    if cfg.mamba_full_memory_ratio is None:
        overrides["mamba_full_memory_ratio"] = 0.25
    if (
        cfg.mamba_radix_cache_strategy == "auto"
        and supports_mamba_cache_extra_buffer(cfg, hf_config)
        and cfg.disaggregation_mode == "null"
        and not cfg.disable_overlap_schedule
        and cfg.pp_size == 1
    ):
        overrides["mamba_radix_cache_strategy"] = "extra_buffer_lazy"
```

## 性能声明

(AgentX closed-loop trace replay,FlashStates harness:GLM-5.3-Flash NVFP4,TP2 B300,48 sessions,2700 s,按 900–2700 s 计分,device only,总 pool 内存取 FlashStates 的 67 GB KV budget,同一 host)

| | Mamba slots / KV tokens / max running | hit rate | TTFT p50 / p90 (s) | output tok/min/GPU |
|---|---|---|---|---|
| main, default | 429 / 5.89M / 85 | 0.056 | 54.8 / 65.6 | 7,563 |
| main + 本 PR, default flags | 179 / 8.91M / 44 | 0.767 | 0.20 / 9.58 | 25,777 |
| #49886 + 本 PR, default flags | 179 / 8.91M / 44 | 0.887 | 0.13 / 4.42 | 35,165 |
| FlashStates (vLLM fork) | - / 10.45M / - | 0.878 | 0.21 / 1.86 | 32,846 |

- 比例来自同预算下的 slot sweep:144–192 slots(ratio 0.19–0.27)可容纳 48 sessions;300 slots(ratio 0.48)已过悬崖(hit 0.217);120 slots 限制 running requests 并损失约 15% throughput。
- 本 PR 单独使输出吞吐较 main 提升 3.4x;叠加 #49886(mamba checkpoints 只在 fork points 与 request ends)达 4.6x,在同一内存预算下超过 FlashStates。
- GSM8K(sgl-eval 0.1.3,1319 题,temperature 1.0,top_p 0.95,TP2 B300):本 PR 97.50%,main 97.57%。

## 验证与回归风险

- `test/registered/unit/test_model_overrides.py` 单测覆盖默认值、显式 ratio / strategy、extra buffer 不受支持、PD、overlap 关闭与 PP;每个条件都做了负面控制。
- 真实 checkpoint 解析验证(GLM-5.3-Flash):default → `extra_buffer_lazy` / 0.25;显式 `extra_buffer` → 保留;显式 ratio 0.9 → 保留;`--disaggregation-mode prefill` → `extra_buffer`;`--disable-overlap-schedule` → `extra_buffer`。
- 风险:0.25 的小 state pool 在大量短并发请求场景会限制 running requests(#50041 因此把高吞吐 recipes 固定回 0.9;B200 DP-attention CI cell 在 0.25 时降到每 rank 17);每个默认均让位于显式值,回退成本低。

## 关联

#49886(mamba checkpoints 仅在 fork points 与 request ends,叠加后 hit 0.887 / 吞吐 4.6x)、#50041(follow-up,把 cookbook High Throughput recipes 与 B200 high-throughput CI 固定回 ratio 0.9)、FlashStates(vLLM fork 对照基线)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
