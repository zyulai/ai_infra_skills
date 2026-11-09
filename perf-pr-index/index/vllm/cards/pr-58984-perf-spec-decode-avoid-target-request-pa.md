# PR #58984: [Perf][Spec Decode] Avoid target request padding in DP drafter

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wangyicong52 |
| 合并 | 2026-10-20 |
| 分类 | Model-specific Fast Path(major) |
| Label | speculative-decoding, ready, mrv2, dflash |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/58984 |

## 一句话结论

DP drafter 不再复用 target FULL CUDA graph padding 后的容量,改用真实最大请求数派发,减少草稿图 padding 浪费。

## 动机

当 target 侧使用 FULL CUDA graph 时,`DPSyncState.num_reqs` 描述的是 graph 容量而非实际调度的请求数。uniform drafter 步骤复用这个容量,会让活跃的 DP rank 选择比需要更大的 draft graph,在 dummy rank 和真实 rank 上都产生多余的 padding 计算。

## 优化点

`DPSyncState.num_reqs` 语义改为"任何 rank 在 graph padding 之前调度的最大真实请求数"(始终取 `num_reqs_across_dp.max()`,不再被 FULL descriptor 的容量覆盖),uniform drafter dispatch 用它替代 graph 容量。同时把 online dummy-rank DFlash batch 在输入准备、graph dispatch、attention metadata、EPLB 准备与 draft 生成之间一致地收缩(propose 时 `num_reqs = min(num_reqs, dp_sync_state.num_reqs)`,`max_target_query_len` 只在前 `num_reqs` 个请求上取 max)。启动期 graph capture 保留所需容量;prefill 必须保持 target batch 形状,只有 decode 允许收缩(`target_dependent_ar/speculator.py` 同样处理)。

## 关键改动

```text
vllm/v1/worker/gpu/dp_utils.py: num_reqs 改为 padding 前真实最大请求数
vllm/v1/worker/gpu/spec_decode/dflash/speculator.py: dummy_run 时按真实请求数收缩输入与派发
vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py: decode 路径同样收缩
```

## 性能声明

DeepSeek-V4.1-Flash,两个非生产节点(每节点 8x NVIDIA H20 96GB,arm 内 DP8 + EP),`FULL_DECODE_ONLY` CUDA graph + dspark投机配置,`vllm bench serve` random 输入 3500 / 输出 1500,seed 42,`num-prompts = 10 * concurrency`:输出吞吐 C1 +1.98%、C8 +14.31%、C16 +8.72%、C64 +1.06%;mean TPOT -1.57% / -9.94% / -9.06% / -1.47%;mean TTFT 除 C64(+2.84%)外均下降(如 C1 -9.88%)。acceptance rate 两臂间波动(C1 -8.58pp、C64 -4.97pp,C8/C16 上升)。GSM8K 全量 1319 题:baseline 96.29% vs 本 PR 95.83%,McNemar 检验 p=0.26318 无统计显著差异;EvalScope mean latency -5.90%、平均输出吞吐 +7.65%。

## 验证与回归风险

测试:GSM8K 全量配对评估 + 端到端 bench(两臂逐点 byte-identical 输入输出,对官方 nightly overlay 本 PR 的 4 个 Python 文件并校验运行时 SHA-256)。回归风险:仅影响 DP + uniform drafter(dspark/dflash 与 target_dependent_ar)路径;acceptance rate 在部分并发点下降,作者归因于两臂间波动;`prepare_dflash_inputs` 中 `assert num_reqs <= input_batch.num_reqs` 约束收缩上限。

## 关联

同区域为 DP 下投机解码调度优化;label 含 dflash、mrv2。pack 未列出具体前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
