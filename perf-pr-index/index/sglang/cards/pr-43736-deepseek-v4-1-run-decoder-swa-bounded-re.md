# PR #43736: [DeepSeek-V4.1] Run decoder SWA bounded replay inside the breakable prefill CUDA graph

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jyizheng |
| 合并 | 2026-10-13 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | deepseek, run-ci |
| Diff | diffstat 缺失(pack 未提供 +add/-del 行数与文件数;diff 内容可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/43736 |

## 一句话结论

让 decoder SWA bounded replay 跑进 breakable prefill CUDA graph,消除 host-launch-bound 的 eager prefill,GB300 上 TTFT p50 约降 40%(≤16 并发)。

## 动机

`--enable-decoder-swa-bounded-replay` 之前拒绝所有 prefill CUDA graph backend,DeepSeek-V4.1 开启 bounded replay 时 prefill 只能 eager 执行。GB300 上 eager prefill 是 host-launch-bound(每 request 每 rank 约 2.5k 次 launch,~230 ms step 里只有 ~10 ms GPU 工作),是低/中并发下 TTFT 的主导成本。这与 #42078、#42909、#43440、#43534 针对的是同一缺口;本 PR 的差异是让 late layers 留在同一个 breakable prefill graph 内跑 bounded-replay tail,而不是单独 capture late-layer graph 或在 full rows 上运行捕获的 late layers。

## 优化点

- late layers(最后一个 KV source layer 之后)在 captured graph 内改跑 per-shape tail block:`tail_rows = min(max_running_requests x 128, shape tokens)`。tail 的 extend/positions/scatter 索引与 SWA 写位置放在固定容量的 persistent buffer(padding 行指向 slot 0 的 dummy),graph replay 无需重分配;每个 block size 一份 tail metadata,按 prefill-graph 方式构建(`use_prefill_cuda_graph=True`)并在每次 replay 时 in-place 刷新,因为被捕获的 low-ratio compressor/indexer 按地址读张量。
- `DeepseekV4Model.forward` 在 enter/exit eager break 处切到 tail(`bcg_deepseek_v4_enter/exit_late_layer_tail`),回放时 re-run 以换到 live batch 的 tail metadata;`_scatter_tail_rows` 新增 graph 分支,用 `scatter_indices` 把 tail 行散回(padding 行落到 extend 之外的 dummy 行再被 slice 丢弃)。
- runner 的 `_trim_logits_output` 透传 `hidden_states_token_indices`,DSpark injector 拿到正确行。
- `init_forward_metadata_in_graph` 在 decode/target-verify graph 路径清掉 `tail_forward_metadata`,避免 prefill graph 捕获/回放残留导致 decode publish 报 `_SparseTable is not a prefill publish`。
- `validate_deepseek_v41_features` 放行 `Backend.BREAKABLE` 与 decoder bounded replay 组合(encoder replay 仍 eager);capture 宽度(`--cuda-graph-max-seq-len-prefill`,默认 16k)不变。

## 关键改动

```text
python/sglang/srt/layers/attention/deepseek_v4_backend.py: tail 容量计算、persistent tail buffers、tail metadata 构建与 replay 刷新(_prefill_graph_tail_metadata)
python/sglang/srt/models/deepseek_v4.py: bcg enter/exit eager break 包装与 _scatter_tail_rows 的 graph 分支
python/sglang/srt/arg_groups/deepseek_v4_hook.py: 校验放行 BREAKABLE + bounded replay
python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py: _trim_logits_output 透传 hidden_states_token_indices
```

## 性能声明

DeepSeek-V4.1-Flash 1P1D(prefill TP4 + decode TP4,DSpark PD,FlashMLA),GB300,真实生产 trace(ISL median 6.7k / p90 127k),16k capture:

- 三臂对比(eager+bounded replay / graph 内 full rows / graph 内 tail):TTFT p50 tail 臂 conc=1 272→152 ms、conc=4 496→327 ms、conc=8 748→545 ms、conc=16 1,215→1,226 ms、conc=32 3,304→3,415 ms;tail 方案保持 eager 级 p90,4-16 并发下每 chip 输出 +5-8%。
- 同 base A/B(Oct-8 nightly,含 any-context replay):TTFT p50 约 −40% 至 16 并发(conc=1 263→142 ms −46%,conc=4 447→210 ms −53%),p90 约 −10%,吞吐 +1…+5%,TPOT 持平;PR 注明 conc=16 的 p90 +46% 为 run variance。
- Idle-worker floor:11-token 请求 258-302→59-60 ms,11k cached-prefix 请求 298-320→65-72 ms(Oct-8 nightly 200→72 ms、230→78 ms)。
- Graph pool 16k capture 在 GB300 约 39 GB(prefill `mem_fraction_static` 0.78→0.70)。

## 验证与回归风险

精度:74 个 shape capture;needle prompts(4.5k/11k/40k)与 greedy 24k/40k/86k-token prompt 的文本与 eager 一致(偏差在 eager run-to-run noise 内);六档 real-trace replay 每档 300 s 全部 100% 成功;在 Sep-24 nightly 系列与 Oct-8 nightly(7760d526)验证,`main` 上的 GPU run PR 自述仍欠。风险:不支持 prefill CP(assert);ROCm 的 DP attention + TP MoE late-layer DP counts 仍不兼容;`can_run_prefill_cuda_graph` 新增 tail 容量检查,超容量退回 eager;依赖 decode graph 路径清除 `tail_forward_metadata`。

## 关联

前序同缺口 PR:#42078、#42909、#43440、#43534;后续:#43733(在此之上实现任意 context length 的 graph replay)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
