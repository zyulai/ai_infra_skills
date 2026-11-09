# PR #51794: [ROCm][Perf] Enable CSA multi-stream overlap for DeepSeek-V4

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @shen-shanshan |
| 合并 | 2026-09-14 |
| 分类 | Overlap / Scheduler |
| Label | rocm, ready, deepseek, cpu, nvidia, DSv4 |
| Diff | diffstat 缺失(pack 未提供 +/− 统计,仅含源码 diff) |
| 链接 | https://github.com/vllm-project/vllm/pull/51794 |

## 一句话结论

DeepSeek-V4 CSA 层在 ROCm 启用三流 overlap,CSA 层快 12.99%,各并发档吞吐均提升。

## 动机

PR 之前,DeepSeek-V4 attention 的三条 aux stream 在 ROCm 上被显式禁用(`current_platform.is_rocm()` 时 `aux_stream_list = None`),原因是此前的 hang 问题。这导致输入投影阶段的两个 compressor `wkv_gate` GEMM 等工作全部落在串行关键路径上。这些 compressor 链与 fused wqa+wkv 主链相互独立、各自直接从 `hidden_states` 出发,天然适合并行,值得为此重建 ROCm 多流路径。

## 优化点

attention forward 在输入 GEMM 前分叉三条 HIP 流:默认流跑 fused wqa+wkv GEMM → q/kv RMSNorm → wq_b → qnorm/RoPE → SWA KV insert;aux stream 0 跑 main-compressor `wkv_gate` GEMM 加压缩 kernel(压缩 KV cache 写入);aux stream 1 跑 indexer-compressor `wkv_gate` GEMM 加 K-cache 写。每条链自包含,不需要细粒度跨流依赖;join 之后 indexer weights GEMM、sparse indexer op(top-k 选择)与 MLA attention 仍在主流串行。核心是 `_enable_csa_multi_stream()` gate:piecewise cudagraph 的 eager breaks 会在 owning stream 上重建 attention 输入,此时分叉侧流要依赖运行时 HIP event sync,而 ROCm 上 event wait 可能 hang,因此只在 cudagraph capture 内或非 dict metadata(全图 / profile run)时启用 overlap,其余区域退回 `_run_sequential_pipeline()` 顺序执行。同时给 indexer 增加 `skip_compressor` 参数,避免 aux stream 1 上已执行的 indexer compressor 被重复调度。

## 关键改动

```text
vllm/models/deepseek_v4/amd/model.py: 取消 ROCm 对 aux_stream_list 的禁用,恢复 3 条 CUDA Stream
vllm/models/deepseek_v4/amd/rocm.py: ROCm CSA fork/join 实现 + _enable_csa_multi_stream gate + 顺序回退
vllm/models/deepseek_v4/attention.py: Indexer.forward 新增 skip_compressor,compressed_kv_score 改为 Optional
vllm/models/deepseek_v4/cpu/cpu_sparse.py: CPU override 适配 Indexer 新签名
```

```python
(q, qr_out, qr_scale_out, kv_out), _ = execute_in_parallel(
    default_chain,
    [main_compressor_chain, indexer_compressor_chain],
    self.ln_events[0],
    self.ln_events[1:3],
    aux_streams[:2],
    enable=True,
)
```

## 性能声明

- Trace 对比(take conc=4 为例):CSA layer 快 **12.99%**。
- Benchmark:SA InferenceX `8k1k` workload,`nightly-rocm100` 镜像,DeepSeek-V4-Pro,TP8、max-model-len 9472、fp8 KV cache、aiter MoE backend、FULL_AND_PIECEWISE cudagraph、async-scheduling。conc=1:输出吞吐 66.30→69.34 tok/s(+4.59%),TTFT 441.89→369.48 ms(−16.39%),TPOT −3.97%;conc=4 吞吐 +3.82%;conc=16 +2.30%(TTFT +0.50%,唯一回退点);conc=64 +1.33%;TPOT 在四档全部下降。

## 验证与回归风险

测试为 SA InferenceX benchmark 与 GSM8K 精度(5-shot exact_match 0.9439,1319 例,超过 94% 阈值)。风险点:ROCm 多流此前正因 hang 被禁,本 PR 靠 `_enable_csa_multi_stream()` 把 overlap 限制在 capture 内/非 dict metadata 区域,不可 overlap 区域走顺序回退;若 gating 判定失效(diff 注释提及 MRV1 eager break 场景曾产生垃圾输出,关联 #51430)可能复现 hang 或错误结果。改动触及共享的 `deepseek_v4/attention.py`(Indexer 签名变化)与 CPU 路径 `cpu_sparse.py`,非 ROCm 平台的 DeepSeek-V4 也需回归。

## 关联

同批 ROCm attention 方向:#57978(AITER MLA page-index 展开并行)。diff 注释引用 #51430(MRV1 垃圾输出问题)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
