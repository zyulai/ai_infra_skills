# PR #56853: [ROCm][Perf] Enable HCA dual-stream overlap for DeepSeek-V4

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @shen-shanshan |
| 合并 | 2026-09-17 |
| 分类 | Overlap / Scheduler（tier: major） |
| Label | rocm, ready, deepseek, DSv4 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files); pack 末尾无 dropped-from-diff 说明(diff 未裁剪文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/56853 |

## 一句话结论

ROCm HCA 层主 compressor 与默认流双流重叠：HCA 层快 7.47%，8k1k 输出吞吐 +0.96~3.43%。

## 动机

#51794 已为 ROCm 上的 CSA 层启用 multi-stream overlap，但 HCA 层（有 main compressor、无 indexer）仍是单流，因为 ROCm attention subclass 在检测到无 indexer 时会丢弃 aux streams。在 `gfx950` 上 HCA main compressor 占 attention layer decode 时间很大一部分，且与 query/KV 链完全独立：两条分支都只读层输入 `hidden_states` 加各自的权重与状态，default-stream 链在 sparse attention 之前根本不消费 compressor 的输出，因此可以并行。

## 优化点

HCA 层的 attention forward 改为在 gated region 内 fork：default stream 跑 fused wqa/wkv GEMM → q/kv RMSNorm → wq_b → qnorm/RoPE → SWA KV insert；aux stream 0 跑 HCA main compressor（fused wkv/wgate score GEMM + DeepseekCompressor）。两条流在 sparse MLA attention 之前 join，后者读取压缩 KV cache 与 SWA window。串行的 MHC fused post+pre prelude 仍在 default stream 上先跑，fork 的 start event 把 aux 链排在它之后。代码上不再"无 indexer 就置 `aux_stream_list = None`"，改为仅 `compressor is None`（dense 层）时才禁用流，并把 `_enable_csa_multi_stream` 更名 `_enable_multi_stream_overlap`。

## 关键改动

```text
vllm/models/deepseek_v4/amd/rocm.py: HCA 层保留 aux_stream_list 并实现 default/aux 双流 fork-join；仅 dense（compressor is None）层禁用；gating 由 _enable_csa_multi_stream 更名为 _enable_multi_stream_overlap
```
```text
        if self.indexer is None:
            # Dense layers have no compressor work to overlap; HCA layers
            # (compressor, no indexer) keep the streams for the dual-stream
            # fork below.
            if self.compressor is None:
                self.aux_stream_list = None
        else:
            # Disable indexer inner overlap.
            self.indexer.aux_stream = None
```

## 性能声明

Trace 对比（以 conc=4 为例）标注 HCA layer 改动后 7.47% faster。Benchmark 为 SA InferenceX `8k1k` workload，`vllm serve deepseek-ai/DeepSeek-V4-Pro`，TP8、max-model-len 9472、max-num-seqs 512、`--kv-cache-dtype fp8`、`--moe-backend aiter`、`--compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'`，全部基于 nightly-rocm100 镜像。Output token throughput（tok/s）Main → This PR：Conc=1 69.61 → 72.00（+3.43%↑）、Conc=4 229.96 → 236.13（+2.68%↑）、Conc=16 642.98 → 653.01（+1.56%↑）、Conc=64 1298.06 → 1310.56（+0.96%↑）；Mean TTFT Conc=1 429.89 → 365.36 ms（-15.01%↓）、Conc=64 1600.01 → 1601.33 ms（+0.08%↑）；Mean TPOT Conc=1 13.92 → 13.51 ms（-2.95%↓）。正文结论：整体正向，主要来自更高输出吞吐与更低 TPOT，低并发收益更明显，高并发 TTFT 基本中性无回归。测试条件：模型 DeepSeek-V4-Pro、硬件为 ROCm（gfx950 / nightly-rocm100 镜像）、FP8 KV cache、8k1k workload；正文未列出具体 GPU 型号与精度以外的更多细节。

## 验证与回归风险

验证包括上述 benchmark 与 gsm8k 精度（`lm_eval`，`local-completions`，num_concurrent 64/256、num_fewshot 5/30、max_gen_toks 2048）：Conc=64 5-shot flexible-extract 0.9477 / strict-match 0.9484；Conc=256 30-shot 0.9515 / 0.9522，全量 1,319 例两档均超过要求的 94% 阈值。回归风险：改动仅作用于 ROCm 上 DeepSeek-V4 的 HCA 层路径，dense 层仍禁用双流；高并发下吞吐与 TPOT 增益收窄、TTFT 基本中性；正文未声明对非 ROCm 后端或其他模型的影响。

## 关联

正文明确 #51794（此前为 ROCm CSA 层启用 multi-stream overlap，本 PR 补齐 HCA 层）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
