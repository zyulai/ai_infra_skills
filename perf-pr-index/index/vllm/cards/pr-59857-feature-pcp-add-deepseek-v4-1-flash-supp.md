# PR #59857: [Feature][PCP] Add DeepSeek-V4.1-Flash support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wangyicong52 |
| 合并 | 2026-10-16 |
| 分类 | Speculative Decoding（tier: major） |
| Label | documentation, ready, ci/build, deepseek, kv-connector, mrv2, DSv4.1, mooncake |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾标注 8 file(s) dropped from diff for size |
| 链接 | https://github.com/vllm-project/vllm/pull/59857 |

## 一句话结论

DeepSeek-V4.1-Flash 新增 PCP，8×H20 下 8192-token prefill 吞吐 +31.48%、TTFT -23.95%(TP2/PCP4)。

## 动机

正文 Overview：在 NVIDIA CUDA 上为 Model Runner V2 的 FlashMLA sparse attention 增加 text-only DeepSeek-V4.1-Flash 的 Prefill Context Parallelism（PCP）支持。该 PR 有意只针对 colocated 单模型实例，P/D disaggregation 与 KV transfer 不在初始范围；目标是在不混入 KV handoff、replay 语义的前提下先确立独立 PCP 的正确性，后续再逐项扩展边界。实现上通过用 gathered predecessor rows 闭合被 PCP chunk 切分的 ratio-2 pair，保持 compressed KV、indexer、SWA、Engram lookback、MoE routing inputs 与 padding 在 PCP ranks 间一致。

## 优化点

Implementation 段列出机制：在每个 rank 的本地 token rows 上压缩 ratio-2 groups，只 gather 每个本地 chunk 的最后 FP32 行来闭合跨 rank pair 并维持 open ring rows 一致；PCP linkage 保存在 model-specific compressor metadata，`InputBatch` 与 `PCPManager` 相对 `main` 不变；在一次与 ratio 1 共享的 cache-insert 路径中 gather BF16 compressor latents，用对齐后的 gathered positions/slots 写 main KV 与 Indexer K；保持 Engram lookback、MoE routing token IDs、padding、compressed KV、indexer keys、SWA metadata 对齐，且 PCP collectives 顺序在各 rank 一致（含空本地 shard 与无本地 decode rows 的请求）。Rank-local Compressor Revision 采用 Lucas Wilkinson 的 wangyicong52/vllm#1（commit `b06bea7abc4eb5cf6965fe704d16257ac7c70697`），移除 `InputBatch.pcp_metadata`/`PCPBatchMetadata` plumbing 与全局 FP32 `kv_score` gather。diff 中 `attention.py` 增加 `use_pcp` 开关、用 `get_pcp_group().all_gather` 汇总 latent/kv、PCP 下走 `fused_deepseek_v4_kv_rope_insert` 且只保留 rank-local queries（`slot_mapping[:0]`）；`fused_compress_quant_cache.py` 的 `fused_save_compress_norm` 新增 `prev_rows`/`prev_row_indices` 供 chunk 读取跨 rank 前驱行；`amd/model.py` 对 PCP>1 直接抛 NotImplementedError（CUDA-only）。

## 关键改动

```text
vllm/models/deepseek_v41/attention.py: 增加 use_pcp 开关，all_gather latent/kv，PCP 下 fused KV rope insert 与 rank-local query 处理
vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py: fused_save_compress_norm 新增 prev_rows/prev_row_indices，闭合跨 PCP chunk 的 ratio-2 pair
vllm/models/deepseek_v41/amd/model.py: PCP>1 抛 NotImplementedError，明确仅 NVIDIA CUDA 支持
tests/kernels/test_compressor_kv_cache.py、test_engram.py、tests/models/test_deepseek_v41_pcp.py: PCP 相关 kernel 与模型测试
.buildkite/test_areas/models_basic.yaml: CI 配置

# 机制摘录 (attention.py: PCP latent all_gather)
        if self.use_pcp and latent is not None:
            assert isinstance(attn_metadata, dict)
            swa_metadata = cast(
                "DeepseekSparseSWAMetadata", attn_metadata[self.swa_cache_layer.prefix]
            )
            assert swa_metadata.cache_positions is not None
            cache_latent = get_pcp_group().all_gather(latent, dim=0)
            cache_positions = swa_metadata.cache_positions
```

## 性能声明

条件：单节点 8× NVIDIA H20（96G, SM90），镜像 `vllm/vllm-openai:nightly-92044241a02f05de51420d654daf669de20b4691`，模型 DeepSeek-V4.1-Flash，`FLASHMLA_SPARSE_DSV41`、`fp8_ds_mla`、block-size 64、MoE marlin、enforce-eager、禁用 prefix caching；SM100 未验证。GSM8K（EvalScope，1319 条、4-shot、batch 64、seed 42、temperature 0）：TP8/PCP1 1281/1319 = 97.1190%、7.36 tok/s；TP4/PCP2 1282/1319 = 97.1948%、7.04 tok/s；TP2/PCP4 1284/1319 = 97.3465%、7.10 tok/s。Prefill（`vllm bench serve`，random prompts、one output token、4 measured waves）：8192 token / concurrency 1 下 input tok/s 6763.52 → 7914.17 (+17.01%) → 8892.92 (+31.48%)，mean TTFT 1210.98 ms → 1034.90 (-14.54%) → 920.97 (-23.95%) ms；32768 / concurrency 8 下 6816.65 → 8135.74 (+19.35%) → 9163.40 (+34.43%)，TTFT 34265.77 → 28708.44 (-16.22%) → 25490.96 (-25.61%) ms。**重要 caveat**：正文明确这些 GPU/模型测量属于前一版 global-order compressor 实现、未在当前 rank-local revision 重跑，「require renewed validation on the target H20 configuration」后才可用于支撑本版本结论。另有 Humming MXFP4 验证（#60447 head `44ea227814`），正文称 Humming 明显低于对应 Marlin 测量、故 launch 命令仍固定 Marlin。

## 验证与回归风险

Rank-local revision 本地验证：101 passed、6 CUDA-only skipped，pre-commit、mypy 3.10/3.12、syntax 与 diff 检查通过；CPU 回归覆盖 PCP2/4 predecessor linkage、replicated short prefills、empty ranks、request isolation、newest open ring rows。Humming 验证中 TP4/PCP2 与 TP2/PCP4 选出 `HUMMING` MXFP4 MoE 与 `HummingMxfp8LinearKernel`、206 个 overlaid runtime 文件全匹配、启动零 Pod restarts 与零 scratch/workspace-lock assertion。质量 caveat：一个 TP4/PCP2 GSM8K 样本达 `max_tokens=65536` 耗时 9701s，对应 TP2/PCP4 样本自然停在 10096 tokens，不应视为 token-identical 或 clean quality-equivalence。未支持范围即潜在影响路径：P/D disaggregation 与 KV connector、ROCm、PP/DP/DCP/microbatching、prefix caching、SWA bounded replay、speculative decoding、CUDA graphs、MegaMoE、多模态、LoRA、其它 attention backends；Engram embeddings 仍 TP-sharded，增大 PCP 可能提高 host 内存需求。

## 关联

PR 正文提及 related #25749、#49109，明确不含 #43809 的 V4 C4/C128 与 MegaMoE 工作；Humming 验证涉及上游 #56997（SM90 优先 Humming）与 #60447（scratch 容量修复）；rank-local revision 采用 wangyicong52/vllm#1（Lucas Wilkinson 著）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
