# PR #67659: [Perf][Hy4] Sequence parallelism for iHC

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @qianlihuang |
| 合并 | 2026-11-17 |
| 分类 | Parallelism / Communication(tier: major) |
| Label | documentation, ready, ci/build |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/67659 |

## 一句话结论

给 HY4 的 iHC(interleaved hybrid connection)路径应用序列并行:decoder 层间保持 iHC 残差按 token 分片,attention/MLP 前后用 all-gather / reduce-scatter,消除 TP 各 rank 上的冗余计算——8xH200 TP8/DP1/EP8 下 prefill 吞吐提升 15.04% / 27.18% / 26.84%(ISL 2048/8192/32768),8K 步的 profiled iHC GPU 时间 193.600 → 24.366 ms/步。

## 动机

HY4 iHC 与归一化在 TP 各 rank 上重复计算相同的全局 token;当步长足够大(≥1024 token 行)时,把 iHC 行分片到 rank、只在进入 attention/MLP 时 gather,可大幅省掉 per-rank 的 iHC 前后处理与 norm 计算。复用 #66549 引入的序列并行基础设施;模型按自身 eligibility 与 token 阈值独立选择 iHC SP,与 attention ShardQ 解耦。

## 优化点

- Decoder 层内:iHC 残差保持 token 分片;每个 subblock 先 `sp_all_gather` 归一化后的 hidden 输入,跑既有 attention/FFN 路径,输出 `sp_reduce_scatter` 后再做本地 post-gating。`defer_reduction` 时 o_proj/MLP 构造传 `reduce_results=False`,非 SP 路径退回 `tensor_model_parallel_all_reduce`。
- 门槛:需 iHC、TP>1、DP1/PP1/PCP1/DCP1、`all2all_backend != "passthrough"`、无 LoRA、非 batch-invariant、speculation 不用 aux hidden states;每步至少 1024 model token 行(`_IHC_SP_MIN_TOKENS = 1024`)才分片,短批次用 replicated 行 + deferred all-reduce。
- 最终 norm 后 gather 回全量行以支持 MTP;MoE 构造断言 factory 通过 `skip_final_all_reduce` 遵守 deferred output reduction。
- eligibility 与 attention ShardQ 独立(iHC SP 也可在 ShardQ 关闭/不合格时生效)。

## 关键改动

```python
# vllm/models/hy_v4/nvidia/model.py
# Smallest step that shards the iHC rows; shorter ones keep them replicated.
_IHC_SP_MIN_TOKENS = 1024

def _forward_ihc(self, positions, hidden_states):
    full_num_tokens = positions.shape[0]
    use_ihc_sequence_parallel = hidden_states.shape[0] < full_num_tokens
    hidden_states = self.hc_attn_layer.prepare_input(hidden_states)
    hidden_states, post_gates, residual = self.hc_attn_layer.pre(hidden_states)
    hidden_states = self.input_layernorm(hidden_states)
    if use_ihc_sequence_parallel:
        hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
    hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
    if self.defer_reduction:
        hidden_states = (
            sp_reduce_scatter(hidden_states)
            if use_ihc_sequence_parallel
            else tensor_model_parallel_all_reduce(hidden_states)
        )
    hidden_states = self.hc_attn_layer.post(hidden_states, residual, post_gates)
```

```diff
 # vllm/models/hy_v4/nvidia/kda.py 同型改动也适用于 mlp/attention 构造
         self.self_attn = ...(
             prefix=f"{prefix}.self_attn",
             layer_idx=layer_idx,
             topk_indices_buffer=topk_indices_buffer,
+            reduce_results=not self.defer_reduction,
         )
```

## 性能声明

- 8xH200 TP8/DP1/EP8(Hy4-preview-FP8 ModelOpt MXFP8 权重、Humming BF16 激活、FP32 iHC 参数、`fp8_ds_mla` KV、native MTP3;OSL1、request-rate infinity):prefill 吞吐 +15.04% / +27.18% / +26.84%,median TTFT -12.06% / -21.40% / -20.99%(ISL 2048/8192/32768;A/B/A 串行,每臂 3 个 seed,如 ISL2048 3.4642 → 3.9846 → 3.4632 req/s,baseline drift -0.031%)。
- Profile(8K 步,5 步覆盖,78 target 层 + 1 MTP 层,每 rank 1024 目标 iHC 行):iHC pre stage 1 137.184 → 17.695 ms、pre stage 2 21.503 → 2.394 ms、post 34.913 → 4.277 ms;Communication 127.334 → 146.762 ms;GEMMs 212.147 → 212.472 ms;FlashMLA sparse attention 67.015 → 66.979 ms;GPU busy interval 729.533 → 571.312 ms。通信调用每步从 161 升到 318(subblock 边界用 all-gather/reduce-scatter)。
- GSM8K(1319 题 5-shot,temperature 0,seed 42):main 0.89613(invalid 0.00379, 87.311 s)vs SP 0.89234(invalid 0.00303, 83.910 s)。

## 验证与回归风险

- 回归:`tests/models/hy_v4/test_sequence_parallel.py`(新)+ `test_shardq_sinks.py` 在 `f2cead2f46` 上 5 passed(368.91 s),覆盖 TP2 真实生成(eager 与默认 graph)、3073-token prompt 与奇数 1025-token chunk、3 个 sink reload。初始 repro 在 main 上以 `('iHC token rows', 255, 128, 255)` 失败;旧 head `fb2c086ca8`(ShardQ 门控 iHC SP)上 ShardQ-disabled 分支仍失败,证明解耦必要。
- 风险(正文明示):吞吐/profile/GSM8K 未在解耦后的 head `f2cead2f46` 上重跑;验证仅覆盖 H200 + FP8 KV;Blackwell 与 NVFP4-KV 验证 pending;TP2 场景吞吐未测。

## 关联

- 复用 #66549 的序列并行调度模式;#67490 在被测配置中启用 sink-aware ShardQ;相关 QSA/iHC 上游文件在 `.buildkite/test_areas/distributed.yaml` 中新增了本 PR 的两个源文件依赖与测试。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
