# PR #63638: [Perf][Kimi K3] Opt-in KV-balanced FP8 MLA decode for agentic workloads

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wzhao18 |
| 合并 | 2026-10-26 |
| 分类 | Speculative Decoding \| tier: major |
| Label | ready, kimi, k3 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明 5 file(s) dropped from diff for size (see PR for full diff). |
| 链接 | https://github.com/vllm-project/vllm/pull/63638 |

## 一句话结论

新增 opt-in KV-balanced FP8 MLA decode，GB300 AgentX decode 提速 1.34-1.59x。

## 动机

正文指出 agentic decode 的一批请求混合了差异极大的上下文长度（如 5K、20K、98K、300K tokens）。为把一个请求的 attention 摊到 GPU 上，decode 会把 KV 切成 splits，由 clusters 并行处理再 reduce。现有 FP8 MLA decode `TOKENSPEED_MLA` 对整个 batch 只用同一个 split 数（由 batch size 与 SM 数决定，以 `tokenspeed_mla_min_split_kv` 为下限）：短请求被切得过碎、每个 split 都要付 setup 与 reduction 开销；长请求 splits 太少、尾部少量 clusters 要流式处理数十万 keys 而其余 GPU 空闲。`min_split_kv=8` 帮长请求却伤短请求，DCP8 N=128 时甚至慢于默认。

## 优化点

新增 opt-in backend `--attention-backend KIMI_K3_KV_BALANCED_MLA`，decode 按各请求自身 KV 长度切分：Triton planner 给每个请求 1~8 个、每个至少 1024 tokens 的 splits，填满最多 8 个 waves，并把 (query tile, request, split) 写成 device work list（不读 host、可入 CUDA graph、每 forward 一次、各 MLA 层复用）；一波 persistent 2-CTA clusters 走 work list，使每个 cluster 拿到相近 KV 量；单 split 的请求直接写最终输出、跳过 merge；长 split 的概率可能为 E4M3 flush 成 0，故 P 按 2^6 缩放再由 LSE 抵消。基于 `MLACommonBackend` / `MLACommonImpl` / `MLACommonMetadataBuilder` 的独立 backend，不依赖 `TOKENSPEED_MLA`，Kimi-K3 的 mla.py 不变，prefill 仍用配置的 MLA prefill backend。

## 关键改动

```text
vllm/models/kimi_k3/nvidia/kv_balanced_mla/backend.py: KimiK3KvBalancedMLABackend / MetadataBuilder / Impl，forward_mqa 运行 kernel
vllm/models/kimi_k3/nvidia/kv_balanced_mla/core.py: planner 与编译后的 kernel
tests/models/kimi_k3/test_kv_balanced_mla.py: 12 个单测，对照 FP32 参考
vllm/models/kimi_k3/nvidia/kv_balanced_mla/__init__.py: backend 包导出
```

## 性能声明

Decode attention（GB300，AgentX context lengths）：相对 `TOKENSPEED_MLA` + `tokenspeed_mla_min_split_kv=8` 快 1.34–1.59x，相对其 default 快 1.91–2.67x（batch size 几何平均，DEP16 与 DCP8、target verify 与 draft）。Microbenchmark（单 rank 每 decode step，CUDA-graph replay，5 个随机 batch 均值，长度取自 AgentX trace，median 约 98K、p90 约 284K tokens）：DEP16 verify kvb vs ts1 2.63x、vs ts8 1.37x；DEP16 draft 2.67x / 1.34x；DCP8 verify 1.91x / 1.52x；DCP8 draft 2.46x / 1.59x。端到端（Kimi-K3 MXFP4，GB300 NVL72，InferenceX AgentX replay + Dynamo 分离式服务 + Mooncake KV offload + DSpark 4-token 投机解码，baseline 为 `TOKENSPEED_MLA` min_split_kv=8）：DEP16 (3P1D c336) TPS/GPU 27,267 → 29,402（+7.8%）、p90 interactivity +20%；DCP8 (3P4D c240) 20,421 → 21,370（+4.6%）、+10%；prefill-bound 2P3D c240 +2.7% / +13%。Accuracy：gsm8k 5-shot、真实投机拒绝下 DEP16 0.961、DCP8 0.967。

## 验证与回归风险

`test_kv_balanced_mla.py` 12 个测试在 GB300 通过，覆盖不等长与空请求、DSpark 非因果 draft block、DCP8 rank 0 与 rank 5（interleave 64，传错 rank/interleave 误差约 80% vs 正确时 2%）、无 P prescale 会失败的 peaked softmax、plan+decode 的 CUDA graph 重放、大于 2 GiB 的共享 workspace；backend 测试校验该 backend 不在任何平台 priority list、会拒绝 FP16 模型 / 非 FP8 KV cache / 非 SM10x GPU / kv_lora_rank≠512。CI：B200 Kimi K3 step 运行新测试，但测试在 GB300（SM103）验证、未在 B200（SM100）跑；MI355 step 跳过。风险与限制：vendored kernel 约 4.9K 行，改用 `tokenspeed_mla` 包复用会让 speedup 少约 20%；split-KV scratch 约 0.5 GB（128 请求、heads=96、q=5），高于 TokenSpeed 的约 0.23 GB；无 fallback decode，kernel 跑不了的 step 会抛 ValueError；仅支持 FP8 E4M3 KV cache、SM10x、BF16 模型与特定 MLA 维度。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
