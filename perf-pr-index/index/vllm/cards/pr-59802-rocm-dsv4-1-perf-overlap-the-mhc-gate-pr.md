# PR #59802: [ROCm][DSv4.1][Perf] Overlap the mHC gate projection with attention and FFN

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @cpersson-amd |
| 合并 | 2026-11-04 |
| 分类 | Overlap / Scheduler(major) |
| Label | rocm, ready, deepseek, DSv4.1 |
| Diff | 素材未提供 diffstat 数字;pack diff 收录 4 个文件(新 Triton kernel、mHC overlap 模块、decoder 模型、测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/59802 |

## 一句话结论

ROCm 上把 DeepSeek-V4.1 mHC seam 的 gate projection 挪到侧流与 attention/FFN 重叠,默认流 seam 从约 10 µs 降到 4.3 µs,MI355X E2E 吞吐 +0.3%~5.4%。

## 动机

DeepSeek-V4.1 的每个 mHC seam 在 ROCm 上默认把整套 fused AITER kernel pair(post block、gate projection、Sinkhorn、带 RMSNorm 的 collapse)串行跑在默认流上,下一个 sublayer 必须等它结束。DSv4.1 共 43 层 × 2 = 86 个 seam,decode 步内仅此一项就累积约 0.5 ms 的串行延迟。

## 优化点

把 seam 拆成两半:新的 Triton kernel `mhc_post_collapse_rms_norm_triton` 留在默认流,只做 post block + collapse + RMSNorm,且与 AITER fused kernel 保持相同表达式顺序,新 residual bit-identical;gate projection 连同 Sinkhorn 等走 AITER 原 fused kernel(`HAS_POST=False`)放到 aux 侧流,与接下来的 attention/FFN 计算重叠。fork/join 用事件实现,在 FULL HIP graph capture 内成为 graph 边,无 host 同步;`record_stream` 保证跨流张量存活。触发条件收得很紧:仅 FULL graph capture、`0 < tokens ≤ MHC_OVERLAP_MAX_TOKENS(128)`、gfx950(CDNA4)、AITER fused delayed seam 存在、`hc_mult==4`、无 ubatching;prefill、piecewise graph、eager、CUDA、DSpark draft 层均保持原路径。layer 0 的 broadcast seam 与 Engram seam 走不同路径不变。

## 关键改动

```text
vllm/model_executor/kernels/mhc/triton.py: 新增 mhc_post_collapse_rms_norm_triton(默认流侧 post+collapse+RMSNorm)
vllm/models/deepseek_v41/amd/mhc.py: 新增 MHCOverlapStream(fork/join 事件)与 mhc_pre_delayed_overlap 调度
vllm/models/deepseek_v41/amd/model.py: decoder 层在 FULL graph capture 内启用 overlap,attention/FFN 后 join
tests/kernels/mhc/test_mhc_kernels.py: 新增 overlap 与 fused seam 的 bit-exact 对照测试
```

关键片段(mhc.py):

```python
residual, layer_input = mhc_post_collapse_rms_norm_triton(...)
overlap.fork()
with torch.cuda.stream(stream):
    _, post_mix, comb_mix, _, next_pre_mix = pre_delayed(...)
for tensor in (residual, pre_mix):
    tensor.record_stream(stream)
```

## 性能声明

- Kernel(MI355X,HIP-graph replay,每 seam):fused 9.91/10.24/10.41 µs(1/4/16 tokens)→ 默认流 input kernel 4.15/4.26/4.37 µs,侧流 gates 8.48–9.17 µs 与后续计算重叠;86 个 seam 合计每 decode 步省约 0.5 ms。
- E2E(DeepSeek-V4.1-Flash,MI355X,TP4,EP,FP8 KV,12 个 sweep 点):吞吐全部上升、TPOT 11/12 点下降,均值 TPOT −2.05%、吞吐 +2.00%;单点范围 +0.3%~+5.4%(8192/1024 MC4:466.71→488.68 tok/s,+4.7%,TPOT −4.8%)。
- TP2/TP4/TP8 × 并发 1–128(random 2048/256):所有点均提升,并发 1 时约 +5.4%,并发 96–128 收窄到 +0.3–2%(约 1% 运行间噪声内)。
- 精度:GSM8K 5-shot 全量 split 0.9227→0.9234(flex),噪声内;residual 与 gates bit-identical,layer input 在 16 tokens 内 bit-identical、以上一个 bf16 ulp 内。

## 验证与回归风险

新增 `test_deepseek_v41_rocm_mhc_overlap_matches_fused_seam`(1/2/7/16/64/128 tokens):与 fused AITER seam 对照,residual/gates 要求 atol=0,layer input 允许 1 ulp,并跨 3 次 HIP-graph replay 复检;另有 decoder 级测试确认 eager/piecewise 不触发 overlap、FULL graph 按 seam 数 fork。CI 在 `:amd: (MI355) mHC Kernels` 运行。风险面:仅在 FULL HIP graph 且 ≤128 tokens 生效(HIP event 运行时等待不可靠,故不覆盖 eager);依赖 AITER fused delayed seam 与 CDNA4;128 token 上限是已测最大点(FULL graph 可到 512),更大 batch 收益已收窄。

## 关联

CUDA 对应版本 #57603(本 PR 沿用其结构、gating 与 join 点,并把 token 上限从 16 放宽到 128);#56853(attention 侧 aux stream,当时 mHC seam 仍串行)、#58655(引入被本 PR 拆分的 fused AITER delayed seam)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
