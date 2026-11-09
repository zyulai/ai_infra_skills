# PR #48484: Replicated embedding and norm fusion for DSV3 flat model

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jeejeelee |
| 合并 | 2026-08-18 |
| 分类 | Speculative Decoding(tier: major) |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/48484 |

## 一句话结论

DSV3 flat model 用复制 embedding + 融合 gather/norm 去掉 all-reduce，GLM-5.2-NVFP4 单并发吞吐 +5%。

## 动机

正文 Purpose 与新增 `fused_embed_norm.py` 的 docstring 说明：vocab-parallel embedding 每步需要一次 all-reduce，而这次 all-reduce 的代价与每步携带多少 token 无关——每步 token 越少，它在 step time 里占比越大。把整张 embedding 表复制到每个 rank（`VLLM_REPLICATE_EMBED=1`）后，lookup 变成无 mask、无 all-reduce 的本地 gather，从而解锁融合的 gather+norm 路径；代价是 TP>1 时每 rank 一份完整表。

## 优化点

(1) `make_input_embedding` 在 `VLLM_REPLICATE_EMBED=1` 时用 `disable_tp` 构建 `VocabParallelEmbedding`，全表驻留每 rank、本地 gather、无 all-reduce；TP>1 且 `tie_word_embeddings` 时直接断言拒绝（复制表无法与 vocab-parallel `ParallelLMHead` 绑定），TP=1 时 `disable_tp` 是 no-op。(2) 新增 `vllm/model_executor/layers/fused_embed_norm.py`（231 行，自包含、不 import model-local 代码），提供两个 Triton 融合：`fused_embed_norm`（gather + 链式 RMSNorm，如首层 `input_layernorm`）与 `fused_embed_eh_norm`（gather + pos-0 清零 + enorm/hnorm + cat，供 MTP/eagle depth layer 的 embed/previous-hidden 输入 norm）。(3) 在 `vllm/models/deepseek_v32/nvidia/model.py` 与 `mtp.py` 接入该路径。(4) 正文称融合数值中性，两臂 MTP acceptance length 均为 4.4–4.7。

## 关键改动

```text
vllm/model_executor/layers/fused_embed_norm.py: 新增（231 行）复制 embedding 工厂 + fused_embed_norm / fused_embed_eh_norm 两个 Triton 融合
vllm/envs.py: VLLM_REPLICATE_EMBED 开关
vllm/models/deepseek_v32/nvidia/model.py: DSV3 flat model 接入复制 embedding 与融合 norm
vllm/models/deepseek_v32/nvidia/mtp.py: MTP depth layer 的 fused_eh_norm 对应路径
tests/kernels/core/test_fused_embed_norm.py: 融合 kernel 测试
```

关键片段（Triton RMSNorm 融合）：

```diff
+@triton.jit
+def _rms_norm(x, w, eps, HIDDEN_SIZE: tl.constexpr):
+    x = x.to(tl.float32)
+    mean_sq = tl.sum(x * x, axis=0) / HIDDEN_SIZE
+    rrms = tl.rsqrt(mean_sq + eps)
+    w = w.to(tl.float32)
+    return (x * rrms) * w
```

## 性能声明

Config：GLM-5.2-NVFP4 / 8×B300 TP8 / MTP(5)；唯一变量 `VLLM_REPLICATE_EMBED=1`（复制 embedding + fused gather/norm）vs `=0`（vocab-parallel embedding + all-reduce）。Workload：speed-bench `throughput_16k`、low_entropy、input ≤10240 / output 1536 / `--ignore-eos` / temp 0；数字为 2 次独立完整重复的均值。conc 1：ITL 10.01 vs 10.80 ms（-7.25%）、TPOT 2.14 vs 2.28 ms（-5.97%）、out tok/s 405.1 vs 384.1（+5.45%）；conc 8：-1.04% / -0.97% / +1.29%；conc 16：-1.38% / -0.86% / +0.79%；conc 32：-0.79% / -0.47% / +0.61%；conc 64：-0.73% / -0.36% / +0.27%。正文 Bottom line：「-7% ITL and +5% throughput at concurrency 1; decays to ~1% by conc 8 and to nothing by conc 64」；No TTFT difference（单臂内 run-to-run 漂移达 ±5% 且每个并发点符号翻转）；No accuracy impact（MTP acceptance length 两臂 4.4–4.7）。噪声地板：ITL 0.25%、TPOT 2.8%、throughput 2.4%、TTFT 5.5%，据此 conc≥8 的约 1% 方向一致但接近噪声。

## 验证与回归风险

Accuracy comparison（GSM8K、5-shot、num_concurrent 256）：本 PR flexible-extract / strict-match 为 0.9340±0.0068 / 0.9348±0.0068，main 为 0.9378±0.0067 / 0.9348±0.0068，正文据此称数值中性；diff 含 `tests/kernels/core/test_fused_embed_norm.py`。风险：TP>1 时每 rank 一份完整 embedding 表（内存代价），且 tied word embeddings 在 TP>1 被拒；收益随并发升高衰减，conc 64 归零，conc≥8 的增益落在噪声地板内；仅在 GLM-5.2-NVFP4 / 8×B300 TP8 / MTP(5) 上测量，正文未给其他模型、硬件或精度的数据。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
