# PR #67133: [ROCm][Attention] AITER FA: dedicated split-KV kernels for decode and speculative-decode verify (GQA 8, head 64, FP8 KV)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mawong-amd |
| 合并 | 2026-11-19 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, performance, rocm |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数);diff 新增 vllm/v1/attention/ops/flydsl_kernels/gqa_verify.py(1014 行)、tests/v1/attention/test_rocm_gqa_verify.py、benchmarks/kernels/benchmark_rocm_gqa_verify.py,并改 vllm/v1/attention/backends/rocm_aiter_fa.py、docs/design/attention_backends.md 等 |
| 链接 | https://github.com/vllm-project/vllm/pull/67133 |

## 一句话结论

在 gfx950(MI355X)为 `ROCM_AITER_FA` 增加专用 FlyDSL split-KV kernel(head 64、GQA 8、FP8 KV),覆盖 plain decode 与 1–7 draft token 的 verify:40 请求混合长度满注意力层 verify 单层 1945 → 212 µs,AgentX c40 吞吐 +99%。

## 动机

有 k 个 draft token 时每个 decode 请求要验证 k+1 个 query;`ROCM_AITER_FA` 的这些 batch 走 AITER `unified_attention`,长上下文下主导 decode 步;plain decode(`paged_attention_v1`)则要等最长的请求。adapt 自 sgl-project/sglang#47371:满注意力层用长度均衡网格(时间跟随总 KV 而非最长请求,前提是每 CU 不足两个请求),metadata builder 每步映射一次工作;滑窗层用窗口掩码加每请求等分。

## 优化点

1. FlyDSL kernel(`vllm/v1/attention/ops/rocm_gqa_verify.py`,入口 `fast_verify`):BF16 Q、FP8 e4m3 分页 KV `[pages, 16 or 32, kv_heads, 64]`,split-KV FP32 部分和;QK/PV 用 BF16 MFMA(K/V 从 FP8 精确上转),转置形式 `S^T = K Q^T`、`O^T = V^T P^T`,softmax 沿 key 行做 lazy rescale(2^8 阈值,FP32 累加精确),V 经 wave 私有 LDS,XOR swizzle 避免 bank conflict;页 id 提前一 chunk group 加载;池超过 2^31 字节走 64-bit 寻址。
2. Builder 侧:`_uses_fast_verify` 门(gfx950 + FP8 e4m3 默认布局 + 非 shuffle);满注意力层的 `verify_map` 每步一次构建并共享;滑窗层 kernel 自行应用窗口,不再需要窗口裁剪的 block table。
3. 覆盖范围之外(其他 batch/层/模型)保持既有路径;依赖 FlyDSL,缺则回退并警告。

## 关键改动

```python
# vllm/v1/attention/backends/rocm_aiter_fa.py(forward 钩子)
+            if (
+                self._fast_verify
+                and attn_metadata.causal
+                and rocm_gqa_verify.fast_verify(
+                    query, key_cache, value_cache, output, attn_metadata,
+                    num_decodes, num_decode_tokens, decode_max_query_len,
+                    self.scale, self.sliding_window[0] + 1, layer,
+                    attn_metadata.decode_metadata.verify_map,
+                )
+            ):
+                return output
```

```python
# 新 kernel 头部注释(FlyDSL,adapted from sglang#47371)
+"""Speculative-decode verify (full or sliding-window attention) on gfx950 in FlyDSL.
+... Transposed form per wave: S^T = K Q^T (keys x query columns) and O^T = V^T P^T on
+16x16x32 BF16 MFMAs. ..."""
```

## 性能声明

数字均照抄 PR 正文(未独立核实)。平台:MI355X, Muse Spark TEP4, FP8 KV, default block size。

- 单层 vs `unified_attention`(32-token kernel blocks,graph replay,µs):24×100k:满注意力 q_len 8:780 → 261;q_len 4:762 → 224;滑窗 q_len 8:73 → 23。40×100k:1479 → 412;1198 → 360;84 → 27。40 混合(200k..3k):1945 → 212;1434 → 181;84 → 27。
- Decode step p50,k=7,D 请求 @100k context(+:其中两个 500k),ms:main 25.8 / 33.5 / 45.4 / 53.4 / 68.9 / 48.0(8+) / 96.1(24+);this PR 21.1 / 25.8 / 33.4 / 41.5 / 46.2 / 27.5 / 42.7(D=4/8/16/24/32/8+/24+)。
- Plain decode(无 spec decode)单层 vs `paged_attention_v1`:1.1–1.5x at 1–128 × 100k、混合长度 1.7–3.2x(满注意力),滑窗层 1.2–1.4x;decode step p50:D=1 10.6 → 10.3;D=8 17.2 → 16.1;D=24 26.1 → 23.0;D=40 33.8 → 28.9;D=64 43.2 → 35.3;24+ 29.4 → 24.3 ms。
- Serving(Muse Spark Open,TEP4 4× MI355X,DSpark k=7 synthetic acceptance 4.05):output tok/s 532.0 → 572.7(**+7.6%**);TTFT p50 6194 → 5850 ms(p90 25424 → 25291);ITL p50 40.84 → 32.60 ms。
- AgentX(SemiAnalysis InferenceX trace,3600 s;两 run 均值,GPU 半区互换):c10 13,123 → 14,796 tok/s/GPU @ ITL p50 7.44 → 5.04 ms(**+13%**);c24 20,522 → 31,637 @ 20.95 → 7.41(**+54%**);c40 20,442 → 40,685 @ 41.48 → 11.28(**+99%**)。
- Accuracy:对 float32 的相对 L2 每 q_len 1–8 均为 2.3–2.4e-3(`paged_attention_v1` 用 FP8 cache 时 QK/PV 走 FP8 MFMA,为 2.7–3.2e-2);greedy MATH-500 0.976(单 pass;前一版 20 passes:0.9718 main vs 0.9758,paired p=0.12),真实 acceptance length 4.35(前一版 4.27);无 spec decode 6 paired passes 0.9697 vs 0.9703,p=0.93;Needle(48 prompts, ~80k)48/48。

## 验证与回归风险

- `pytest tests/v1/attention/test_rocm_gqa_verify.py`:gfx950 上 725 passed, 1 skipped(FlyDSL 缺失用例),单卡 5–7 分钟;覆盖 q_len 1–8、16/32 页、每 rank 1–16 KV head(窗口边界、零长 padding、窗口外陈旧 NaN、2^31 字节以上池、graph replay 后新 seq_lens)对 float32 参考的对比,以及 plain-decode 与 verify batch 经 backend(gate、builder map、forward hook)的路径和"kernel 不覆盖的 batch 不碰输出"。
- 回归风险/门条件:head size 64、GQA 8、每 rank ≤16 KV head、无 sinks/soft cap/ALiBi、FP8 e4m3 KV 默认布局、所有 decode 请求相同 q_len(1–8)、满注意力每 CU 少于 2 请求——gpt-oss 形状符合但用 sinks 故不走;每 q_len/页大小/网格一次 FlyDSL 编译,eager batch 超出最大捕获尺寸时服务期编译一次并缓存;依赖 flydsl 0.4.2 + aiter 0.1.28.post1(vLLM ROCm 镜像 pin 的 aiter v0.1.29 也要求 flydsl 0.4.2);与 drafts #66741/#66743 相比满注意力 verify 1.2–1.5x 更快(q_len 8),其 ragged q_len 思路互补。

## 关联

adapted from sgl-project/sglang#47371;相关 drafts #66741 / #66743(@sawsa307,依赖未发布的 AITER 变更);后续方向:其他 ROCm backends;文档 `docs/design/attention_backends.md` 新增 "ROCM_AITER_FA GQA decode kernel" 一节。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
