# PR #63544: [Perf][GLM-5.3-Flash] KDA decode and prefill through FlashInfer

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @suiyoubi |
| 合并 | 2026-11-18 |
| 分类 | Speculative Decoding（tier: major） |
| Label | ready, nvidia, glm |
| Diff | pack 标注 +None/-None, None files；pack 末尾说明「1 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/63544 |

## 一句话结论

把 GLM-5.3-Flash 的 KDA（Kimi Delta Attention）层迁移到 FlashInfer 融合 kernel：decode 每层由 3 个 Triton kernel 合为 1 个 fused kernel，decode 步 1.9–2.1x 提速、混合步 46–61% 提速，GB300 聚合部署 TPS/GPU 最高 +24.9%。

## 动机

GLM-5.3-Flash 有 34 个 KDA 层（64 heads × 128）。当前每个 decode 步每层要跑 3 个 Triton kernel（conv+SiLU、recurrence、gated RMSNorm），且混合步（prefill+decode）的 decode 行走 chunked prefill kernel。Kimi K3 已通过同一套 `vllm.utils.flashinfer` wrapper 与 `kda_decode_backend` / `kda_spec_decode_backend` / `kda_prefill_backend` 选项用上了这些 FlashInfer kernel，本 PR 把同样的路线带给 GLM-5.3-Flash。注意依赖 FlashInfer 0.7.3（flashinfer-ai/flashinfer#7132 提供 16/64 heads 支持与 DS conv-state 布局、#6201 提供 SM107 prefill）。

## 优化点

- **Decode（默认 auto）**：普通 decode 步每层调用一次 `flashinfer_fused_kda_decode`（conv + SiLU 含 conv-state 更新、recurrence、gated RMSNorm 一次完成）；MTP draft-verify 步调用 `flashinfer_packed_fused_kda_decode`；无 spec 行的混合步把排在前面的 decode 行（vLLM 会把 decode 排前）剥给 fused kernel，decode-row peel 避免了 FlashKDA prepare kernel 扫描数百个单 token 序列。静态支持检查 `_is_flashinfer_kda_decode_supported`（SM10x、head_dim 128、conv width 4、bf16 激活与 conv state、fp32/bf16 recurrent state、有界 gate、head 数 8–96 等），`auto` 不满足时回退 Triton，显式 `flashinfer` 则启动报错。
- **Prefill（opt-in `--kda-prefill-backend flashinfer`）**：调用 `recurrent_kda(backend="cute-dsl")`，通过 `ssm_state_indices` 原地读写 recurrent state pool（免 gather/scatter/终态 buffer）；fresh 序列的 slot 先用一次 `index_fill_` 清零；单 token prefill 行退回 recurrent kernel；不产生 prefix-cache checkpoint。
- **Norm 位置**：fused kernel 自带 `o_norm`，`_norm_in_forward` 属性保证 `_forward` 在所有步型下都返回已 norm 的输出，以兼容 breakable CUDA graph 重放。
- **`_int64_cu_seqlens`**：把 int64 偏移按步缓存并供该步所有 KDA 层共享，避免每层各转一次、并让 FlashInfer few-sequence 路线每步只做一次阻塞 host 读。

## 关键改动

```text
vllm/models/glm5next/common/kda.py:
  + from vllm.utils.flashinfer import (flashinfer_fused_kda_decode,
  +     flashinfer_packed_fused_kda_decode, flashinfer_recurrent_kda, ...)
  + def _is_flashinfer_kda_prefill_supported(...)  # SM100/SM103/SM107, D=128, bf16, fp32/bf16 state, bounded gate
  + def _int64_cu_seqlens(metadata, cu_seqlens)    # 每步一次、各层共享的 int64 cu_seqlens
  + def _is_flashinfer_kda_decode_supported(...)   # query_len=1: heads (8,12,16,24,32,48,64,96); packed: SD only, ≤7 draft tokens
  + def _resolve_kda_decode_backend(...)           # auto / flashinfer / triton
  + @property _norm_in_forward: return self.kda_decode_backend == "flashinfer"
  + def process_weights_after_loading(self)        # 构建 FlashInfer 的 _fi_conv_weight / _fi_norm_weight
  + def _flashinfer_decode_rows(...) / _flashinfer_decode(...)
tests/models/glm5next/test_kda_flashinfer.py: 新增 527 行（backend 选择、decode/混合/spec/prefill 对照 Triton 与 fp32 参考、CUDA graph 重放、strided slot indices 等）
tests/models/glm5next/test_kda_flashinfer_decode.py / test_kda_flashinfer_prefill.py: 新增
tests/models/glm5next/test_kda_conv_weight_refit.py / test_kda_recurrent.py: 适配 kda_decode_backend="triton" 与 _forward 签名
```

## 性能声明

- **Decode kernel**（VR200，34 层/步，CUDA graph，L2 flushed）：1–8 rows/GPU 快 1.9–2.1x；32–512 rows 快 1.3–1.8x（H64 fp32 @512 rows：19.5 → 12.4 ms/步；H64 bf16：16.2 → 9.2 ms）。纯 decode 步跑在 FULL CUDA graph（FULL_AND_PIECEWISE，KDA 已捕获）；eager（PIECEWISE）步 host enqueue 3.0 → 0.95 ms/34 层。
- **混合 prefill+decode 步**（单层 H64）：快 46–61%（32K prefill tokens + 512 decode rows：6.96 → 2.72 ms），主要来自 decode-row peel。
- **FlashInfer prefill（opt-in）**：4+ 长序列时每层快 8–12%；eager 步 1 序列 H64 或少数序列 H16 时慢 3–8%，主要来自调用 host 时间（约 95 µs vs FlashKDA 18 µs @VR200），因此保持 opt-in。
- **Serving**（GB300/GB200，native srt-slurm sa-bench 8K/1K，Dynamo，fp8 KV，FlashInfer TRT-LLM NVFP4 MoE）：agg DEP2 GB300 C512/C1024 TPS/GPU 2847 → 3237（+13.7%）/ 2402 → 2999（+24.9%）；GB200 +13.5% / +14.8%（KV 容量受限）；disagg DS 布局经 NIXL C512：GB300 +5.1%、GB200 +6.0%；agg TP4 C1–C32：+1.4–3.0%。GB200 DEP2 TTFT 58 → 42 s（C512）、176 → 145 s（C1024）。
- **GSM8K 40-shot**（1319 题，greedy，6 runs/arm，agg DEP2，VR200）：base 91.85 ± 0.59%，本 PR（decode+prefill on FlashInfer）91.88 ± 0.76%。MTP（2 draft tokens）：base 92.08 vs PR 91.55；draft acceptance 88.6% vs 88.7%。
- **数值**：FlashInfer decode 误差不超过 Triton 路径；conv states 与 Triton 逐位一致；prefill max abs error 4.2e-3 vs FlashKDA 5.1e-3（H64，8×8192 tokens）。

## 验证与回归风险

新增 `tests/models/glm5next/test_kda_flashinfer_decode.py` 与 `test_kda_flashinfer_prefill.py`（backend 解析与回退、8/16/32/64 heads × fp32/bf16 × SD/DS、混合步、spec-verify 行、CUDA-graph 捕获重放、fresh/continued 序列、strided slot indices 等）。实测 VR200（SM107）180 passed / 18 skipped / 0 failed；GB300（SM103）183 passed / 15 skipped / 0 failed，均含全部 50 个 FlashInfer KDA 测试。验证过程发现一个 MTP 部署下的 bug：非 spec slot indices 是 [N, 1+num_spec] block table 的 strided 列，FlashInfer prefill 拒绝，warmup 报 `ValueError: backend='cute-dsl' does not support this recurrent_kda prefill contract`；修复为传连续索引并加单测。回归风险：硬性要求 FlashInfer 0.7.3（#7132、#6201），0.7.2 下静态检查会放过 0.7.2 首次调用才拒绝的配置；FlashInfer prefill 在低并发少序列场景 TTFT 更高（GB300 TP4 C1 143 → 163 ms、C8 603 → 650 ms，host 开销约 190 µs/层 vs FlashKDA 50–56 µs，1–4 序列 kernel 还慢 7–13%），故保持 opt-in；CI 的 `tests/models/glm5next` 跑在 H200（SM90），SM10x 覆盖需要 Blackwell lane；未包含更快的 prefill conv、64-row norm tiles、流水化 bf16-state decode kernel。

## 关联

- 依赖 flashinfer-ai/flashinfer#7132（16/64 heads 与 DS conv-state 布局）与 flashinfer-ai/flashinfer#6201（SM107 prefill）；需要 FlashInfer pin 升级到 0.7.3 的单独 PR。
- Kimi K3 走同样的 wrapper 与后端选项；upstream 相关 PR：vllm-project/vllm#61614（更快的 prefill conv）、#66512 未出现于此 PR；pack 未提供其他 vLLM issue/PR 关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
