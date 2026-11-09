# PR #30113: [KDA] Add FlashInfer SM100 KDA decode + MTP (target_verify) backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-07-15 |
| 分类 | Speculative Decoding |
| Label | speculative-decoding, run-ci, flashinfer, run-ci-extra, linear-attention, release-highlight |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 全文含 4 个源码文件,另有 2 个文件因超长被丢弃 |
| 链接 | https://github.com/sgl-project/sglang/pull/30113 |

## 一句话结论

给 KDA 线性注意力补上 FlashInfer SM100 decode 内核与首个数值正确的 MTP/target_verify 路径,使 KDA 模型首次可用投机解码。

## 动机

KDA(Kimi Delta Attention)此前完全没有 `target_verify` 路径,导致 Kimi-Linear、ling-v3 等 KDA 模型无法使用任何投机解码(EAGLE / MTP / n-gram)。GDN 已具备完整的 decode + prefill + MTP 支持,本 PR 通过封装 FlashInfer 的 `recurrent_kda` 单一算子,把 KDA 在 Blackwell(SM100)上的 decode + MTP 能力补齐到与 GDN 对等。Prefill 仍留在 Triton / CuTe DSL,因为 FlashInfer 没有 KDA chunk-prefill 内核。

## 优化点

新增 `FlashInferKDAKernel` 封装 `flashinfer.kda_decode.recurrent_kda`(CuTe DSL,仅 SM100),实现 `decode`(原地更新 SSM pool)与 `target_verify`(topk=1)。decode 走 in-kernel `cu_seqlens` 路径,由内核直接按 `ssm_state_indices` 索引 committed pool,避免 python 侧 gather/scatter(该写法在 B=64 约 141us 开销)。verify 侧把每个 draft token 的 post-state 写入 `intermediate_ssm` scratch(slot = request*T + step),第 0 列先从 committed state 播种,复用现有中心化 rollback `update_mamba_state_after_mtp_verify`。网关契约对齐 Triton 参考:softplus gate `g = -exp(A_log)*softplus(a+dt_bias)`(`use_gate_in_kernel=True`)、beta 传 `sigmoid(b)`、状态布局统一 `[N, HV, V, K]` 无需转置。另外修正 conv-window dedup:KDA 的 conv-state 布局为 `[pool, width, dim]`,dedup 的 `as_strided` 重叠视图在 KDA 的 transpose 后会把窗口写错物理列导致状态损坏,故 `conv_window_ded_enabled(..., is_kda)` 对 KDA 保持 dense 布局。KDA MTP 仅派发到 FlashInfer;Triton / CuTe DSL decode 后端 `verify_kernel = None`,启用投机解码时明确报错。EAGLE tree verify(topk>1)在 setup 阶段提前拒绝。

## 关键改动

```text
python/sglang/srt/layers/attention/linear/kernels/kda_flashinfer.py: 新增 FlashInfer recurrent_kda 封装(decode + target_verify,SM100 guard,懒加载)
python/sglang/srt/layers/attention/linear/kda_backend.py: dispatcher 增加 is_flashinfer() decode 分支与 target_verify() 方法;KDAAttnBackend 增加 _forward_target_verify(conv1d 中间窗口 checkpoint + rollback)
python/sglang/srt/layers/attention/linear/kernels/kda_triton.py: TritonKDAKernel.target_verify(fused_sigmoid_gating_delta_rule_update, is_kda=True, disable_state_update=True),作为单测参考
python/sglang/srt/mem_cache/memory_pool.py: conv_window_dedup_enabled 增加 is_kda guard,KDA 保持 dense conv-window
test/registered/attention/test_kda_decode_flashinfer.py: 11 个单测(decode/verify 对 Triton 参考、checkpoint 状态对 sequential-decode 参考)
benchmark/bench_linear_attention/bench_kda_flashinfer_mtp.py: FlashInfer-vs-Triton 正确性与逐调用延迟
```

## 性能声明

- ncu(B=64)下 `recurrent_kda` decode 内核 29.6 us,vs Triton `fused_sigmoid_gating_delta_rule_update` 36.8 us。
- CUDA-graph 纯 GPU decode:FlashInfer 大 batch 胜出、小 batch 更慢——B=1 0.67x、B=64 0.93x、B=128 ~1.0x、B=256 1.35x。
- 端到端正确性(B200, SM100, Kimi-Linear-48B-A3B-Instruct, TP2, gsm8k 200 题 5-shot, n-gram 投机解码):non-spec 参考 0.900,本 PR FlashInfer decode+verify 0.895,Invalid 0.000;conv dedup 时 gsm8k 仅 0.03(dense conv 修复后 0.895,对齐 0.90 参考)。
- gsm8k + n-gram accept rate 低(accept_len ~1.5),MTP 输出吞吐 ~625 tok/s ≈ non-spec ~667 tok/s;PR 明确不在 gsm8k 上宣称 MTP 加速,MTP 收益取决于高 accept rate 的重复/结构化 workload。
- eager 微基准(K=V=128, H=HV=16, bf16):decode 约 0.89-0.93x(Triton 更快);target_verify(T=8)0.53-0.59x(FlashInfer 更慢)。

## 验证与回归风险

新增 11 个单测 `test_kda_decode_flashinfer.py`:FlashInfer decode/verify 与 Triton 参考逐 token 对齐(output tol ~5e-2),且 `target_verify` 每步 checkpoint 状态与 sequential-decode 参考一致(rollback 正确性依赖此属性);需 SM100 + 支持 `recurrent_kda` 的 FlashInfer,否则跳过。风险面:KDA conv-window 布局改为 dense 会影响 KDA 模型的显存占用;decode 后端混用(premature dispatch 到非 FlashInfer 后端)会触发清晰报错;ReplaySSM 对 KDA 保持接线但不推荐。Triton verify 的 per-draft-token 输出正确但端到端 rollback 尚不对,未接入生产路径。

## 关联

同区域:PR #26888(KDA target_verify 投机解码支持)、PR #33722(FlashInfer KDA MTP verify 的 fused-accept state advance)。GDN 对应能力见 PR #20627(packed decode,本 PR 注释中引用)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
