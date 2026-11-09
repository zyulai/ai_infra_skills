# PR #29677: [AMD] perf: compact Triton extend-attention for ragged prefill (AMD/HIP-only)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @valechen |
| 合并 | 2026-08-06 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, high priority, amd, deepseek, npu, run-ci, jit-kernel, bypass-fastfail |
| Diff | diffstat 缺失(pack 未提供 +add/-del 行数与文件数;diff 内容可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/29677 |

## 一句话结论

Triton extend-attention 新增 AMD/HIP-only 的 compact query-tile grid,消除 ragged mixed-prefill 下矩形 grid 的浪费:固定负载吞吐最高 +314%,open-loop 吞吐 +155%。

## 动机

Triton extend kernel 的 launch grid 是矩形、按 `max_len_extend` 定尺寸的 `(batch_size, head_num, cdiv(max_len_extend, BLOCK_M))`:混合 batch 中每条短 decode/partial-prefill 行都要付出与最长 extend 行成比例的 tile 工作量。flash-attention varlen kernel(aiTER 后端经 `flash_attn_varlen_func`/`mha_batch_prefill_func` 调用)本来就按 cu_seqlens 做 ragged-aware launch,工作量与 per-request query tile 之和成比例,不付这笔税;只有 Triton extend kernel 用字面矩形 grid。PR 明确自我定位:这不是新技巧,而是把 Triton extend kernel 补齐到与 flash-attn ragged-aware launch 的对等水平,对 aiter 后端无影响也不需要。ragged mixed-prefill 是生产真实负载(不同 prompt 长度混批,mixed chunking 把 decode 行与长 prefill chunk 放进同一次 attention 调用)。

## 优化点

- compact grid:`(compact_q_tiles, head_num)`,其中 `compact_q_tiles = sum_i ceil(extend_len_i / BLOCK_M)`,只 launch 实际 per-request query tiles;kernel 内 `USE_COMPACT_TILE_GRID` 分支通过顺序扫描 `qo_indptr` 把一维 output_tile 映射回 `(cur_seq, cur_block_m)`。`_compact_extend_q_tiles_per_head` 在 compacting 无收益时(如 uniform batch、batch_size≤1)返回 None 回退 legacy grid。
- 平台与开关:compact grid 与 scheduler tile-budget 均以 `_is_hip`/`_IS_HIP` gate,AMD/HIP-only;`SGLANG_TRITON_COMPACT_EXTEND_ATTENTION`(EnvBool,默认 True,AMD 上默认启用)可在 launch 时读取,`use_compact_tile_grid` 显式参数保持 vendor-neutral 供测试。
- `triton_backend.py` 把 `forward_batch.extend_seq_lens_cpu` 传入 `extend_attention_fwd()`,compact 路径拿到精确 per-request extend 长度。
- 可选 scheduler tile-budget admission:`estimate_prefill_extend_tile_metrics()` 估计 legacy/compact 每 head query tiles,`SGLANG_PREFILL_TILE_BUDGET`(默认 0 关闭)超预算返回 `OTHER`,`SGLANG_PREFILL_TILE_BUDGET_MODE=compact|legacy` 选计量口径;BLOCK_M 从 attention backend 自动探测(`extend_attention_block_m`),非 HIP scheduler 行为完全不变。
- 顺带修复 `verify_splitkv.can_handle`:MLA(head_dim != v_head_dim,如 DeepSeek 576 vs 512)使用共享 latent KV 与 absorbed-attention 布局,split-KV verify kernel 会 GPU fault,现在拒绝并回退 `extend_attention_fwd`。

## 关键改动

```text
python/sglang/kernels/ops/attention/extend_attention.py: _compact_extend_q_tiles_per_head 与 kernel 的 USE_COMPACT_TILE_GRID 分支
python/sglang/srt/layers/attention/triton_backend.py: 传入 extend_seq_lens_cpu,自动探测 BLOCK_M
python/sglang/srt/managers/schedule_policy.py: estimate_prefill_extend_tile_metrics 与 tile-budget admission(HIP gate)
python/sglang/srt/environ.py: SGLANG_TRITON_COMPACT_EXTEND_ATTENTION(默认 True)等 env 定义
python/sglang/kernels/ops/attention/verify_splitkv.py: can_handle 拒绝 MLA head_dim 不匹配形状
```

## 性能声明

硬件 4× AMD Instinct MI355X(gfx950);Qwen2.5-7B-Instruct(dummy weights,TP=4,`--attention-backend triton --enable-mixed-chunk --chunked-prefill-size 2048`,`SGLANG_USE_AITER=0`),唯一变量为 compact flag:

- 固定 workload(`--request-rate inf`,CUDA graphs 开,3 seeds 均值,CV ≤ 2.5%):Ragged 4096/128×64 req 14,581→49,877 tok/s(+242.0%,TTFT −67.2%,TPOT −76.2%);Ragged 8192/64×32 req +201.2%;Uniform 4096/128×64 req 11,976→49,610 tok/s(+314.2%,TTFT −72.4%,TPOT −80.0%)。PR 注明:默认路径瓶颈在未被 graph 捕获的矩形 prefill grid,compact 后 decode graph 才能生效。
- Open-loop 8 req/s(saturating 前后):sustained request throughput 3.48→8.86 req/s(+154.6%),total throughput 11,182→28,439 tok/s(+154.3%),Mean E2E 16.68 s→0.88 s(−94.7%),P99 E2E 30.96→1.43 s(−95.4%),Mean TTFT 2,485→87 ms(−96.5%),Mean TPOT 147.2→8.2 ms(−94.4%)。
- 17 个模型 kernel microbench(单卡 MI355X,数值 bit-exact,max abs diff = 0.0):mixed(1×2048+31×1)加速 7.0×–15.0×,ragged(4×1024+28×8)3.4×–5.8×;覆盖 GQA/MQA/MHA、dense/MoE、head_dim 64/128/256(closed-weight 行按公开形状)。
- TinyLlama-1.1B 端到端(TP=1,ragged 2048/128,2 seeds):total throughput 8,375→67,477 tok/s(+706%),Mean TTFT 8,777→1,167 ms(−87%),Mean E2E 24,411→2,768 ms(−89%)。
- DeepSeek R1 MXFP4 prefill-only(4×MI355X,TP=4,Quark MXFP4,4 req/s,64 req):throughput 15,810→16,181 tok/s(+2.4%),mean latency 1.25→0.83 s(−33.8%),P50 −27.6%,P99 −14.8%;profiler 显示 attention 仅占 prefill 12%(communication 42%、MoE 34%),PR 据此解释收益有限;mixed workload(1 prefill + 10 decode)仅 +0.1%。
- Caveat(PR 原文):大收益仅限 Triton attention backend 的 mixed prefill,不得外推到 AITER / DeepSeek-R1-MXFP4 路径。

## 验证与回归风险

精度:GSM8K 1,319 题 5-shot,`amd/DeepSeek-R1-MXFP4` TP=4、Triton backend、compact 开启,accuracy 0.940(PR 标注 passes required >0.94)、Invalid 0.000、output throughput 1022.1 token/s,与 AITER 参考 0.946 在 run-to-run noise 内——该 e2e 路径实际执行了被修改的 Triton extend kernel。单元测试:`test_compact_extend_attention_tile_count`(tile 计数,含 None 回退)、`test_extend_attention_compact_grid`(compact vs legacy 输出 allclose)、tile-budget admission 3 项(含 `test_prefill_tile_budget_disabled_on_non_hip` 非 HIP no-op)、compact 模式下现有 extend-attention 与 sliding-window 测试通过。风险:kernel 新增一维 tile 索引到 (seq, block) 的运行时扫描分支;`verify_splitkv` 对 MLA 形状的行为变化(现在回退而非 fault);env 默认开启,需要用 `SGLANG_TRITON_COMPACT_EXTEND_ATTENTION=0` 做 A/B 对比。

## 关联

无(pack 未给出同区域其他 PR 编号;对比对象为 flash-attn varlen kernel 与 AITER 后端)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
