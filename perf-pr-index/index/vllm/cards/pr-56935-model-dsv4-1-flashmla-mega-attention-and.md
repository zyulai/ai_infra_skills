# PR #56935: [Model][DSv4.1] FlashMLA mega attention and the NVFP4 compressed KV cache

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zyongye |
| 合并 | 2026-09-16 |
| 分类 | Quantization |
| Label | performance, ready, deepseek, nvidia, DSv4.1 |
| Diff | diffstat 缺失(素材未记录 +/- 行数与文件数);diff 因超长被截断,16 个文件未含(backend 主体不在素材内) |
| 链接 | https://github.com/vllm-project/vllm/pull/56935 |

## 一句话结论

DSv4.1 接入 FlashMLA mega attention(Q RoPE/稀疏注意力/inv RoPE/FP8 cast 一 launch)与 288 B NVFP4 压缩 KV 记录,SM100 默认后端,TP1 decode 最高 1.45×。

## 动机

取代 #56344(base 分支被删)。DeepSeek V4.1 此前走 `FLASHMLA_SPARSE_DSV41` split-KV 管线:KV insert 后接 `flash_mla_with_kvcache`,再单独 `fused_inv_rope_fp8_quant`;FlashMLA 的 mega-attention kernel 把 Q RoPE、sparse attention、输出 inverse RoPE 与 FP8 cast 合并进一次 launch,直接写 `wo_a` 消费的 buffer;V4.1 还引入 288 B 的 NVFP4 压缩 KV 记录(相对 528 B fp8 记录小 45%),只有该 kernel 能读。

## 优化点

1) 新 `FlashMLAMegaAttnBackend`(`FLASHMLA_MEGA_ATTN_DSV41`)带自有 attention layer:Q 侧靠 `accepts_unnormed_unroped_query` 能力把 fused KV insert 缩为 zero-pad;输出侧 `_alloc_attn_out`/`_o_proj` 直接说 `QuantizedActivation`(kMxfp8Dynamic:e4m3 + packed-ue8m0 MN-major scales),一个 buffer pair 覆盖整步,一个 `fp8_einsum` 覆盖全部 N tokens。2) `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` 增加两个正交旋钮:`apply_q_rope`(runtime,KV 恒旋转)与 `is_q_interleaved`(模板,以 mega 的 32 chunks of [heads,16] 布局读写 Q,只改寻址);`q_head_padded=0` 表示 KV-only insert(TP1 的 64 live heads 已够,无 Q pass)。3) `nvfp4_ds_mla` KV 记录 = MXFP8 sliding-window record + 288 B 压缩记录(256 B e2m1 对 + 32 个 e4m3 scale/16 dims),沿用 byte-width dispatch,`rope_quant_insert`/`dequantize_and_gather_k_cache` 无需改调用点(gather 走 Triton,CuteDSL 无 fp4)。4) SM100 上未指定 backend 时默认解析为 mega;`is_available_for` 检查 SM100/头数/构建可用性,TP16(4 local heads)回落 `FLASHMLA_SPARSE_DSV41`。`wq_b`/`wo_a` 在 load 时一次性 permute 到 chunk-interleaved 布局,DSpark draft 层同样处理。

## 关键改动

```text
csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu: apply_q_rope / is_q_interleaved 旋钮与 KV-only 实例化
benchmarks/kernels/benchmark_dsv41_mega_attn.py: 新增 mega vs split-KV decode benchmark
vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py(推断): backend/layer(16 个文件被截断)
tests/kernels/test_compressor_kv_cache.py: NVFP4 record 字节级对照测试
```

```cpp
template <bool Q_INTERLEAVED>
__device__ __forceinline__ int64_t qElemOffset(int const tokenIdx,
                                               int const slotIdx,
                                               int const laneId,
                                               int const dim_base,
                                               int const num_heads) {
  if constexpr (Q_INTERLEAVED) {
    // FlashMLA mega layout: 32 chunks of [num_heads, 16] per token;
    // padding heads land at the tail of every chunk.
    return static_cast<int64_t>(tokenIdx) * num_heads * kHeadDim +
           static_cast<int64_t>(laneId) * num_heads * kElemsPerLane +
           static_cast<int64_t>(slotIdx) * kElemsPerLane;
  } else {
    return (static_cast<int64_t>(tokenIdx) * num_heads + slotIdx) * kHeadDim +
           dim_base;
  }
}
```

## 性能声明

- `benchmark_dsv41_mega_attn.py`,单 GB300,CUDA graphs,计入每步全部 op(topk_swa=128、topk_extra=512、NVFP4 压缩 cache),64-head 模型,mega/`FLASHMLA_SPARSE_DSV41` µs:TP1(64 live heads)s_q=512 时 96/140(最高 1.45×),s_q=128 时 31/40;TP2–TP8 各 s_q 基本持平(±几个百分点,如 TP8 s_q=512 为 107/104)。决定变量是 live heads 对 padded heads(恒 64),无 fallback 阈值。
- 默认后端的选择依据为 TP1 的收益 + 45% 更小的压缩记录,PR 明确"不是均匀提速"。
- 精度(GB300 TP4,DeepSeek-V4.1-Flash pinned snapshot,DSpark k=5 真实 rejection sampling):mega(NVFP4)gsm8k flexible/strict 0.9318/0.9318 vs sparse(fp8_ds_mla)0.9265/0.9265;gpqa exact/pass@4 0.9053/0.9545 vs 0.9066/0.9545;均在历史 IQR 内;draft acceptance gpqa 0.4376 vs 0.4341、gsm8k 0.4957 vs 0.4945。
- Prefill 性能未测;eager 模式下 launch overhead 会数倍夸大 fused 路径,数字须在 CUDA graphs 下测。

## 验证与回归风险

5 个 kernel 测试套件 522 passed / 17 skipped:NVFP4 record 字节级对照(由 FlashMLA `tests/quant.py` 推导的 reference quantizer,scales 精确、e2m1 值半步容差、未触及行与页 padding 保留)、insert→gather round trip、`wq_b`/`wo_a` permutation 对照、interleaved Q pad(live 元素 bit-exact、padding 头为零、两种 grid 变体、KV-only launch)、norm/RoPE 与布局的正交性(8 组合逐 bit)、graph-replay 下 Q buffer 重填。风险:仅 SM100(kernel 无 SM90 实例化,默认在非 SM100 主动让位);FlashMLA FP8 decode 只接受 h_q ∈ {64,128},TP16 不可用;prefill 未测;系列中两个 commit 单独不可 build(影响 bisect,不影响 tip);显式 `nvfp4_ds_mla` 在 plain-row(FlashInfer)层抛 ValueError。

## 关联

取代 #56344(同一集成的 flag-gated 版本,已关闭);构建于 #56893(V4.1 MXFP8 KV record,已合并);与 #56441(KV-only context insert,同翻译单元)的交互已在 rebase 中处理。索引同区域另有 pr-13747(FlashMLA integration)、pr-43008(DSv4 cluster cooperative decode)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
