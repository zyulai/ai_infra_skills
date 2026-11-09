# PR #41790: [DSA] Q8KV8 FP8 Sparse Prefill on GLM-5.3-Flash with BF16 KV Cache

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JackChuang |
| 合并 | 2026-10-13 |
| 分类 | Quantization(tier: major) |
| Label | documentation, quant, run-ci, jit-kernel |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及约 11 个文件(素材另注明 4 个文件因超长被裁剪) |
| 链接 | https://github.com/sgl-project/sglang/pull/41790 |

## 一句话结论

让 GLM-5.3-Flash 在 Hopper 的 bf16 KV cache 上跑 FP8 稀疏 prefill:每 DSA 层 1.53–1.72x,端到端吞吐 +11%–12%。

## 动机

SM90 上 FP8 的峰值 tensor-core 吞吐是 BF16 的两倍、每操作数字节减半,`flashmla_sparse_q8` 已服务 DeepSeek-V3.2(#30514)与 GLM-5.2(#31888),但 GLM-5.3-Flash 完全无法启用,三个模型特性各自阻塞:(1) Hopper 下该模型默认 bf16 KV cache,而 q8 后端只接受 `fp8_e4m3`;(2) NoPE MLA(`qk_rope_head_dim=0`)使 fused fp8 q-prep Triton kernel 里 `tl.arange(0, ROPE)` 在 ROPE=0 时编译失败,`SGLANG_ENABLE_DSA_Q8KV8_KV_CAT_FUSION` 背后的 KV concat-cast kernel 同理;(3) KPool indexer 会在每个 top-k 行尾追加至多 `index_kpool-1` 个 tail token,行宽 2051,而 kernel 按 64 宽块成对走 top-k、要求 `topk % 128 == 0`,且 KPool tail guard 不放行 q8。

## 优化点

不改动 SM90 attention kernel 本身(`d_qk=512` 已是其支持宽度,local heads pad 到 64 满足契约),只修 kernel 周边:(1) bf16 KV cache——validator 接受 bf16 pool,q8 extend 批无论 pool dtype 都走 RAGGED top-k transform,prefix 行按 slot gather 并在调用时 cast 成 fp8(新增 `gather_cast_kv_fp8_pad_paged`:NoPE 行用一个向量化 Triton kernel 完成 gather+cast 并同时清零 -1 哨兵 pad 带);(2) NoPE——两个 Triton kernel 用静态 `if ROPE > 0:` 跳过 rope 段,ROPE>0 时输出逐字节不变;(3) KPool——top-k 用 -1 pad 到 2176(输出不变),tail guard 对 bf16 pool 放行 q8 prefill(fp8 pool 保持 guard,因未测试)。性能上把 tilelang 每层新建 pad 张量的做法换成 per-backend 持久 grow-only `(rows, 2176)` int32 缓冲(-1 填充一次、跨 DSA 层与 forward 复用,8192 chunk 时 71.3 MB)。另补启动期 guard(拒绝 mixed-chunk、HiSparse、unpacked fp8 pool、NoPE+fp8 pool)与 RAGGED-only 运行时检查。默认关闭,需显式 `--dsa-prefill-backend flashmla_sparse_q8`,SM90-only。

## 关键改动

```text
python/sglang/srt/layers/attention/dsa_backend.py: q8 validator、topk pad 持久缓冲、bf16 pool 的 KV gather-cast 接线
python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py: 新增 gather_cast_kv_fp8_pad_paged;KV cast kernel 支持 ROPE=0
python/sglang/kernels/ops/kvcache/cache_ops.py: q-prep kernel 静态跳过空 rope 段
python/sglang/srt/layers/attention/dsa/dsa_backend_kpool.py: KPool tail guard 对 bf16 pool 放行 q8 prefill
test/registered/kernels/ops/attention/test_q8kv8_dsa_nope_bf16_prefill.py: 逐字节一致与 torch reference 用例
```

```python
def _pad_topk_width_buffered(self, topk_indices, *, multiple):
    """Widen top-k to a multiple of ``multiple``, padding with -1.

    The destination is a persistent grow-only buffer rather than a fresh
    ``torch.cat``: only the real columns change between calls, while a cat
    reallocates and rewrites the whole ``[num_tokens, padded_width]`` int32
    tensor on every DSA layer call (~71 MB at an 8192-token chunk).
    """
```

## 性能声明

全部 H200。Per-DSA-layer(该模型 45 层中仅 11 层为 DSA):16 heads(TP4)在 8k–131k context 下 1.53–1.72x(如 s_q=4096/kv_len=8192:3.07→1.79 ms),64 heads 1.79–1.98x。端到端 prefill-dominated workload(random dataset,1-token outputs):TP4/EP4 总 token 吞吐 +11.16%–12.44%,TTFT 低 8.37%–10.97%(8k–120k);TP8/EP8 吞吐 +5.82%–12.06%,TTFT 低 6.06%–11.28%。绝对值示例(中位数):120k input TP4/EP4 吞吐 29,344→32,611 tok/s,TTFT 4,109→3,699 ms。kernel 时间分解(TP8/EP8):sparse MLA prefill 占 tilelang 总 kernel 时间 30.7%,q8 使其 256.1→149.0 ms(1.72x),q8 prep 开销 0.4%,总 kernel 时间 -10.6%。accuracy:gsm8k 1319 题 × 4 个 shot 数(5/20/60/120),无任何 shot 数 q8 低于 base;LongBench-v2 336 题 delta −0.003(单次噪声约 ±0.027)。

## 验证与回归风险

新增两个注册测试:CPU routing 单测 24 cases(top-k layout、启动接受/拒绝、pad 缓冲复用、KPool tail guard);1-GPU 单测 16 cases(q-prep/KV cast/prefix gather 与 plain torch cast 逐字节一致;该模型形状下 prefix 与非 prefix 输出一致、padding 无影响、对齐 torch reference)。既有 `test_sparse_mla_q8kv8_prefill_sm90.py` 等不变且通过。回归面:fp8 KV 行为(DeepSeek-V3.2/GLM-5.2)除启动检查外不变;默认路径(opt-out)零改动。fp8 pool + NoPE 模型尚不支持,future work 为接 fp8 KV cache 直接读 fp8 行。pack 中 CI:Base 通过,Extra 与 AMD ROCm 10 为失败标记。

## 关联

前序:#25746(SM90 Q8KV8 FP8 sparse-attention roadmap)、#30514(DeepSeek-V3.2 q8 sparse prefill)、#31888(扩展至 GLM-5.2)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
