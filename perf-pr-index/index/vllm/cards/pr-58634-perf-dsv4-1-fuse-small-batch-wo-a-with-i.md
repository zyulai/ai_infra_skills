# PR #58634: [Perf][DSv4.1] Fuse small-batch WO-A with inverse RoPE and MXFP8 quant on SM100/SM103

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Juntian777 |
| 合并 | 2026-09-27 |
| 分类 | Quantization |
| Label | ready, deepseek, nvidia, quantization, verified, DSv4.1 |
| Diff | diffstat 缺失(pack 中为 +None/-None, None files);diff 含 2 个新文件(fused_wo_a.py 约 599 行、o_proj.py)及 3 个修改文件,另有 1 个文件被截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/58634 |

## 一句话结论

在 SM100/SM103 上用单个 CuTe-DSL kernel 把 DSv4.1 O-projection 前三个 latency-bound kernel(逆 RoPE+FP8 量化、WO-A GEMM、MXFP8 量化)合成一个,小 batch WO-A 链路最快 2.08×。

## 动机

DSv4.1 的 O-projection 在 SM100/SM103 上每层要跑 4 个 kernel:attn_out 先经 Triton 逆 RoPE + per-32 FP8 量化,再 DeepGEMM `fp8_einsum wo_a`,再 FlashInfer MXFP8 quantize,最后 FlashInfer CuTe-DSL `wo_b`。decode 小 batch 下前三个都是 latency-bound,在 GB200(DSv4.1-Flash,TP=4)合计约 11-12 µs,是小 batch decode 的固定开销。DeepGEMM 26/09 版本虽能让 `fp8_einsum` 从 GEMM epilogue 直接产出 MXFP8,但只覆盖第三个 kernel 且只能喂 DeepGEMM 的 `wo_b`,无法把逆 RoPE+量化融进 A 操作数侧,因此选择自写 CuTe-DSL kernel。

## 优化点

新增 `FusedWoAKernel`(CuTe-DSL,遵循 `VllmCuTeDSLJitKernel` warmup 契约),把 pre-`wo_b` 的整条链折叠为一个 kernel,直接输出预量化 `QuantizedActivation` 供 `wo_b` 跳过自身量化。机制:每个 cluster 拥有 128 行 `wo_a` 输出、每个 CTA 负责 K 方向一个 head 的 split-K——(1) PDL 下权重预取,`griddepcontrol.wait` 前 TMA 预载 64 KiB FP8 权重切片与 packed scales,与 attention kernel 重叠;(2) 逆 RoPE(仅每 head 末 64 维)+ per-32 UE8M0 量化直接写入 swizzled SMEM MMA 操作数,无全局往返;(3) `tcgen05` MXFP8 MMA 流水线,每个 128 宽 K stage 独立 mbarrier,最后一级触发 `griddepcontrol.launch_dependents` 放行 `wo_b`;(4) split-K 部分和经 DSMEM 归约、按原路径舍入到 BF16 后量化为 FlashInfer F8_128x4 scale 布局。显式 `mul.rn.f32`/`fma.rn.f32` 保持原 RoPE 舍入顺序。融合触发条件:SM100/SM103、每 group ≤8 个本地 head、`o_lora_rank` 为 128 倍数、`wo_a` per-32 FP8 且 `wo_b` 为 FlashInfer CuTe-DSL MXFP8;推送的 head 支持 1–32 tokens,本地扩展到 1–96,其余 batch 走 `deep_gemm_fp8_o_proj`。

## 关键改动

```text
vllm/models/deepseek_v41/nvidia/ops/fused_wo_a.py: 新增 FusedWoAKernel(PDL 权重预取 + 逆 RoPE/量化 + tcgen05 MXFP8 MMA + split-K 归约与输出量化)
vllm/models/deepseek_v41/nvidia/ops/o_proj.py: dsv41_o_proj 分发(满足条件走融合,否则 deep_gemm_fp8_o_proj)与 warmup 注册
vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py / flashmla.py: _o_proj 改调 dsv41_o_proj 并注册 warmup
vllm/models/deepseek_v41/attention.py: _wo_b_proj 接受 QuantizedActivation,GEMM-RS 仅对非预量化输入生效
tests/kernels/test_fused_inv_rope_fp8_quant.py: 新增 test_dsv41_fused_wo_a_matches_deepgemm
```

融合守卫(节选):

```python
return (
    current_platform.is_device_capability_family(100)
    and layer._einsum_recipe == (1, 1, 32)
    and get_input_quant_key(layer.wo_b) == kMxfp8Dynamic
    and isinstance(getattr(method, "kernel", None),
                   FlashInferCutedslMxfp8LinearKernel)
    and layer.n_local_heads // layer.n_local_groups <= 8
    and (layer.nope_head_dim, layer.rope_head_dim) == (448, 64)
    and layer.o_lora_rank % 128 == 0
)
```

## 性能声明

Kernel microbenchmark(1× GB200,DSv4.1-Flash TP=4 单 rank 形状,BF16 `[T,16,512]`,FP8 权重 `[2,1024,4096]`,CUPTI + CUDA graph + 冷 L2,WO-B 不计入):WO-A 链 T=1 10.976→5.280 µs(2.079×),T=8 2.074×,T=32 1.602×,T=64 1.422×,T=96 1.028×;分段中位:1–32 tokens 1.867×,33–64 1.558×,65–96 1.072×,97–128 0.978×(111 起慢 2–7%,故 dispatch 止于 96)。

Serving benchmark(4× GB200,TP4,无投机解码,random 8,192 输入/1,024 输出,`ignore_eos`,prefix caching 关,`FLASHINFER_MLA_SPARSE_DSV41`):并发 1 时 ITL p50 4.39→4.12 ms(−6.2%)、TPOT p50 −6.1%、输出吞吐 217.7→231.2 tok/s(+6.2%);并发 8 时 ITL −4.7%、吞吐 +4.4%;并发 32 时 ITL −2.1%;并发 96(融合仅 1.03×)时 ITL p50 高 0.6%、吞吐低 0.8%。

精度:GSM8K 全量 1,319 题,PR DSpark strict 93.03%/93.33%/93.33%/93.86%(4 次运行)vs base 93.03%/92.57%,acceptance length 与 base 差 ≤0.007,声明为 parity 而非提升;GPQA Diamond 单次 91.92%(±1.9)。

## 验证与回归风险

测试:`tests/kernels/test_fused_inv_rope_fp8_quant.py` 新增 T∈{1,3,6,8,12,16,18,32} 对未融合 Triton+DeepGEMM+FlashInfer 参考链的对比(CUDA graph replay、scale padding 校验,相对 L2 < 3e-3,多数 bit 级一致);最新优化后 90 passed,768 个 CUDA graph case 与前版融合 kernel 比对全部 bit-identical,Compute Sanitizer racecheck 0 错误。96-token 扩展对未融合链最大相对 L2 0.00074270298。风险面:仅影响 SM100/SM103 上满足守卫条件的 DSv4.1 O-projection 小 batch 路径,`wo_b` GEMM-RS 对预量化输入自动跳过(GEMM-RS 本就从 128 tokens 才启用);warmup 覆盖所有可 dispatch 的 token 数(96 keys 编译 54.11 s)以避免运行时编译;PR 注明本地 FlashInfer 0.6.18 的 sparse-attention API 不兼容会阻止 server 初始化,评测数字来自 FlashInfer 0.7.0 环境。AI 辅助(Claude Code/OpenAI Codex)已在 PR 中声明。

## 关联

同区域:#57977(draft,同样针对 `wo_a`→`wo_b` 交接但保留 `fp8_einsum`、`wo_b` 改 DeepGEMM,call site 重叠)、#58621(改 DSv4 而非 DSv4.1)、#56562(DSV4.1 input metadata 融合,同为小 batch latency 优化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
