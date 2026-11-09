# PR #18612: [Perf][Kernel] Fuse SiLU+Mul into NVFP4 Expert Quantization for CUTLASS MoE

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JackChuang |
| 合并 | 2026-06-29 |
| 分类 | MoE (tier: major) |
| Label | quant, sgl-kernel, blackwell, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 数;diff 本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/18612 |

## 一句话结论

CUTLASS NVFP4 MoE 中把 SiLU+mul 与 FP4 expert 量化融合成单个 CUDA kernel,消除一次中间 buffer 分配与一次 kernel launch,隔离步骤快 1.28-1.42x,e2e 延迟/吞吐约 +5%。

## 动机

CUTLASS FP4 MoE 流水线里 GEMM1 与 GEMM2 之间原本要三步:分配中间 buffer -> `silu_and_mul` -> `scaled_fp4_experts_quant`。这意味着多一次显存分配、多一次 kernel launch,以及 SiLU(gate)x up 中间结果的一次完整 HBM 往返。设计灵感来自 vLLM PR #31832。由于 NVFP4 expert-quant kernel 是 Blackwell 专属(sm100a),优化目标硬件为 B200/GB200。

## 优化点

新增融合 kernel `silu_and_mul_scaled_fp4_experts_quant_packed_sm100a`:输入为 `(m, 2*k)` 的 gate+up 拼接张量,kernel 在一次遍历内读两半、算 SiLU(gate)x up、再做 FP4 量化并写 packed 输出与 block scale。实现上给 `cvt_fp16_to_fp4`(low-latency 与 offset-based 两个变体)加了独立的 `use_silu_and_mul` 开关——此前 SiLU+mul 隐式绑定在 `mask != nullptr` 上;新入口用 expert offsets(而非 mask)处理各 expert 间不均匀的 token 分布,并保留 blockscale 的 128-row padding 语义。Python 侧注册为 torch custom op,wrapper 负责维度推导(k = shape[1]//2)、输出/scale 分配(scale 输出 reinterprete 为 `float8_e4m3fn` 供 GEMM2 使用)。MoE 集成处 `cutlass_moe_fp4` 用一次融合调用替换三步路径;同时让 plain CUTLASS runner backend 跳过 MoeRunner 构建、直接走 `cutlass_moe_fp4` 路径。

## 关键改动

```text
python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_expert_quant.cuh: cvt_fp16_to_fp4 加 use_silu_and_mul 开关;新增 packed 融合入口(sm100a)
python/sglang/jit_kernel/nvfp4.py: 注册 fused torch custom op 与 Python wrapper
python/sglang/srt/layers/moe/cutlass_moe.py: cutlass_moe_fp4 用单次融合调用替换 alloc+silu_and_mul+quant 三步
python/sglang/srt/layers/moe/moe_runner/runner.py 与 modelopt_quant.py: cutlass backend 直接走 cutlass_moe_fp4 路径
test/registered/jit/test_silu_and_mul_scaled_fp4_experts_quant_packed.py: 不均匀 expert offsets 下与 unfused 路径 bit-exact 对比 + CUDA graph 性能回归底线
```

关键片段(cutlass_moe.py,替换后的调用):

```python
# fused: SiLU + mul then FP4 quant (expert-packed)
int_fp4, int_blockscale = silu_and_mul_scaled_fp4_experts_quant_packed(
    c1, a2_gscale, params.expert_offsets, params.blockscale_offsets, num_topk,
)
```

## 性能声明

实验硬件 GB200x4,模型 nvidia/Qwen3-30B-A3B-NVFP4:

- 隔离步骤(CUDA graph,Qwen3-30B-A3B MoE 维度 n=768、128 experts):tokens=1024 时 1.36x,tokens=4096 时 1.42x,tokens=16384 时 1.28x
- GSM8K 基准(8-shot):Accuracy 0.910 不变,Latency 13.969 s -> 13.298 s,Output throughput 1840.902 -> 1933.693 token/s(延迟与吞吐均约 5% 提升)
- bench_serving 吞吐场景(random-input 3500/output 1500,512 prompts):Request throughput 15.06 -> 15.28 req/s,Total token throughput 37021.74 -> 37543.55 tok/s(约 1.4% gain)
- bench_serving 延迟场景(random-input 100/output 100,8 prompts):Mean E2E 435.32 -> 426.12 ms,Mean TTFT 129.90 -> 125.13 ms(约 2% gain)

## 验证与回归风险

新增 UT `test_silu_and_mul_scaled_fp4_experts_quant_packed.py`:在刻意不均匀的 expert offsets(极小 expert、恰好 128 行、跨 128-row padding 的 expert)下,断言 fused 与 unfused 路径的 packed FP4 nibbles 与 de-swizzled block scales bit-identical,并用 fp32 `F.silu(gate)*up` 参考锚定(rel_l2 约 0.09,容差 0.2)防止两个 kernel 共有 bug 造成的空转通过;3 种布局 x n in {256,768} x {bf16,fp16} 共 12 例。性能测试另有 `>=1.05x` 的保守回归底线断言。UT 注册到 `base-b-kernel-unit-1-gpu-b200`。回归面:仅影响 Blackwell(SM100+)NVFP4 CUTLASS MoE 路径;需 sm>=100 的 RuntimeCheck;wrapper 依赖 MODELOPT_MAX_TOKENS_PER_EXPERT 上限断言;CI 状态显示 PR Test (Base)/(Extra) 均为失败标记(见 pack CI States),合并时需留意。

## 关联

灵感来源 vLLM PR #31832(vllm-project/vllm);同文件系既有测试 `test_nvfp4_blockwise_moe`、`test_silu_and_mul_quantize_to_fp4_grouped` 为其 sibling。pack 未列出同分类其他 sglang PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
