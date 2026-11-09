# PR #61343: [Perf] per_token_group_quant_8bit: multiply by reciprocal instead of dividing per element

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gau-nernst |
| 合并 | 2026-10-13 |
| 分类 | Quantization |
| Label | quantization |
| Diff | diffstat 缺失(pack 未提供 +N/-N 与文件数;diff 内容可用,覆盖 2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/61343 |

## 一句话结论

FP8 分组量化 CUDA 内核由逐元素除法改为每组一次倒数乘法,GB300 内核提速 1.15–1.29x。

## 动机

`per_token_group_quant_8bit`(FP8 per-token-group 量化)内核里,`div.rn.f32` 是多指令慢序列,且原实现对每个元素都除一次 group scale,而每个 group 一次除法本就足够。该写法继承自移植来源 SGLang 内核(#21083),register-resident 路径(#55330)为与 smem 路径 bit-exact 而保留;本 PR 指出这一 bit-exact 约束只是测试便利、非精度要求,可两端同改。内核自身的 UE8M0 路径(2 的幂 scale,倒数精确)已经用倒数乘法,Triton fallback 亦然。

## 优化点

在 group scale 为正/需要反转的所有 fp8 group-quant 路径上,把"每元素 `x / y_s`"改为"每组算一次 `inv_y = 1.0f / y_s`,元素级乘法":smem 调度的 `QuantizeGroup` 与 register-resident 内核同步修改,保持两条调度 bitwise 一致(既有测试不变式);fused RMSNorm+group-quant 内核(`layernorm_utils.cuh`,`RMSNormQuantFusionPass` 的目标)也切到同一倒数乘法,使 fused 与 unfused 路径数值一致(此前两者在约 0.05% 元素上差 1 fp8 ulp),fused 内核每 vec4 还省 4 次 div.rn。per-token fp8 保留逐元素精确除法(FBGemm parity 要求),int8 路径不动。无 Python/dispatch 改动;文件经 hipify 进入 ROCm,只用可移植 float 运算(ROCm 未单独 benchmark)。作者对比过一个 flat-index Triton 变体(大 M 可达 6.7 TB/s 但小 M 更差),结论是改现有 CUDA 内核风险更低。

## 关键改动

```text
csrc/quantization/w8a8/fp8/per_token_group_quant.cu: smem/register 两调度改倒数乘法
csrc/quantization/fused_kernels/layernorm_utils.cuh: fused RMSNorm+quant 同步改倒数乘法
```

```cuda
// One div.rn per group instead of per element; the register-resident
// kernel does the same, keeping both schedules bit-identical.
const float inv_y = 1.0f / y_s;
auto scalar_op_quant = [&] __device__(DST_DTYPE & dst, const T& src) {
  float q = fminf(fmaxf(static_cast<float>(src) * inv_y, min_8bit), max_8bit);
  dst = DST_DTYPE(q);
};
```

## 性能声明

- 内核 microbench(GB300,K=4096,group=128,bf16→fp8e4m3 + fp32 scales,FlashInfer CUPTI + CUDA graph + 冷 L2,交错的若干轮取中位数,两侧均为 nvcc 13.1 独立构建):M=1..32768 全范围 1.15–1.29x;M=32768 为 101.25us→78.34us(1.29x),achieved BW 从 4.0 提到 5.2 TB/s。内核是 memory-bound。
- E2E(Qwen3.5-397B-A17B-FP8,TP4,8k input/1k output,`vllm bench serve` median TPOT,同一 GB300 节点顺序对比):TPOT 并发 1/4/16/64 分别 -0.6%/-1.9%/-2.0%/-1.2%(6.27→6.23、8.30→8.14、12.06→11.82、24.98→24.67ms);并发 64 请求吞吐 +1.2%。作者注明:e2e 数据含工具链效应(branch 用本地 nvcc 13.1 源码构建,基线是 nvcc 13.0 wheel,其 stock 内核被编译器误调度),略微高估纯源码改动效果;同工具链的内核数字才是诚实声明。
- 数值:scale 输出与 main bit 一致;量化值在约 0.05% 元素上可能差 1 fp8 ulp(商落在 fp8 舍入边界 1 fp32 ulp 内时)。

## 验证与回归风险

`pytest tests/kernels/quantization/test_per_token_group_quant.py` 848 passed,含 `test_per_token_group_quant_fp8_register_matches_smem`(register vs smem bitwise 相等,两调度同步修改后不变式保持);fused 侧由 `test_fused_quant_layernorm` 与 `test_mamba_prefix_cache` 覆盖一致性。精度评测(GSM8K flexible/strict -0.99pp/-1.67pp,GPQA-Diamond +1.01pp)在二项噪声(约 1.06pp)内,但 GSM8K 方向性略降,与 0.05% 元素舍入翻转一致。风险:所有走 CUDA fp8 group-quant 的模型路径数值行为微变(1 ulp 级);ROCm 分支同 numerics 但未 benchmark。

## 关联

#60903(为 prefill 大 M 增加 Triton row kernel,与本 PR 互补);#55330(register-resident 调度引入处);#21083(内核移植来源)。同分类还有 #18844 等 quant 调优类 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
