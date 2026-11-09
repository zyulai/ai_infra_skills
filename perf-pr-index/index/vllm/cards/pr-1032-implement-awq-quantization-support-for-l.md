# PR #1032: Implement AWQ quantization support for LLaMA

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @WoosukKwon |
| 合并 | 2023-09-16 |
| 分类 | Quantization |
| Label | (none) |
| Diff | diffstat 缺失(素材未记录 +/- 行数与文件数);diff 因超长被截断,12 个文件未含 |
| 链接 | https://github.com/vllm-project/vllm/pull/1032 |

## 一句话结论

vLLM 首个 AWQ(4-bit 权重)支持:接入 AWQ 反量化 GEMM kernel,LLaMA 系模型可 `--quantization awq` 运行。

## 动机

AWQ(Activation-aware Weight Quantization,MIT-HAN-LAB 论文)以 4-bit 权重大幅压缩显存,但 vLLM 当时没有任何量化方案,大模型只能以 FP16 跑。本 PR 整合社区前序 PR #762、#926(作者 @ri938、@casper-hansen、@julian-q)的成果,先覆盖 LLaMA 系模型,是 vLLM 量化体系的起点。

## 优化点

三层改动:1) CUDA 侧,`csrc/quantization.cpp` 绑定 `awq_gemm`;`gemm_kernels.cu` 提供 m16n128k32 与 m16n64k32 两个 4-bit GEMM kernel(按 OC 是否 128 的倍数选择,Adapted from llm-awq),`dequantize.cuh` 的 `dequantize_s4_to_fp16x2` 用 `lop3` + `sub.f16x2`/`fma.rn.f16x2` PTX 把一个 32-bit 寄存器里的 8 个 int4 一步转成 FP16 对,再用 `ldmatrix`/`mma.sync` 做 tensor-core GEMM,支持 `split_k_iters`;2) 配置层,`ModelConfig` 增加 `quantization` 参数与 `_verify_quantization` 白名单(当时仅 `awq`),CLI 新增 `--quantization/-q`;3) 层注册,新增 `quantized_linear` 包与 `ParallelLinear` 工厂,按 `quant_config.get_name()` 从注册表分派到 `AWQColumnParallelLinear`/`AWQRowParallelLinear`,`create_weights` 建 `qweight`/`qzeros`/`scales`,`apply_weights` 调 `awq_gemm`。

## 关键改动

```text
csrc/quantization/awq/gemm_kernels.cu: awq_gemm 4-bit GEMM kernel(m16n128k32 / m16n64k32)
csrc/quantization/awq/dequantize.cuh: s4→fp16x2 PTX 反量化
vllm/model_executor/layers/quantized_linear/: AWQ Column/Row ParallelLinear 与分派工厂
vllm/config.py + vllm/engine/arg_utils.py: quantization 配置校验与 CLI
```

```cpp
static constexpr uint32_t BOTTOM_MASK           = 0x000f000f;
static constexpr uint32_t TOP_MASK              = 0x00f000f0;
static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;
// Note that the entire sequence only requires 1 shift instruction.
const uint32_t top_i4s = i4s >> 8;
asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(h[0])
                : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
```

## 性能声明

PR 正文未提供量化前后的性能/benchmark 数字,仅列出功能验证命令:`casperhansen/vicuna-7b-v1.5-awq` 单卡与 `-tp 2`、`abhinavkulkarni/lmsys-vicuna-33b-v1.3-w4-g128-awq -tp 2` 均可正常生成。

## 验证与回归风险

验证仅三个 example 命令,PR 无单元测试、无 accuracy eval。风险面:`awq_gemm` 入口要求 OC 为 64 与 8 的倍数、group size 为 32 的倍数(否则 throw);仅支持 LLaMA 系与半精度输入;`ld_A_flag` 注释明确 TODO 不支持 bsz > 16 的部分加载路径;作者自注 weight loading 与 ParallelLinear 逻辑"messy",后续需重构。作为 2023 年的早期 PR,其接口(quantized_linear 注册表)在后续版本中被更通用的 quantization framework 取代。

## 关联

采纳自 #762、#926(@ri938、@casper-hansen、@julian-q);kernel 出处为 mit-han-lab/llm-awq。素材未列其他关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
