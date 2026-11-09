# PR #6612: [Kernel][Core] Add AWQ support to the Marlin kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @alexm-redhat |
| 合并 | 2024-07-21 |
| 分类 | Quantization |
| Label | ready |
| Diff | diffstat 缺失;pack 明示 diff 因超长截断(18 个文件未含) |
| 链接 | https://github.com/vllm-project/vllm/pull/6612 |

## 一句话结论

给 Marlin kernel 加 AWQ 端到端支持(zero-point dequant + 在线 repack),AWQ 权重可用 Marlin GEMM。

## 动机

AWQ 模型此前只能走原 AWQ kernel;Marlin 是面向 Ampere+ 的高速 4/8-bit GEMM,但只支持 GPTQ 的 scale 格式、不含 zero-point。要让 AWQ 权重受益于 Marlin,需要在 kernel 内支持 AWQ 的 zero-point 量化语义并转换其权重布局。

## 优化点

三部分:(1) 修改 `gptq_marlin.cu` 支持 zero-point——Marlin kernel 模板新增 `has_zp` 维度,新增 `dequant_4bit_zp`/`dequant_8bit_zp`(half 用 LOP3/hsub2/hfma2 魔数,bf16 用 fp32 中转的 byte_perm)与 `sub_zp`,zero-point 以与 scale 相同的 shared-memory/register 流水加载(`zp_ptr` 参数、`zp_sh_stage` 等),dequant 后先减 zp 再乘 scale;(2) 新增 AWQ 在线 repack kernel `awq_marlin_repack.cu`,双缓冲 `cp.async` 流水把 AWQ 打包布局转换为 Marlin tile 布局,支持 4/8 bit,要求 CUDA_ARCH>=8.0;(3) 新增 linear layer `awq_marlin.py`(在截断部分)。dispatch 侧 `__CALL_IF` 增加 HAS_ZP 维度并拆出 `GPTQ_CALL_IF`/`AWQ_CALL_IF` 两组特化(AWQ 走 has_zp=true)。顺带把 namespace 从 `gptq_marlin` 统一为 `marlin`(`fp8_marlin.cu` 跟随)。

## 关键改动

```text
csrc/quantization/gptq_marlin/gptq_marlin.cu: has_zp 模板维度、zp dequant/加载流水、GPTQ/AWQ CALL_IF 拆分
csrc/quantization/gptq_marlin/awq_marlin_repack.cu: AWQ→Marlin 在线 repack kernel(新文件)
csrc/ops.h: gptq_marlin_gemm 加 b_zeros/has_zp 参数;新增 awq_marlin_repack 声明
vllm/model_executor/layers/quantization/awq_marlin.py: 新 linear layer(在截断的 18 个文件内)
```

```c
if constexpr (has_zp) {
    frag_b0 = dequant_4bit_zp<scalar_t>(b_quant);
    ...
    sub_zp<scalar_t>(frag_b0, frag_zp[j], 0);
}
```

## 性能声明

PR 正文仅以两张截图给出 `awq_marlin`(本 PR)vs `awq`(main)的基准对比,未提供文字数字,本卡无法引用具体数值。测试条件:Llama3-70B AWQ、2xA100(prompt 512 / decode 256);Llama3-8B(图注写作 Llama3-7B)AWQ、1xA100(prompt 1024 / decode 512)。正文无其他性能声明。

## 验证与回归风险

PR 正文 TODO 明确:AWQ 模型的端到端 correctness test(对齐 GPTQ 已有的 fp16/bf16 测试)留待本 PR 合入后补;PR 未附测试结果,本卡未见测试证据。风险:Marlin 特化组合因 `has_zp` 维度翻倍(编译产物与编译时间);AWQ repack 为加载期一次性在线转换,首载开销增加;group_blocks 组合(-1/2/4/8)与 act_order 组合的矩阵需各自验证;18 个文件被截断,`awq_marlin.py` 与量化配置接线细节需看 PR。

## 关联

无(正文未引用其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
