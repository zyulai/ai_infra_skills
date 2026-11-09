# PR #5218: Support W4A8 quantization for vllm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @HandH1998 |
| 合并 | 2024-07-31 |
| 分类 | Quantization |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供,正文 diff 另有 10 个文件被裁剪) |
| 链接 | https://github.com/vllm-project/vllm/pull/5218 |

## 一句话结论

为 vLLM 引入 QQQ W4A8 量化方案与配套 Marlin int8 tensor core GEMM 内核,对比 FP16 最高 2.24x 加速。

## 动机

PR 之前 vLLM 已有 W4A16(Marlin)、W8A8 等量化路线,但缺少 weight 4-bit + activation 8-bit 的 W4A8 路线。作者团队在论文 QQQ(arXiv 2406.09904)中提出 W4A8 量化方案,目标是兼得 4-bit weight 的压缩率与接近 W8A8/W4A16 的推理性能,并将其集成进 vLLM。

## 优化点

核心是基于 Marlin GEMM 改写的 W4A8 GEMM CUDA 内核:activation 由上层量化为 int8,用 int8 tensor core 指令(`mma.sync...s32.s8.s8.s32`)做矩阵乘;4-bit weight 在内核内经 LOP3/`prmt` 快速 dequant 成 int8 后参与计算。内核支持三级 scale:activation 的 per-token scale(s_tok)、weight 的 per-channel scale(s_ch)与 per-group scale(s_group),并在 shared memory 流水线中异步预取。后续更新将 dynamic activation quantization 融合进 `QQQLinearMethod`,以支持所有模型(作者注明该融合会损失部分推理性能,论文数字需用其 fork 复现)。为复用代码,把 dense Marlin 内核中的 `base.h`/`mem.h` 公共工具抽到 `marlin/dense/common/`。

## 关键改动

```text
csrc/quantization/marlin/qqq/marlin_qqq_gemm_kernel.cu: 新增 W4A8 Marlin GEMM 内核(int8 mma + LUT dequant,约 1243 行)
csrc/quantization/marlin/dense/common/base.h, mem.h: 从 dense 内核抽出的公共 base/mem 工具
csrc/ops.h: 暴露 marlin_qqq_gemm 算子接口
.buildkite/lm-eval-harness/configs/models-small.txt: 新增 Meta-Llama-3-8B-QQQ 评测配置
```

dequant per-group 关键片段:

```cuda
uint32_t t0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
*reinterpret_cast<half2*>(&t0) = __hfma2(..., *reinterpret_cast<half2*>(&double_s),
    *reinterpret_cast<const half2*>(&MAGIC_NUM));
frag_b[0] = (uint8s ^ UINT8s_TO_INT8s_MASK);  // 4 half -> 4 int8 打包进 1 个 uint32
```

## 性能声明

PR 正文:QQQ 相比 FP16、W8A8、W4A16(Marlin)分别最高 2.24x、2.10x、1.25x 加速。测试条件:vllm v0.4.1,1x A100 80G,CUDA 11.8,LLaMA-2 量化模型同 batch 吞吐对比,input length 1024 / output length 128(具体数字以截图给出)。W4A8 GEMM 内核相对 PyTorch FP16 GEMM(CUTLASS)的加速比随输入 token 数变化,weight 矩阵 (N=8192, K=21760),数值在截图中未以文字给出。模型质量在 WikiText2 与五个 zero-shot 任务上评估(结果为截图)。其余条件 PR 未注明。

## 验证与回归风险

CI 侧新增 lm-eval-harness 的 QQQ 模型配置。内核依赖 `cp.async` 与 int8 mma,仅在 `__CUDA_ARCH__ >= 800`(Ampere+)编译可用。融合 dynamic activation quant 后性能有所回退,论文数字需用作者 fork(HandH1998/vllm w4a8-fusion 分支)复现;QQQ 权重导出工具当时仅支持 llama。diffstat 缺失,pack 中另有 10 个文件(含 Python 侧实现)被裁剪,影响面以 PR 全量 diff 为准。

## 关联

无(同一批 Quantization PR 索引中暂无直接前序/后续编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
