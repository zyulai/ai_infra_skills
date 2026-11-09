# PR #38040: [diffusion] quantization: ConvRot INT8 online W8A8 for DiTs (sgl-kernel + `convrot_int8_customkernel`)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @alex0dd |
| 合并 | 2026-10-02 |
| 分类 | Quantization(major) |
| Label | documentation, quant, sgl-kernel, run-ci, diffusion, jit-kernel, bypass-fail-fast |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/38040 |

## 一句话结论

新增免校准的 online ConvRot INT8 W8A8 量化 DiT,Hopper/Blackwell 上比 BF16 快 1.13-1.61x、权重减半,比 kitchen_int8 快约 2x。

## 动机

DiT 推理由少数大 GEMM 主导,权重显存占大头(Qwen-Image DiT 为 38 GiB)。既有的 online INT8 方案 `kitchen_int8` 依赖外部包 `comfy-kitchen`,且在 Hopper/Blackwell 上比 BF16 还慢(GB200 上仅 0.43x)。需要一个无校准数据、无额外依赖、无特殊 checkpoint、且在这些架构上真正快于 BF16 的 INT8 路径。

## 优化点

新增 `--quantization convrot_int8_customkernel`(文档化名称 `convrot_int8`,`kitchen_int8` 降级为别名):按 ConvRot(arXiv:2512.03673)在做 INT8 量化前施加 group-wise Hadamard 旋转——采用 regular Hadamard(行和为 1 的 4x4 块的 Kronecker 幂),使 per-row activation 与 per-channel weight scale 在通道离群值和非零组均值下仍准确;特意否决了 Sylvester Walsh-Hadamard(全 1 行把组均值集中到一个系数上,GELU 输出误差大 1.3-1.9x)。每个 linear 一次 fused op 完成 rotate + per-row 动态 INT8 量化 + CUTLASS INT8 GEMM + per-row x per-column 反量化;旋转核按 warp 寄存器驻留实现(1024 元素 tile,4-point stage + shuffle),K≤32768 无需共享内存瓶颈。后端:CC 9.0 走 WGMMA、CC 10.0 走 tcgen05(sm_100a)、CC 12.0/12.1 走 CUTLASS 2.x mma.sync;CC 10.3 显式拒绝(Blackwell Ultra 的 INT8 吞吐远低于 BF16,建议 FP8/NVFP4)。运行时自动 bit-exact 融合:共享输入的 q/k/v 投影只旋转一次;FFN down-projection 把 GELU(tanh) 折进旋转核(经 BF16 round-trip 保持 bitwise 一致)。量化在加载 BF16 checkpoint 后在线完成,输入宽度为 256 倍数的 linear 全部量化,支持 `--quantization-ignored-layers`。

## 关键改动

```text
python/sglang/kernels/jit/csrc/gemm/convrot_int8/convrot_int8_entry.cuh: 新增文件;Hadamard 旋转+量化核与 CUTLASS INT8 GEMM(WGMMA/tcgen05/mma.sync)
sgl-kernel csrc/gemm/convrot_int8_gemm.cu: convrot fused linear op(rotate+quantize+GEMM+dequant)
python/sglang/multimodal_gen: ConvRotInt8CustomKernelConfig/LinearMethod,qkv 单次旋转与 GELU 折叠调度
docs/docs/sglang-diffusion/quantization.mdx: ConvRot INT8 章节(backend 选择、TP/offload 限制)
```

```cpp
// Regular (non-Sylvester) Hadamard, the Kronecker power of
//   H4 = [[1,1,1,-1], [1,1,-1,1], [1,-1,1,1], [-1,1,1,1]]:
// one 4-point stage per base-4 digit ... every row sums to +1,
// so a group's mean stays spread over the coefficients.
template <int B0, int B1>
SGL_DEVICE void rot_stage_regreg(float (&v)[kRotElemsPerLane]) {
  for (int r = 0; r < kRotElemsPerLane; r++) {
    if ((r & (B0 | B1)) == 0) {
      const float a0 = v[r], a1 = v[r | B0], a2 = v[r | B1], a3 = v[r | B0 | B1];
      const float S = (a0 + a1) + (a2 + a3);
      v[r] = S - 2.f * a3;  ...
    }
  }
}
```

## 性能声明

均为 eager(无 CUDA graph)、每格 n=3、commit e5af511a87 上测得,Qwen-Image 1024x1024/50 steps/CFG 4 除注明外:

- GB200:BF16 7916 ms → 本 PR 6670 ms(1.19x);2 GPU CFG-parallel 4096 → 3617 ms(1.13x);peak 显存 56.0 → 37.0 GB。`kitchen_int8` 为 0.43x。
- H100 PCIe:22207 → 15150 ms(1.47x,kitchen 0.99x)。
- RTX PRO 6000 Blackwell:24118 → 15011 ms(1.61x,kitchen 1.42x)。
- DGX Spark(GB10):108439 → 73876 ms(1.47x)。
- Qwen-Image-Edit:21868 → 16571 ms(1.32x);FLUX.2-klein-base-4B:3882 → 2961 ms(1.31x,该模型当前 main 上 BF16 也输出噪声,仅作吞吐数据点)。
- Kernel 级(bench_kernels.py vs cuBLAS BF16):FLUX.2 形状 geomean 1.12x(kitchen 0.29x);Qwen-Image 形状 geomean 1.28x,FFN up-projection(3072→12288,8448 rows)1.69x。相对 L2 误差 ConvRot 0.009-0.012,kitchen 0.010-0.013。

## 验证与回归风险

测试:67 个 op 等式/Hadamard 参照/校验用例(`test_convrot_int8.py`)、multimodal_gen 单测、`test_qwen_image_joint_qkv.py`、`bench_convrot_int8.py`;全部在 GB200 通过,同一流程在 H100/H200、RTX PRO 6000、DGX Spark 通过,默认 BF16 路径与 main byte-identical。精度:Qwen-Image/Qwen-Image-Edit 同 seed 视觉对比等价(snowy-storefront edit 对 BF16 的 PSNR:ConvRot 25.2 dB vs kitchen 24.8 dB)。风险:INT8 输出与 BF16 非 bit-exact,不是一致性 ground-truth 模式;仅 MiniMax-H3、Qwen-Image、Qwen-Image-Edit 经过质量校验,其他 DiT 只保证可加载;LoRA 走 dynamic 模式,FSDP 下 `auto` 强制 comfy_kitchen;无 offload 加载需短暂容纳 BF16 权重的显存(Qwen-Image 38 GiB);jit 后端首次编译约 1 分钟。

## 关联

`kitchen_int8`(`comfy_kitchen.int8_linear`)为其前序 online INT8 方案,本 PR 后降级为别名并保留作 fallback 后端(Turing+);另有序列化 Comfy ConvRot INT8 checkpoint(`--component-weights-paths`)的离线权重流。方法出处 arXiv:2512.03673。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
