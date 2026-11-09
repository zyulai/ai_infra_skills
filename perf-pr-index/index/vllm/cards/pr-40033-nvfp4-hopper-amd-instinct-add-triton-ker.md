# PR #40033: [NVFP4][Hopper/AMD Instinct] Add Triton kernels for NVFP4 dequantization and QDQ emulation

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fxmarty-amd |
| 合并 | 2026-04-30 |
| 分类 | Quantization |
| Label | performance, rocm, ready, quantization, verified |
| Diff | diffstat 缺失(pack 未提供) |
| 链接 | https://github.com/vllm-project/vllm/pull/40033 |

## 一句话结论

为 NVFP4 模拟路径新增 Triton dequant/QDQ 内核,替代 torch eager 参考实现,dequant 提速约 10-12x、quant-dequant 提速约 28-58x。

## 动机

在不原生支持 NVFP4 dtype 的设备(AMD Instinct MI3xx、H100)上,vLLM 用量化模拟(emulation)运行 NVFP4 模型,但参考实现是大量堆叠的 torch 小 kernel,`--enforce-eager` 下反复读写 HBM;且这些堆叠调用会触发 pytorch/pytorch#180341 与 ROCm/rocm-systems#5071——近期 ROCm 版本 / pytorch nightly 上直接 HSA 崩溃,MI355X 上跑 NVFP4 模拟的 `vllm bench serve` 必须设 `HSA_TOOLS_DISABLE_REGISTER=1` 才能运行,而该环境变量本身很慢。接续 PR #35733 与 #35737 的工作,本 PR 用融合 Triton kernel 替代参考实现。

## 优化点

在 `nvfp4_emulation_utils.py` 中新增两个 Triton kernel,并在 CUDA-alike 平台上于 `dequantize_to_dtype` / `ref_nvfp4_quant_dequant` 入口自动替换 torch eager 路径:(1) dequant kernel——2D grid(rows x tiles),用 3 级二叉树 `_e2m1_inline` 替代 7 次顺序比较做 E2M1 查表,scale 以 uint8 视图载入后 bitcast 为 float8e4m3,结果 `tl.interleave` 交错以便合并连续 store;按行数自适应 tile(行数 >= 4096 整行一个 tile、>= 2048 整行 2 warps、否则 64-block tile 4 warps)兼顾 CU 利用率;(2) fused quant-dequant kernel——逐 group 取 `max(abs)` 生成 fp8e4m3 block scale,按参考实现的阈值四舍五入到 E2M1 再反量化,全程单 kernel。另将 E2M1 查找表在 CUDA graph capture 前显式搬到设备(`.to(device)` 不允许发生在 capture 期间),并把 `ref_nvfp4_quant_dequant` 返回签名从 `(Tensor, None)` 改为 `Tensor`,同步 `moe_kernel_quantize_input`。

## 关键改动

```text
vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py: 新增两个 Triton kernel 与 CUDA-alike 自动 dispatch
tests/kernels/quantization/test_nvfp4_emulation.py: 正确性(atol=0/rtol=0)+ microbenchmark,用 nvidia/Qwen3-30B-A3B-NVFP4 真实权重
vllm/model_executor/layers/fused_moe/oracle/nvfp4.py: EMULATION backend 在 capture 前搬 E2M1 LUT 到设备
vllm/model_executor/layers/fused_moe/utils.py: 适配 ref_nvfp4_quant_dequant 新签名
```

```python
if not swizzle and current_platform.is_cuda_alike():
    return _triton_dequantize_nvfp4(
        tensor_fp4, tensor_sf, global_scale, dtype, block_size)
```

## 性能声明

dequantize(2D):[512,1024] 12.63x、[1024,1024] 10.74x、[2048,1024] 10.10x、[512,2048] 10.93x;dequantize(3D,MoE experts 堆叠):[128,768,1024] 11.27x、[256,768,1024] 11.35x、[128,1536,1024] 11.31x、[128,768,2048] 11.61x。quant_dequant:[1x16]-[1024x1024] 约 55.11x-58.44x;大形状 [5120x2048] 31.68x、[2048x4096] 30.28x、[4096x7168] 28.30x、[8192x8192] 43.23x。基准为 pytest 内置 do_bench,对比 CUDA 上被禁用 Triton dispatch 的 torch eager 参考路径;测试运行设备为 `CUDA_VISIBLE_DEVICES=7` / `=5`,具体 GPU 型号 PR 未注明。

## 验证与回归风险

两个新测试 `test_triton_dequantize_nvfp4` / `test_triton_nvfp4_quant_dequant`:Triton 输出与参考实现 bitwise 一致(atol=0, rtol=0),覆盖 2D/3D、多种形状与 global scale,dequant 测试用真实 NVFP4 checkpoint 权重。风险:仅替换 `swizzle=False` 路径,swizzle 布局仍走原实现;签名变更触及 fused MoE quantize 调用点;LUT 预搬移依赖 `convert_to_nvfp4_moe_kernel_format` 时机,对 EMULATION backend 生效。

## 关联

PR #35733、PR #35737(同区域 NVFP4 模拟工作,本 PR 与后者独立但紧密相关)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
