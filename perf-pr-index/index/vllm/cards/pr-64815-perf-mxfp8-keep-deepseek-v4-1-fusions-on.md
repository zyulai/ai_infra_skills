# PR #64815: [Perf][MXFP8] Keep DeepSeek-V4.1 fusions on the torch._scaled_mm kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Juntian777 |
| 合并 | 2026-11-02 |
| 分类 | Quantization |
| Label | documentation, ready, ci/build, deepseek, nvidia, DSv4.1 |
| Diff | diffstat 缺失(pack 标注 +None/-None);可见 diff 涉及 4 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/64815 |

## 一句话结论

让 cuBLASLt(`torch._scaled_mm`)MXFP8 kernel 接受 DeepSeek-V4.1 的输入量化融合,长 prefill 场景 prefill forward 快 2.4~3.7%(opt-in)。

## 动机

DeepSeek-V4.1 的 input-quant 融合(`QuantizedActivation` 通路)此前只被 FlashInfer CuTe-DSL / CUTLASS MXFP8 kernel 消费;走 `TorchMxFp8LinearKernel`(cuBLASLt via `torch._scaled_mm`)时输入需要重复量化,GEMM 本身在部分 shape 上也慢于 CuTe-DSL。作者希望以一个已有开关让 long-prefill 实例能把 MXFP8 linear GEMM 跑在 cuBLASLt 上且保留融合;默认 kernel 行为不变。早期修订曾用专门的 DeepSeek-V4.1 linear method 与 `VLLM_DSV41_MXFP8_CUBLASLT_MIN_M`,本版改为通用 kernel 能力扩展。

## 优化点

给 `TorchMxFp8LinearKernel` 增加 `input_quant_key()`(CUDA 上返回 `kMxfp8Dynamic`,对齐 FlashInfer F8_128x4 swizzled scale 布局),`apply_weights` 改为接受 `torch.Tensor | QuantizedActivation`:若上游已产出量化激活则直接用 `qa.data/qa.scale` 送入 `torch._scaled_mm`,否则维持原 `mxfp8_e4m3_quantize` 路径(XPU 不变)。随后把 DeepSeek-V4.1 侧的融合判定 `can_fuse_query_quant` 与 `_can_fuse_wo_a` 的 consumer 白名单加入该 kernel。通过 `--kernel-config '{"linear_backend_per_quant":{"mxfp8":"torch"}}'` 显式选择,保持 opt-in。

## 关键改动

```text
vllm/model_executor/kernels/linear/mxfp8/pytorch.py: input_quant_key + apply_weights 消费 QuantizedActivation
vllm/models/deepseek_v41/common/ops/query_quant.py: can_fuse_query_quant 白名单加入 TorchMxFp8LinearKernel
vllm/models/deepseek_v41/nvidia/ops/o_proj.py: _can_fuse_wo_a 白名单加入 TorchMxFp8LinearKernel
tests/fusion/test_quant_activation_contract.py: 该 kernel 加入 SUPPORTING 集合
```

```python
def input_quant_key(self) -> QuantKey | None:
    # CUDA activations use FlashInfer's F8_128x4 swizzled scale layout.
    return kMxfp8Dynamic if current_platform.is_cuda() else None
```

## 性能声明

- GB200,DEP4 target-model prefill forward(`vllm bench forward`,TP1/DP4/EP4,dummy weights,autotune on,`max-num-batched-tokens` 32768,每 arm 单次 run):2048 rows 33.39→32.57 ms/step (-2.44%),8192 rows 90.59→87.26 (-3.67%),16384 rows 165.61→161.29 (-2.61%),32768 rows 325.56→315.63 (-3.05%),对 autotuned CuTe-DSL。
- 对改动前的同 kernel:query 路径 GPU 时间 -36%~-84%,每 call host 时间约 -75%(不再重复量化);24 shape 上 plain tensor 输出 24/24 bitwise equal,query-quant 路径 19/24(最大相对 L2 8.8e-5),Fused WO-A 6/8(1.65e-3)。
- 纯 GEMM 210 shape(7 个 (N,K) x 30 行数,GB200,CUPTI + CUDA graph,冷 L2):>=8192 rows cuBLASLt 在 54/70 shape 更快,<8192 rows 为 85/140、最多慢 32%;每 call host 时间中位数 45.5 µs vs CuTe-DSL 147.9 µs——这是 opt-in、面向 long-prefill 主导实例的原因。
- GB300 上早期修订数据(经由 DSV4.1 method):>=8192 rows 时 cuBLASLt 6/7 shape 更快(均值 +2.15%~+4.67%),<8192 rows CuTe-DSL 多数占优、最多快 38%。
- 无本修订的完整权重精度或 serving 结果(见 PR Limitations);CUDA graph capture/replay 验证 bitwise 等价。

## 验证与回归风险

依赖 `tests/fusion/test_quant_activation_contract.py`(将 Torch kernel 纳入 quant-activation 契约测试);输出等价性以 CUPTI GEMM 级对比与 graph replay 验证,query-quant/WO-A 路径存在来自既有 fused kernel 的微小非 bitwise 差异(0-3/32768 rows)。风险面:默认 kernel 不变、特性 opt-in;融合判定改动影响 DeepSeek-V4.1 的 query-quant 与 WO-A 融合路径在 torch kernel 下的触发;capture 需先初始化 cuBLASLt(冷 capture 会报 `cudaErrorStreamCaptureInvalidated`);作者自述无完整权重 accuracy/serving 数据。

## 关联

同区域:索引内 DeepSeek-V4.1 MXFP8/量化相关卡 PR #49636(DSV4 opt-in FlashInfer MoE)、PR #51473(ROCm DSV4 原生 MXFP4)、PR #52861(NVIDIA 将 DSA 模型路由到 CUDA attention);FlashInfer MoE A2A 相关 PR #36022、#57995。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
