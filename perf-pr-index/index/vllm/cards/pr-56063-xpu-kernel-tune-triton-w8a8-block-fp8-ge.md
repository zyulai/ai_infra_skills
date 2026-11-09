# PR #56063: [XPU][Kernel] Tune Triton W8A8 block-FP8 GEMM for Intel B70

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @pmanczak |
| 合并 | 2026-10-02 |
| 分类 | Quantization \| tier: major |
| Label | performance, intel-gpu, quantization |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注 12 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/56063 |

## 一句话结论

为 Intel Arc Pro B70 提供 W8A8 block-FP8 GEMM 调优配置，DeepSeek-V2-Lite 端到端 109.63 → 140.88 tok/s（1.285x）。

## 动机

PR 正文 Purpose：沿用 #53065（XPU Triton fused MoE tuning）的模式，把 `benchmarks/kernels/benchmark_w8a8_block_fp8.py` 移植到 Intel XPU，并为 `block_shape=[128,128]` 的 W8A8 blockwise FP8 GEMM 在 Intel(R) Arc(TM) Pro B70 Graphics 上交付调优后的 Triton kernel 配置。正文指出 B70 上单配置 Triton JIT 编译约 1-8s（CUDA 低于 100ms），因此需精简调优搜索空间。

## 优化点

benchmark 脚本放宽平台断言支持 `is_xpu()`，用 `current_platform.device_type` 替代硬编码 `"cuda"`，并 guard `torch.cuda.init()`；新增 `get_xpu_tuning_space()`，把调优网格由 1280 个配置精简为 96 个（num_stages 2/3、block_m 32/64/128、block_k 64/128、block_n 128/256、num_warps 4/8/16/32、group_size 1/32），仅在 `current_platform.is_xpu()` 时启用，`tune_on_gpu` 按平台选择 search_space。在 `vllm/model_executor/layers/quantization/utils/configs/` 下新增 26 个 tuned config JSON，命名含 `device_name=Intel(R)_Arc(TM)_Pro_B70_Graphics,dtype=fp8_w8a8,block_shape=[128,128]`，覆盖 DeepSeek-V3、DeepSeek-V2-Lite(TP=1)、Llama-3.1-8B(TP=1，packed QKV/gate-up) 及 Qwen3-8B 的 dense-linear 形状。

## 关键改动

```text
benchmarks/kernels/benchmark_w8a8_block_fp8.py: 支持 XPU 平台、device_type 泛化、新增 get_xpu_tuning_space() 与按平台选择 search_space
vllm/model_executor/layers/quantization/utils/configs/N=*,K=*,device_name=Intel(R)_Arc(TM)_Pro_B70_Graphics,dtype=fp8_w8a8,block_shape=[128,128].json: 26 个新调优配置（如 N=576,K=2048、N=22016,K=2048、N=24576,K=7168 等）
```

```text
+    assert (
+        current_platform.is_cuda()
+        or current_platform.is_rocm()
+        or current_platform.is_xpu()
+    ), "Only support tune w8a8 block fp8 kernel on CUDA/ROCm/XPU device."
+    search_space = (
+        get_xpu_tuning_space()
+        if current_platform.is_xpu()
+        else get_configs_compute_bound()
+    )
```

## 性能声明

Intel Arc Pro B70，fp8；baseline 为 kernel 内置默认配置（`BLOCK_SIZE_M=64, GROUP_SIZE_M=32, num_warps=4, num_stages=2`），tuned 使用生成的 JSON。Per-batch 扫描（10 个大小）对 DeepSeek-V2-Lite(tp=1) 两个形状的几何平均加速为 2.73x（`N=576,K=2048`）、2.00x（`N=22016,K=2048`）；表中 batch=1 分别 5.08x / 2.67x、batch=16 分别 5.17x / 3.28x。数值校验：每个配置对照 FP32 参考 GEMM（相对 L2 误差 < 2e-2），DeepSeek-V2-Lite 60/60、Llama-3.1-8B 60/60、Qwen3-8B 新形状 40/40 通过。端到端（单张 B70，`yiliu4/DeepSeek-V2-Lite-Chat-Block-wise-FP8-Test-Only`，`--linear-backend triton`，512 输入 /128 输出 /32 prompts /inf 请求率，1 次 warm-up + 3 次 seeds 2-4）：DeepSeek-V2-Lite-Chat 109.63 → 140.88 tok/s（1.285x）、Llama-3.1-8B-Instruct 36.99 → 84.20（2.276x）、Hunyuan-MT-7B 37.04 → 84.55（2.282x）、Qwen3-8B 36.29 → 82.89（2.284x）。正文注明默认 oneDNN(auto) 路径在这四个模型上仍比 tuned Triton 快约 2-5.7x。

## 验证与回归风险

Test plan 列出：在 XPU 上 `benchmark_w8a8_block_fp8.py tune --tp-size <tp> --batch-size <bs>` 调优、用 fp32 参考逐个校验 winner、确认 serving 加载精确 JSON（`fp8_utils.py` 日志）、在调优 batch 网格上对比默认/调优配置、以相同负载与新 engine 对比 heuristic/tuned 端到端。数值校验结果见性能声明。正文明确要求保持 CUDA/ROCm 行为不变，XPU 剪枝由 `current_platform.is_xpu()` 保护；并说明本 PR 价值在于调优管线与配置，而非主张 Triton 取代 B70 上的 oneDNN。

## 关联

正文 Purpose 明确说明沿用 #53065（XPU Triton fused MoE tuning）的模式；正文 "Note on scope" / 结果讨论涉及默认 oneDNN 路径对比，但未给出其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
