# PR #23504: [Perf] Add Triton config for DeepSeek V3 FP8 EP32 H200

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @minosfuture |
| 合并 | 2025-08-25 |
| 分类 | Quantization |
| Label | performance, ready, deepseek |
| Diff | diffstat 缺失(pack 中为 +None/-None, None files);diff 含 4 个文件(2 个新 JSON config、1 个新 README、1 个 benchmark 脚本小修) |
| 链接 | https://github.com/vllm-project/vllm/pull/23504 |

## 一句话结论

为 DeepSeek V3 在 H200 上 DP8TP4EP32 的 FP8 block-quant GEMM 与 fused MoE 落盘调优 Triton config,消除 "Performance might be sub-optimal" 回退到默认 config 的告警。

## 动机

在 H200 上以 DP8/TP4/EP32 部署 DeepSeek V3 时,vLLM 找不到匹配形状的 kernel config,只能回退默认配置并告警:一处是 `fp8_utils.py` 找不到 `N=2112,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json`(dense W8A8 block FP8 GEMM),另一处是 `fused_moe.py` 找不到 `E=8,N=2048,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json`(MoE)。默认 config 在这些形状上性能次优。

## 优化点

本 PR 不改任何 kernel 代码,纯粹补充调优产物:新增 fused MoE config(E=8, N=2048, H200, fp8_w8a8, block_shape=[128,128]),覆盖 batch size 1–8192 共 19 档,小 batch 档 BLOCK_SIZE_M=16、大 batch 档多为 64,num_warps 基本为 4;新增 W8A8 block FP8 GEMM config(N=2112, K=7168, H200),含 2048/3072/4096 三档(M 档)。另在 `quantization/utils/configs/` 下新增 README,指向 `benchmarks/kernels/` 下的 config 生成脚本;顺带把 `benchmark_w8a8_block_fp8.py` 的 `import tqdm` 修为 `from tqdm import tqdm`。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/configs/E=8,N=2048,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json: 新增 MoE tuned config(19 档)
vllm/model_executor/layers/quantization/utils/configs/N=2112,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json: 新增 dense FP8 GEMM tuned config(3 档)
vllm/model_executor/layers/quantization/utils/configs/README.md: 说明 config 由 benchmarks/kernels/ 脚本生成
benchmarks/kernels/benchmark_w8a8_block_fp8.py: 修 tqdm import
```

示例(MoE config,batch=64 档):

```json
"64": {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 4
}
```

## 性能声明

测试条件:`deepseek-ai/DeepSeek-V3-0324`,DP8TP4EP32,`benchmark_serving.py`,random 数据集,`--ignore-eos --num-prompts 4096 --request-rate inf --random-input-len 2048 --random-output-len 1024 --max-concurrency 2048`,无 prefix caching:

- tuned vs untuned:输出吞吐 5630.83 vs 5598.65 tok/s,总吞吐 16872.50 vs 16776.08 tok/s,Mean TTFT 56772.23 vs 56991.46 ms,Mean TPOT 305.71 vs 307.65 ms——提升约 0.5%–0.6%,作者未就此给出加速比结论。
- PR 还保留了一组带 prefix caching 的历史结果并标注 "pls ignore; kept here just for records":prefix caching 场景下输出吞吐 7164.09→11483.21 tok/s(benchmark duration 292.73→182.63 s),Mean TTFT 34949.84→7982.04 ms,Mean TPOT 247.35→170.09 ms(2,048 prompts)。

## 验证与回归风险

无新增测试;验证即上述 serving benchmark 对比。风险面极小:两个 JSON config 仅在匹配的形状(DeepSeek V3 的 E=8/N=2048 MoE、N=2112/K=7168 dense FP8)与 NVIDIA H200 上被选中,其他形状/硬件仍走原路径;benchmark 脚本 import 修复不影响运行时。

## 关联

同类型 tuned-config PR:#20516(为 FP8 E=16 MoE 添加 B200 config)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
