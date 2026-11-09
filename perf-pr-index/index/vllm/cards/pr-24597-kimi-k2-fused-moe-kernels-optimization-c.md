# PR #24597: Kimi K2 Fused MoE kernels Optimization configs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @samanamp |
| 合并 | 2025-09-11 |
| 分类 | MoE \| tier: major |
| Label | performance, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/24597 |

## 一句话结论

为 Kimi K2 Fused MoE 新增调优 config，Output token throughput 提升 17.8%（553.73 → 652.64 tok/s）。

## 动机

PR 正文 Purpose 仅写 "Optimize Kimi K2 Fused MoE kernels Optimization configs"，说明执行层面在 `Output token throughput (tok/s)` 上看到 17.8% 改善，且改善主要来自 `Inter-token Latency`。正文未描述改动前的具体瓶颈或为何需要新配置。

## 优化点

改动为新增 Fused MoE 的调优配置文件（JSON），位于 `vllm/model_executor/layers/fused_moe/configs/`，命名形如 `E=384,N=<128|256>,device_name=<NVIDIA_B200|NVIDIA_GB200|NVIDIA_H200>,dtype=fp8_w8a8,block_shape=[128,128].json`。每个文件按 batch size（"1"、"2"、"4"、"8"、"16"、"24"、"32"、"48"、"64"、"96" 等键）给出 `BLOCK_SIZE_M`、`BLOCK_SIZE_N`、`BLOCK_SIZE_K`、`GROUP_SIZE_M`、`num_warps`、`num_stages` 组合，例如 batch=1~8 采用 `BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=4`，较大 batch 调整 `BLOCK_SIZE_K=256` 或 `num_stages=3`。覆盖 B200 / GB200 / H200 三种 device 与 N=128 / N=256 两种专家宽度。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/configs/E=384,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json: 新增 B200/N=128 调优配置
vllm/model_executor/layers/fused_moe/configs/E=384,N=128,device_name=NVIDIA_GB200,...json: 新增 GB200/N=128 调优配置
vllm/model_executor/layers/fused_moe/configs/E=384,N=128,device_name=NVIDIA_H200,...json: 新增 H200/N=128 调优配置
vllm/model_executor/layers/fused_moe/configs/E=384,N=256,device_name=NVIDIA_B200,...json: 新增 B200/N=256 调优配置
vllm/model_executor/layers/fused_moe/configs/E=384,N=256,device_name=NVIDIA_GB200,...json: 新增 GB200/N=256 调优配置
```

```text
{
    "1": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 4
    },
```

## 性能声明

正文："Execution improvement: We see 17.8% in `Output token throughput (tok/s)`. The bulk of improvement is in `Inter-token Latency`." 测试条件：single benchmark `python benchmarks/kernels/benchmark_moe_old.py --model $MODEL --dtype "fp8_w8a8" --tp 4`；Model E2E server `MODEL=moonshotai/Kimi-K2-Instruct`，`python -m vllm.entrypoints.openai.api_server --model $MODEL --enable-expert-parallel --max-model-len=65000 -tp 4 --port 8101 --gpu_memory_utilization=0.95`，bench `benchmark_serving.py --backend vllm --dataset-name random --random-input-len 7500 --random-output-len 7500 --max-concurrency 128 --num-prompts 512`。E2E BEFORE → AFTER：Benchmark duration 499.10 → 424.97 s，Request throughput 0.40 → 0.47 req/s，Output token throughput 553.73 → 652.64 tok/s，Total Token throughput 3558.65 → 4181.70 tok/s，Mean TTFT 12740.87 → 13136.72 ms，Mean TPOT 272.12 → 269.90 ms，Mean ITL 62.89 → 52.96 ms，Median ITL 57.45 → 47.41 ms。PR 未注明 E2E 所用硬件型号（config 标注 fp8_w8a8）。

## 验证与回归风险

Test Plan 给出 single benchmark 与 Model E2E 两条命令；Test Result 附三张 single benchmark 截图（图片链接，无可读数值）与 BEFORE/AFTER 的 Serving Benchmark Result 文本。正文称改善主要在 Inter-token Latency，而 Mean TTFT、Mean TPOT 变化很小（TTFT 12740.87 → 13136.72 ms 略升）。新增文件仅为特定 device / shape 的 config JSON，正文未说明对其他硬件、模型或精度路径的影响，也未给出新增单元测试。正文 Checklist 中 Purpose、Test Plan、Test Result 三项勾选，文档与 release notes 更新项未勾选；BEFORE/AFTER 两份 Serving Benchmark Result 还列出 Median/P99 TTFT、Median/P99 TPOT、Median/P99 ITL 等分项，其中 Mean TTFT 由 12740.87 升至 13136.72 ms、P99 TTFT 由 21087.93 升至 22379.70 ms。正文在 Test Result 中把三张单算子基准截图与完整的端到端服务基准文本并列给出，并说明单算子基准命令使用 fp8_w8a8 精度与 TP=4，端到端服务启用 expert parallel、最大模型长度 65000、随机数据集且输入与输出长度均为 7500。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
