# PR #24266: QWEN3 Coder Fused MoE kernels Optimization configs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @samanamp |
| 合并 | 2025-09-04 |
| 分类 | MoE（tier: major） |
| Label | performance, ready, qwen |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾未见 "N file(s) dropped from diff" 标注，diff 段含 9 个 fused_moe config JSON 新增文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/24266 |

## 一句话结论

为 Qwen3-Coder-480B-A35B-FP8 的 fused MoE 补齐 B200/GB200/H100 tuned config，E2E Output tok/s +6.5%、P99 TPOT 改善 5.9×。

## 动机

正文 Purpose 仅写 "Optimize QWEN3 Coder Fused MoE kernels Optimization configs"，未展开说明改动前的具体瓶颈。从 diff 看，PR 为 QWEN3 Coder 的 fused MoE 在 NVIDIA B200 / GB200 / H100 上补齐 `fp8_w8a8`、`block_shape=[128,128]` 的 tuned 配置（E/N 组合为 E=160,N=640；E=20,N=2560；E=40,N=2560，各 3 个设备共 9 个文件）。

## 优化点

改动全部是新增 9 个 `vllm/model_executor/layers/fused_moe/configs/*.json` 配置文件，未提供算法或 kernel 代码改动。每个 JSON 按 batch size（键 "1"、"2"、"4"、"8"、"16"、"24"、"32"…）给出 tuned 的 `BLOCK_SIZE_M`、`BLOCK_SIZE_N`、`BLOCK_SIZE_K`、`GROUP_SIZE_M`、`num_warps`、`num_stages`，供 fused MoE Triton kernel 在不同 batch 下选用分块配置。正文未说明这些配置是如何搜索或调优得到的，也未给出与旧配置的具体差异。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/configs/E=160,N=640,device_name=NVIDIA_{B200,GB200,H100},dtype=fp8_w8a8,block_shape=[128,128].json: E=160,N=640 的 3 个设备配置
vllm/model_executor/layers/fused_moe/configs/E=20,N=2560,...json: E=20,N=2560 的 3 个设备配置
vllm/model_executor/layers/fused_moe/configs/E=40,N=2560,...json: E=40,N=2560 的 3 个设备配置
```

```json
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

测试条件（Test Plan / Test Result）：1) Single benchmark `python benchmarks/kernels/benchmark_moe_old.py --model $MODEL --dtype "fp8_w8a8" --tp 4`，结果为 3 张截图，PR 未给出可读数值。2) Model E2E：`MODEL=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`，server `python -m vllm.entrypoints.openai.api_server --model $MODEL --enable-expert-parallel --max-model-len=65000 -tp 4 --port 8101 --gpu_memory_utilization=0.95`；bench `python benchmarks/benchmark_serving.py --backend vllm --model $MODEL --dataset-name random --random-input-len 7500 --random-output-len 7500 --max-concurrency 128 --num-prompts 512 --port 8101`。BEFORE / AFTER：

| Metric | BEFORE | AFTER |
|---|---|---|
| Successful requests | 256 | 256 |
| Maximum request concurrency | 128 | 128 |
| Benchmark duration (s) | 961.48 | 934.94 |
| Output token throughput (tok/s) | 1891.57 | 2014.25 |
| Total Token throughput (tok/s) | 3887.42 | 4066.78 |
| Mean TTFT (ms) | 18056.20 | 18337.78 |
| Median TTFT (ms) | 8936.43 | 8664.39 |
| P99 TTFT (ms) | 56461.27 | 56497.93 |
| Mean TPOT (ms) | 73.51 | 60.35 |
| Median TPOT (ms) | 63.22 | 59.73 |
| P99 TPOT (ms) | 490.19 | 83.55 |
| Mean ITL (ms) | 61.99 | 58.93 |
| Median ITL (ms) | 55.98 | 52.33 |
| P99 ITL (ms) | 490.46 | 489.33 |

正文结论（Purpose 与 Test Result 段照抄）："We see 6.5% in `Output token throughput (tok/s)`. 17% in `Median TPOT (ms)` and 5.9x improvement in P99 TPOT (ms)."

## 验证与回归风险

验证：PR 给出 Single benchmark 的 3 张截图（无可读数值）与上述 Model E2E 的 BEFORE/AFTER serving benchmark 结果。PR 未注明新增/依赖的具体测试，也未说明这些 config 对其他模型、其他 GPU、其他 dtype/量化组合的影响或回归风险。

正文 Test Plan 还给出 Single benchmark 命令 `python benchmarks/kernels/benchmark_moe_old.py --model $MODEL --dtype "fp8_w8a8" --tp 4`，其 Test Result 以三张截图呈现、未给出可读数值；Model E2E 的 BEFORE/AFTER 均为 256 successful requests、128 maximum request concurrency，Total input tokens 均为 1918985、Total generated tokens 为 1818711→1883193。checklist 中 documentation update 与 release notes 两项均未勾选。9 个 config 文件的命名覆盖 `E=160,N=640`、`E=20,N=2560`、`E=40,N=2560` 三种专家数与宽度组合，以及 B200、GB200、H100 三种设备，dtype 均为 `fp8_w8a8`、`block_shape=[128,128]`；每个文件按 batch size 键（"1"、"2"、"4"、"8"、"16"、"24"、"32"…）给出分块参数。另：BEFORE 的 Benchmark duration 为 961.48 s、AFTER 为 934.94 s，两者 Request throughput 同为 0.27 req/s，Total input tokens 均为 1918985；正文 Purpose 段与 Test Result 段的结论措辞完全一致。

## 关联

无（PR 正文与 diff 未给出其他 PR 编号或前序/后续线索）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
