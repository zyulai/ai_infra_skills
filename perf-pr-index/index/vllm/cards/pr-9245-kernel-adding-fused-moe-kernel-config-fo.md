# PR #9245: [Kernel] adding fused moe kernel config for L40S TP4

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @bringlein |
| 合并 | 2024-10-11 |
| 分类 | MoE \| tier: major |
| Label | (none) |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/9245 |

## 一句话结论

为 L40S TP4 的 mixtral-8x7b 补 fused_moe Triton 配置，吞吐最多 +10%、延迟约 -6%。

## 动机

PR 正文说明这是为 L40S 上 TP4 的 mixtral-8x7b 添加 `fused_moe` 的 Triton 配置，结果是吞吐最多提升 10%、延迟约改善 6%。正文给出延迟测量方式：用 `benchmarks/benchmark_latency.py` 在 batch size 1/2/4/8/16/32/64 上各跑 100 iters，并列出本 PR 与 upstream 基线的 Avg latency 对比；此外还用 fmperf 在稍大的 workload 上复测，称改进更明显。正文未进一步描述补充配置之前的具体瓶颈或为何该设备缺失配置。

## 优化点

新增一个 `fused_moe` 的 Triton 配置文件 `E=8,N=3584,device_name=NVIDIA_L40S.json`，对应 8 experts、N=3584 的 NVIDIA L40S 设备形状，为 `fused_moe` 提供该设备/形状下的调优配置。pack 的 diff 仅含这一个 JSON 配置文件，未改动 kernel 源码。正文说明其余测试通过 `benchmark_latency.py` 完成，未描述配置内部的具体调优项。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_L40S.json: 新增 L40S TP4、E=8 / N=3584 形状的 fused_moe Triton 配置（pack 中唯一改动文件）
```

## 性能声明

正文称"Adding triton configurations for `fused_moe` on L40S (mixtral-8x7b with TP4). Results in up to 10% more throughput, ~6% better latency."，pack 的 regex 性能声认为 `up to 10%`、`10% more throughput`。测试条件：`benchmarks/benchmark_latency.py --model=/models/mixtral-8x7b --input-len 1000 --output-len 50 -tp 4 --num-iters 100`，batch size 依次 1/2/4/8/16/32/64，硬件 NVIDIA L40S（TP4），PR 未注明精度。本 PR 的 Avg latency（秒）：0.8366 / 1.2283 / 1.7394 / 2.3766 / 3.4703 / 5.4072 / 8.9833；对比 upstream commit `dc4aea677ab0520d91ff4979e80340cb5a090095`：0.8259 / 1.2335 / 1.7657 / 2.5269 / 3.7281 / 5.7599 / 9.5335。另用 fmperf 在稍大 workload 下测得"even better improvements"（以截图呈现，未给具体数字）。

## 验证与回归风险

PR 正文未注明测试计划、单元或回归测试结果，也未说明回归风险。改动仅为一个新增的 L40S TP4 配置文件，正文未说明是否影响其他设备、其他模型或其他形状。

## 关联

无。正文仅链接到对比用的 upstream commit `dc4aea677ab0520d91ff4979e80340cb5a090095` 与 fmperf 项目，未给出可关联的 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
