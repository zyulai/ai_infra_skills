# PR #55511: [Kernel] Add fused MoE tuned config for E=256,N=512 on NVIDIA A100 80GB PCIe

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @bakiburakogun |
| 合并 | 2026-09-06 |
| 分类 | MoE(major) |
| Label | ready, nvidia, verified |
| Diff | 素材未提供 diffstat 数字;diff 为新增单个调优 JSON 配置文件(147 行) |
| 链接 | https://github.com/vllm-project/vllm/pull/55511 |

## 一句话结论

为 A100 80GB PCIe 补上 E=256、N=512(Qwen3.5-122B-A10B,TP=2,bf16)的 fused MoE Triton 调优配置,prefill 级 batch kernel 时间下降 18–19%。

## 动机

fused MoE Triton kernel 按 `(E, N, device)` 查找调优配置。Qwen3.5-122B-A10B 在 TP=2 下展开为 E=256 experts、moe_intermediate_size=1024(N=512/rank)的形状,仓库里只有 H100/H20 条目而缺 A100 80GB PCIe,A100 PCIe 部署只能回落到默认启发式配置,在大 batch 下明显次优。

## 优化点

不改 kernel 代码,纯新增一份针对该设备/形状的调优结果:在 8×A100 80GB PCIe 主机(无 NVLink,vLLM v0.28.0 镜像,Ray workers)上用标准 tuning 脚本 `benchmark_moe.py --model Qwen/Qwen3.5-122B-A10B --tp-size 2 --tune` 全量搜索,为 batch 1–4096 的各档位落定 `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps`、`num_stages`(小 batch 用小 BLOCK_M=16/多 stages,prefill 级 batch 用 BLOCK_M=64–128 + BLOCK_N=256 等),文件名按 `E=256,N=512,device_name=NVIDIA_A100_80GB_PCIe.json` 约定被 vLLM 启动时自动拾取。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_A100_80GB_PCIe.json: 新增 19 个 batch 档位的 Triton kernel 调优参数(triton_version 3.7.1)
```

关键片段(配置内容示意):

```json
"1024": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256,
         "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16,
         "num_warps": 8, "num_stages": 3}
```

## 性能声明

`benchmark_moe.py` kernel 时间(bf16、TP=2 形状、单张 A100 80GB PCIe、vLLM v0.28.0、Triton 3.7.1,default vs tuned):
- batch 1:73.35→67.54 µs(−7.9%);batch 4:−3.1%;batch 16:−2.8%;batch 64:−2.5%。
- batch 256:1607.70→1468.64 µs(−8.6%);batch 1024:2102.24→1694.01 µs(−19.4%);batch 4096:3300.12→2719.18 µs(−17.6%)。
- 即 decode 档位 2–9%、prefill 档位约 18–19% 提速,所有测量点均改善。
- 全量调优耗时 10292 s(约 2 小时 51 分)跨 8 GPU;sanity 确认 JSON 可加载且 vLLM 启动日志拾取该配置。

## 验证与回归风险

验证为 `json.load` 结构检查、tuner 产生的 batch size key 核对与启动日志拾取确认;diff 未新增单元测试。风险面:配置与 Triton 3.7.1、vLLM v0.28.0 调优环境绑定,Triton 版本演进后可能需要重调;仅覆盖 E=256/N=512/bf16/A100 PCIe 这一个形状,其他形状或 dtype 不受影响(继续走各自配置或默认)。

## 关联

同类 tuned-config PR:#28373(AMD Triton MoE tuning)、#34130(int4 W4A16 fused MoE benchmark/config)、#53065(Intel XPU MoE 调优)、#62763(H200 FP8 fused MoE 配置)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
