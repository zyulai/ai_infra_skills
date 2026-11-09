# PR #28200: [Kernel][Model] Tune fused_moe Triton configs for MiniMax-M2 on H100

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @minatoaquaMK2 |
| 合并 | 2025-11-06 |
| 分类 | Attention / Kernel(major) |
| Label | (none) |
| Diff | diffstat 缺失(素材包未提供 +x/-x);diff 仅含 1 个新增配置 JSON |
| 链接 | https://github.com/vllm-project/vllm/pull/28200 |

## 一句话结论

为 MiniMax-M2 新增 H100 fp8_w8a8 fused MoE Triton 调优配置,vllm bench serve 输出吞吐 +2.3~4.8%、Mean TTFT −15.1%。

## 动机

MiniMaxAI/MiniMax-M2(E=256, N=384,block_shape [128,128],fp8_w8a8)在 H100 上缺少专属 fused MoE Triton 配置,内核按默认参数运行效率不足。用 MoE tuning benchmark 脚本为该模型生成 tuned 配置。

## 优化点

纯配置新增,无代码改动:按 M=1..4096 共 18 档(moe runner 触发的 token 数分桶)给出 BLOCK_SIZE_M/N/K、GROUP_SIZE_M、num_warps、num_stages。小 M 用 BLOCK_SIZE_M=16、BLOCK_SIZE_N=128;M≥512 逐步加大 BLOCK_SIZE_M(32→64);num_stages 在 3–5 间按档选择,triton_version 标注 3.5.0。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/configs/E=256,N=384,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json(新增)
```

## 性能声明

vllm bench serve,MiniMaxAI/MiniMax-M2,random 数据集,input 2048 / output 1024,并发 10,100 prompts(硬件按配置文件名推断为 NVIDIA H100 80GB HBM3,正文未另行注明):Output token throughput 554.22→566.84 tok/s(+2.3%);Peak output 620.00→650.00(+4.8%);Total token throughput 1760.57→1798.96(+2.2%);Mean TTFT 304.97→258.76 ms(−15.1%);Median TPOT 17.00→16.36 ms(−3.8%);Median ITL 16.57→15.90 ms(−4.0%);基准时长 169.77s→166.22s。

## 验证与回归风险

无新增测试(纯配置);配置不影响数值结果。风险限于该 (E, N, device, dtype, block_shape) 键命中的 fused MoE 路径,其他模型/硬件/dtype 不受影响;Median TTFT 一项 +30.0%(124.20→161.44 ms)为回退点。

## 关联

无(pack 未列出关联 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
