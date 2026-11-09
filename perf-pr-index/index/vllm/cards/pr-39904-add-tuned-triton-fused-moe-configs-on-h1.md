# PR #39904: Add tuned triton fused_moe configs on H100 for gpt-oss

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zhangxin81 |
| 合并 | 2026-04-28 |
| 分类 | Attention / Kernel(tier: major) |
| Label | ready, gpt-oss, verified |
| Diff | diffstat 缺失(素材未提供 +/− 统计);diff 素材完整可用(单文件新增) |
| 链接 | https://github.com/vllm-project/vllm/pull/39904 |

## 一句话结论

为 gpt-oss 120b 新增 H100 tuned fused_moe 配置,kernel 时间几何平均降约 3.35%。

## 动机

gpt-oss 120b 的 fused MoE(E=128,N=2880)在 H100 上此前没有 per-shape 配置文件,fused_moe kernel 落到默认配置;对该形状显式调参可以拿到稳定但温和的 kernel 级收益。

## 优化点

纯配置新增:产出 E=128,N=2880,device_name=NVIDIA_H100_80GB_HBM3.json(triton_version 3.6.0),覆盖 18 个 batch size 档位(1 到 4096)。小 batch(1-256)段以 BLOCK_SIZE_M=16 为主、BLOCK_SIZE_N 在 32/64 间选择;大 batch(1024+)切到 BLOCK_SIZE_M=64/128、BLOCK_SIZE_N=128/256;num_stages 在 3-5、num_warps 在 4-8 间按档位选定。不改 kernel 代码,运行时按配置查表生效。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/configs/E=128,N=2880,device_name=NVIDIA_H100_80GB_HBM3.json: 新增 18 档 Triton fused_moe 配置
```

```json
"8":   { "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 128,
         "GROUP_SIZE_M": 64, "num_warps": 4, "num_stages": 4 },
"2048": { "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
          "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 5 }
```

## 性能声明

fused_moe kernel benchmark,18 个匹配的 batch-size 点,引入配置文件 vs 默认配置:几何平均加速约 1.035x,等价 kernel 时间平均减少约 3.35%。最大收益:bs=8 −8.09%(1.088x)、bs=96 −7.47%(1.081x)、bs=128 −6.68%(1.072x);最小收益:bs=2 −0.12%、bs=2048 −0.54%、bs=3072 −0.83%。PR 正文未注明 dtype/精度等测试条件(硬件与形状见配置文件名:NVIDIA_H100_80GB_HBM3,E=128,N=2880;模型为 gpt-oss 120b)。

## 验证与回归风险

验证为 kernel benchmark 对比;label 含 verified,无单测(纯配置文件)。风险很低:按 device_name+shape 命中,仅影响 H100 上 gpt-oss 120b 的 MoE 形状;配置绑定 triton_version 3.6.0,Triton 升级后可能需要重调。

## 关联

同模型卡:#28345(gpt-oss cudagraph capture size)。同分类调参卡:#19205(H20-3e DSv3 配置)、#23748(block-FP8 GEMM H100/H200)、#14068(tuning 基础设施)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
