# PR #31407: Add Fused MoE Triton kernels for GLM-4.5-Air, GLM-4.5v, GLM-4.6v on 2x RTX Pro 6000

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mratsim |
| 合并 | 2025-12-28 |
| 分类 | MoE(major) |
| Label | performance, ready |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,新增 2 个 JSON 配置文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/31407 |

## 一句话结论

为 2x RTX Pro 6000(Blackwell)上的 GLM-4.5-Air / GLM-4.5V / GLM-4.6V 新增 Fused MoE Triton kernel 调优配置,加速 fp8 MoE GEMM。

## 动机

Fused MoE Triton kernel 的 tile/warp/stages 配置按 (E, N, device, dtype) 键查表。GLM-4.5-Air 及其视觉/后续变体在 NVIDIA RTX Pro 6000 Blackwell Workstation Edition 上没有对应的 fp8_w8a8 配置文件,kernel 只能落到默认参数,无法在该硬件上发挥吞吐。作者在 issue #26838 的语境下补齐这组硬件 × 模型的调优结果。

## 优化点

本 PR 不改 kernel 代码,只新增两个调优配置 JSON:`E=128,N=704` 与 `E=129,N=704`(N=129 对应 GLM-4.5-Air;128 覆盖相关变体),device_name 为 `NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition`,dtype 为 `fp8_w8a8`。每个文件按 M(0-1 token 到 4096)分档给出 `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps`、`num_stages`,例如 M=1 用 16x64x64、num_stages=4,大 M(512+)转向 64-256 宽 tile 并降 stages。两个文件内容相同,E=128 与 E=129 共享同一组参数。配置由 SGLang tuner 产出,运行时按 batch size 自动选档。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/configs/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json: GLM-4.5-Air 的 MoE kernel 调优表(M=1..4096)
vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json: E=128 变体(GLM-4.5V/4.6V 等)同一调优表
```

关键片段(E=129 配置,M=1 档):

```json
"1": {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 4
}
```

## 性能声明

PR 正文未给出量化数字,仅称 "Tested locally on vLLM and in SGLang"、结果 "It works",并指向 issue #26838 下的评论(素材中未包含该评论内容)。硬件为 2x RTX Pro 6000,dtype fp8_w8a8;具体加速比 PR 未注明。

## 验证与回归风险

纯配置新增,按文件名键匹配命中,不影响其他 (E, N, device, dtype) 组合;回归风险主要在配置若与硬件实际不符可能选到次优 tile。PR 未注明新增/依赖的自动化测试。

## 关联

MoE 分类下的调优配置类 PR(同模式:按设备新增 configs JSON);关联 issue #26838。PR 未注明其他前序/后续 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
