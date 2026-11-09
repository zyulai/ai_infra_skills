# RFC #48480: [RFC]: Make Triton kernel unit tests hardware-agnostic and cover untested kernels

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48480](https://github.com/vllm-project/vllm/issues/48480) |
| 作者 | adobrzyn |
| 创建日期 | 2026-07-13 |
| 状态 | 🟢 open |
| labels | stale |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Attention / Kernel |

## 一句话结论
提议把 vLLM 的 Triton kernel 单元测试改为硬件无关,并为缺乏专属测试的 kernel 补齐隔离测试,每个 PR 覆盖一个 kernel。

## 提案动机
vLLM 附带大量 Triton kernel(MoE、Mamba/SSD、GDN/FLA、KDA、linear/flash attention、INT8/FP8 量化、归一化),Triton 天生可移植、这些 kernel 已在 Intel GPU(XPU)等非 NVIDIA 后端经同一 Triton 路径编译执行,但测试未反映这一点,存在两个具体缺口:1) 测试钉死 CUDA——硬编码 device="cuda"、torch.set_default_device("cuda") 或调用 CUDA-only host API(torch.cuda.get_device_capability() 等),纯 Triton kernel 在 ROCm/XPU 上跑得对却没有测试(报错或被跳过);2) 很多 kernel 只有间接覆盖或没有测试——如 Mamba-2 SSD chunk kernel 只能经 mamba_chunk_scan_combined_varlen 触达,若干 DeepGEMM EP-scatter/gather 与 GDN/FLA helper kernel 无专属测试;回归发生时故障远离成因、难以定位。作者评审了 Triton kernel 面,选出广泛使用模型族热路径上最重要的 kernel。

## 提出的方案
- 每个 PR 针对一个 kernel(或紧耦合 kernel 族),在描述中说明覆盖对象与理由;两种 PR 形态:
- "Adjusted":把既有 CUDA 绑定测试的 device 字面量换成 current_platform.device_type;模块级 CUDA skip 换成 is_cuda_alike()/is_xpu() 的 skipif;真正 CUDA-only 的断言(设备能力、stream capture、CUDA-only C++ ops)收进平台检查,CUDA 行为不变
- "New":无专属测试的 kernel 新增独立测试,按生产 wrapper 同样方式直接启动并校验 kernel;按生产所见 shapes/dtypes/flags 参数化以覆盖各编译变体;从第一天起设备无关
- 上游安全保证:CUDA/ROCm 无源行为变化(仅平台检查门控 host 调用)、纯 Triton 不改任何 kernel、数值不变(容差沿用上游先例,如把 ROCm 对 selective_state_update 双倍 bf16 atol 的做法扩展到 XPU)、模式已被上游接受(tests/kernels/test_mhc_kernels.py、test_layernorm.py 及已合并的 #42736、#43307)
- 范围(按 kernel 组,各列代表 PR):MoE(#43407、#44298、#45809)、Mamba/SSD(#42736 已合并、#45016、#45140)、GDN/FLA(#44314、#44315 等)、KDA(#44982、#45694)、DeepGemm EP scatter/gather 与 SiLU-mul(#45921 等)、Attention(#46134、#47862)、INT8/FP8 量化(#45382 等)、Linear/Lightning Attn(#43307 已合并)

## 优化点与收益
正文未给出量化数字;收益为回归就地定位(间接覆盖不能定位回归)、上游 CI 对所有硬件更强;已合并的 #42736 与 #43307 证明该做法无 CUDA 回归。

## 现状与落地
🟢 open(labels 已挂 stale);#47862 与 #47976 因达到 PR 数量上限暂为 draft;每个 PR 在 GPU 与 Intel XPU(Battlemage)上验证,CUDA/ROCm 继续由上游 CI 覆盖,gated 的增量测试不改变这一点。

## 关联
- #42736、#43307(已合并先例)、#43407、#45016、#46134(各组代表 PR)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
