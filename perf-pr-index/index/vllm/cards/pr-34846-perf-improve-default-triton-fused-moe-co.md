# PR #34846: [Perf] Improve default triton fused moe configs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-02-24 |
| 分类 | MoE(major) |
| Label | performance, ready |
| Diff | diffstat 缺失(diff 全文已提供,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/34846 |

## 一句话结论

重写无调优配置文件时 Triton fused MoE 的默认 kernel 参数:batch 感知的 tile/调度,多专家模型大 batch 下相对旧默认最高 2x 加速且无回退。

## 动机

当找不到 per-model/device 的 tuned config JSON 时,fused MoE 回退到硬编码默认值,而这些默认值缺乏考量:`BLOCK_SIZE_K=32` 仅匹配不到 1% 的 tuned 配置,`num_warps`/`num_stages` 从未显式设置,且完全没有 batch size 感知——decode 小 batch 与 prefill 大 batch 用同一套 tile,性能差距显著。

## 优化点

新默认值从分析仓库全部 298 个 tuned config 的模式推导。`get_default_config` 通用路径改为:BLOCK_SIZE_M 随 batch 16→32→64→128 分级(M≤32/≤96/≤512/更大);BLOCK_SIZE_N 在 M≤64 时 64、否则 128;BLOCK_SIZE_K 取 128(fp8_w8a8 或 M≤64)否则 64;GROUP_SIZE_M 按 tokens-per-expert 比率(>128 时 16 否则 1,多专家下每 expert token 少、分组无益);num_warps 按 M≤128 取 4/8;num_stages 小 batch 4、大 batch 3、ROCm 恒 2(规避 OutOfResources)。FP8 block 量化路径单独调:decode 类小 batch(M≤64)用 BLOCK_SIZE_M=16、M≤16 时 GROUP_SIZE_M=1(大 batch 时 scale 访存受益于 L2 tile 复用才开分组)。另新增 `benchmarks/kernels/benchmark_moe_defaults.py` 脚本,内置常见模型形状,同表对比 tuned/旧/新三套配置并标记 >5% 回退。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/fused_moe.py: get_default_config 重写(batch 感知分级 + 显式 num_warps/num_stages)
benchmarks/kernels/benchmark_moe_defaults.py: 新增对比脚本(tuned vs old vs new,16 种模型形状)
```

batch 感知 tile 节选:

```python
if M <= 32: block_m = 16
elif M <= 96: block_m = 32
elif M <= 512: block_m = 64
else: block_m = 128
tokens_per_expert = M // max(E, 1)
group_m = 16 if tokens_per_expert > 128 else 1
```

## 性能声明

全部在 H100 上用新脚本测(kernel 时间,us)。Mixtral fp8(E=8,N=7168,K=4096,topk=2)示例:batch 1024 时 1799.26→528.32 us(0.29x),batch 4096 时 6240.76→1740.57 us(0.28x)。多专家模型汇总(new/old,越低越好):batch 1-8 约 0.89-1.02x,batch 16-128 约 0.74-1.08x,batch 256-4096 约 0.49-1.00x;PR 称多专家模型大 batch 下最高 2x 加速、无回退。相对 tuned config:worst New/Tuned 1.03-1.12x,典型 0.93-1.04x。FP8 block 量化路径中性或略好(GROUP_SIZE_M 修正后无回退)。精度评估未做(纯 kernel 配置改动,Test Plan 为跑现有 CI)。

## 验证与回归风险

依赖现有 CI;新增 benchmark 脚本非测试。风险面:`get_default_config` 是所有 Triton fused MoE 的兜底路径,ROCm `num_stages=2` 约束保留;batch 边界(M=32/64/96/128/512)处配置跳变可能造成个别形状轻微波动,脚本以 5% 阈值标记回归;有 tuned config 的用户不受影响。

## 关联

同区域:fused MoE tuned config 生态(如 #29757 附带 H200 配置文件)、#31837 等 MoE kernel 优化;具体前序/后续编号素材未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
