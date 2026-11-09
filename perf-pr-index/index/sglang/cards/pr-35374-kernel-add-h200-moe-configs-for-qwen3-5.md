# PR #35374: [Kernel] Add H200 MoE configs for Qwen3.5 and Qwen3.6

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @vedantjh2 |
| 合并 | 2026-08-27 |
| 分类 | MoE(tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/35374 |

## 一句话结论
为 Qwen3.5/Qwen3.6 FP8 与 Qwen3.6 BF16 添加 H200 Triton 3.6 fused-MoE 调优配置,GSM8K 吞吐最高 +7.05%。

## 动机
H200 上这三个模型没有对应的 Triton 3.6 fused-MoE 调优配置,up/gate 与 down 投影只能用回退(如 up 表兜底 down),大 routed-token 场景下 GEMM tile 参数未调优。纯配置变更:`E=256,N=512`,TP1,FP8 为 block FP8 `[128,128]`,up/gate 与 down 分表。

## 优化点
新增 4 个 JSON 配置文件到 `moe_runner/triton_utils/configs/triton_3_6_0/`:FP8 `fp8_w8a8,block_shape=[128,128]` 的 up/gate 表与独立 down 表(启用 `USE_TMA`,按 routed tokens 1–1024 给出 BLOCK_SIZE_M/N/K、GROUP_SIZE_M、num_warps、num_stages 的最优组合),以及 BF16 的 up/gate 与 decode-aware down 表。Qwen3.6 FP8 checkpoint 与 Qwen3.5 共用同一 lookup key 的表。同时删除了不支持的 Qwen3.8 MoE 条目(Qwen3.8-27B 是 dense hybrid-attention 模型,不能用这些 MoE 表;其 packed-GDN 与 ReplaySSM 候选单独调优后端到端中性,均未收录)。

## 关键改动

```text
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=256,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json 及 _down.json: FP8 up/gate + down 调优表(USE_TMA)
.../E=256,N=512,device_name=NVIDIA_H200.json 及 _down.json: BF16 up/gate + decode-aware down 调优表
```

## 性能声明
单卡 H200、TP1、SGLang v0.5.17、官方 GSM8K bench(5-shot、temp 0、20 题预热、A/B/A 三相各 10 次 200 题、parallel 64):Qwen3.5 FP8 -7.81% 延迟 / +3.82% 吞吐(3151.08→3290.12 tok/s);Qwen3.6 FP8 -6.96% 延迟 / +7.05% 吞吐(2965.50→3128.58 tok/s);Qwen3.6 BF16 中性(-0.08% 吞吐)。Qwen3.6 BF16 在 saturated parallel 128(1319 题)bracket 估计 +2.07% 吞吐。isolated kernel:Qwen3.5 FP8 down 在 1024 routed tokens -17.87%(107.87→88.59us);Qwen3.6 BF16 up/gate 1024 tokens -35.56%、down 1024 tokens -20.92%。另测 Qwen3.8-27B:packed-GDN 与 ReplaySSM-spec kernel 候选均端到端中性被拒;非投机 ReplaySSM 的 no_buffer 配置 +16.06% 吞吐属服务器配置结果,不属本 PR 代码改动。

## 验证与回归风险
验证为 A/B/A 三相 GSM8K 全模型对比 + isolated kernel microbench(附完整复现脚本);serving 日志确认调优文件被 up/gate 与 down 两个投影实际选中。精度:A1/tuned/A2 accuracy 记录在案(Qwen3.5 FP8 0.775/0.768/0.768,Qwen3.6 FP8 0.415/0.425/0.413,BF16 0.518/0.518/0.525;saturated 相位有 output/accuracy 漂移,作者明确为饱和吞吐结果而非延迟/质量声明)。风险:配置按 (E,N,device,dtype,block_shape,triton 版本) 查找,只影响 H200 + Triton 3.6 + 对应模型的组合;BF16 主表端到端中性,收益主要来自 down 表。

## 关联
无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
