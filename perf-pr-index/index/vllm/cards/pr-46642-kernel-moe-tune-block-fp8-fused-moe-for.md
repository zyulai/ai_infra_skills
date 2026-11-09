# PR #46642: [Kernel][MoE] Tune block-FP8 fused MoE for low-batch decode

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-06-25 |
| 分类 | MoE(major) |
| Label | performance, ready |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,涉及 10 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/46642 |

## 一句话结论

放开 block-FP8 fused MoE 的 N tile 限制,低 batch 下改用 64 宽 N tile 翻倍 thread block 数,decode 速度提升 2.2%-17.5%。

## 动机

block 量化 FP8 fused MoE 在所有 batch size 下都被锁死 `BLOCK_SIZE_N = block_shape[0]`(128)——运行时启发式(`get_default_config`)和 tuner 搜索空间都是如此。decode 时 gate-up GEMM 是 SM-bound:128 宽的 N tile 产生的 thread block 太少,填不满 GPU。这并非 kernel 的硬性要求:Triton kernel 按元素索引 block scale(`offs_bn // group_n`),只要 N tile 能整除量化块就合法。

## 优化点

机制核心是把"N tile 必须是量化块整数倍"的约束改为"tile 与量化块互为整数倍(可整除)":(1) `get_default_config` 对 CUDA 上的 fp8 block-quant 在 M<=8 且 `block_shape[0] % 64 == 0` 时改用 `BLOCK_SIZE_N=64`,num_stages 按 M 取 4(M<=4,内存延迟受限,深流水线隐藏权重加载)或 3(M=8 起转向 occupancy/SMEM bound);ROCm 与更大 batch 不变。(2) `benchmark_moe.py` 的 tuner prune 过滤器同步放宽——此前 64 宽 tile 被静默剔除,任何调优都不可能找到它。(3) 依赖 Hopper swap-AB kernel(PR #36559,已在 main)保证 64 宽 tile 在最小 batch 下仍高效,重调受影响的 H100/H200 配置低 batch 档(E=256,N=256 上 N=32 仍更优)。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/fused_moe.py: get_default_config 低 batch 用 N=64 并调 num_stages
benchmarks/kernels/benchmark_moe.py: tuner 搜索空间允许整除量化块的更窄 tile
vllm/model_executor/layers/fused_moe/configs/*H100/H200*block_shape=[128,128].json: 重调低 batch 档(8 个文件)
```

关键片段(get_default_config):

```python
elif M <= 8 and block_shape[0] % 64 == 0:
    block_n = 64
    num_stages = 4 if M <= 4 else 3
```

## 性能声明

在 H100 80GB HBM3(与 H200 同为 132-SM,swap-AB 激活)、topk=8,经 `benchmark_moe.benchmark_config` 对全部 29 个改动的 config key 重测:0 回归,提升 +2.2% 至 +17.5%。代表性数据:H200 E=384,N=128 M=1 42.1µs → 38.2µs(+9.2%)、M=4 70.0 → 63.7µs(+8.9%)、M=8 98.1 → 90.5µs(+7.8%)、M=16 154.9 → 144.0µs(+7.0%);H200 E=128,N=768 M=1 61.2 → 50.5µs(+17.5%);H200 E=256,N=256 M=1 48.8 → 44.3µs(+9.1%,N=32);H100 E=512,N=256 M=1 32.8 → 30.4µs(+7.4%)。swap 路径数值验证:swap-on vs swap-off 相对差 6.5e-3,在 fp8 容差内。受益范围:Hopper block-F8 TP 路径(oracle 选 TritonExperts);EP(FlashInfer CUTLASS)与 Blackwell(E8M0 DeepGEMM)路径不受影响。作者注明使用了 Claude 辅助分析/测试/编辑。

## 验证与回归风险

验证方式为上述逐 key 重测(0 回归)与 swap 路径数值对照;PR 未注明新增 pytest 用例。风险面:`get_default_config` 是无调优配置时的回退路径,非 Hopper 或无 swap-AB 的环境若命中 M<=8 分支可能不达预期(代码限定 CUDA 且 ROCm 走旧分支);tuner 放宽后未来新调出的配置依赖 kernel 的 per-element scale 索引正确性。

## 关联

前序:PR #36559(swap-AB kernel,本 PR 效率前提)。同区域:PR #31407、PR #38086(configs 调优)。PR 未注明其他后续编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
