# PR #13797: [1/N] Optimize All Reduce - Benchmark different AR operations

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-01-26 |
| 分类 | Parallelism / Communication |
| Label | performance, Multi-modal, benchmark, run-ci, vlm |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/13797 |

## 一句话结论

系列第一弹:新增 all-reduce 微基准工具对比 SGLang Custom AR 与 Torch Symm-Mem 各消息尺寸延迟,为按流量选 AR 算法铺路。

## 动机

面向 issue #13363 的 step-4。实测部分 MoE 模型(尤其 VLM+MoE)中 all-reduce 占用大量 GPU 周期:Qwen3-VL-MoE TP8 的 torch profiling 显示 prefill 阶段 AR 占 GPU cycles 超过 80%(模型小、激活专家少、rank 间计算与传输数据量小,而 RS+AG 类 AR 的固定开销占比被放大)。decode 阶段 AR 开销显著下降,原因之一是 Piecewise CUDA Graph(另一 PR 方向),之二是 decode 数据量更小。需要基准工具让用户在不同 TP size 与环境下选择最优 AR:小数据用 one-shot 等轻量实现,大数据用 two-shot。

## 优化点

新增 `benchmark/kernels/all_reduce/benchmark_all_reduce.py`:torchrun 2/4/6/8 卡运行,对 32KB~64MB 共 12 个消息尺寸(fp16)分别基准 SGLang `CustomAllreduce` 与 `TorchSymmMemCommunicator`,<=1MB 50 iters、>1MB 20 iters,barrier + CUDA sync 计时,跨 rank all-gather 求平均,不可用实现记 None 跳过。附带修正 `custom_all_reduce.py` 的 `should_custom_ar` 边界:`inp_size < max_size` 改为 `<=`(恰好等于 max_size 的消息此前会被错误拒绝)。本 PR 为 [1/N] 系列,不改推理路径;按流量自动选择 one-shot/two-shot AR 留给后续 PR。

## 关键改动

```text
benchmark/kernels/all_reduce/benchmark_all_reduce.py: 新增 AR 微基准工具(351 行)
python/sglang/srt/distributed/device_communicators/custom_all_reduce.py: should_custom_ar 边界 < 改 <=
```

```python
if world_size not in (2, 4, 6, 8):
    print(f"[rank {rank}] WARNING: world_size={world_size} not in supported set (2,4,6,8). "
          "Custom AR may disable itself.", file=sys.stderr)
```

## 性能声明

基准示例(具体 GPU 型号 PR 未注明,4 卡与 8 卡,nccl 2.27.3,fp16):TP4 下 CustomAR 32KB 0.049 ms vs SymmMem 0.090 ms、256KB 0.036 vs 0.050、32MB 0.251 vs 0.312;TP8 下 32KB 0.046 vs 0.056,但 8MB 处 SymmMem 反超(0.107 vs 0.133)、32MB 0.335 vs 0.286。两组数据支撑"小消息 Custom AR 更快、大消息另有更优实现"的选型动机。PR 无 e2e 推理性能数字与精度测试。

## 验证与回归风险

验证即基准工具自身运行(4 卡/8 卡日志);无单测与精度测试,checklist 全部未勾选。风险面:`should_custom_ar` 边界放宽使 max_size 恰好等于消息尺寸的请求进入 custom AR 路径,属于行为变化但方向是修复误拒;基准工具仅 benchmark 目录,不影响运行时。

## 关联

关联 issue #13363(step-4);正文预告后续 PR 将按流量选择 one-shot/two-shot AR。同区域(all-reduce 优化)其他 PR 素材未提供,无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
