# PR #24248: [PERF] Allreduce fusion. Support torch native matching. Tuning of the thresholds

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ilmarkov |
| 合并 | 2025-11-10 |
| 分类 | Parallelism / Communication(tier: major) |
| Label | performance, ready, torch.compile, ci/build |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);diff 另有 6 个文件被截断(可见 1 个新增 benchmark 文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/24248 |

## 一句话结论

FlashInfer allreduce+RMSNorm(+FP8/FP4 quant)融合的阈值调优与 torch 原生匹配:新增 kernel 基准确定 oneshot/twoshot 切换阈值,Hopper/Blackwell 更新阈值,并把 allreduce 移出 moe_forward custom op 让 MoE 模型也能匹配融合。

## 动机

FlashInfer 的 `trtllm_allreduce_fusion`(allreduce + rmsnorm + 可选量化)此前已有接入但阈值未经硬件调优,且没有系统基准来决定 oneshot/twoshot 算法在何种输入尺寸下切换。同时 MoE 模型的 allreduce 被包在 `moe_forward` custom op 内部,torch.compile 的融合 pass 看不到 allreduce+rmsnorm 模式,MoE 模型无法享受该融合。这是 fused allreduce 改进的第一部分(基于 #24604)。

## 优化点

三个方向。其一,新增 `benchmarks/kernels/benchmark_fused_collective.py`(torchrun 多进程、CUDA graph 计时):在多个 token 数下对比 FlashInfer 融合实现(oneshot/twoshot,pattern 覆盖 kARResidualRMSNorm、kARResidualRMSNormFP8Quant、kARResidualRMSNormFP4Quant,PDL、fp32 累加)与标准 `tensor_model_parallel_all_reduce` + RMSNorm(+ QuantFP8 / `scaled_fp4_quant`)的耗时,并按最快基线输出 speedup,作为阈值确定的工具。其二,依据该基准更新 Hopper 与 Blackwell 设备上的 flashinfer allreduce 阈值,在 two-shot 算法性能更优的输入尺寸区间启用 two-shot。其三,把 allreduce 移出 `moe_forward` custom op,使 fusion pass 能对 MoE 模型做 torch 原生模式匹配(non-custom-op fusion)。

## 关键改动

```text
benchmarks/kernels/benchmark_fused_collective.py: 新增 allreduce 融合基准(1129 行;oneshot/twoshot vs 标准 AR+RMSNorm/FP8/FP4 quant,CUDA graph 计时)
其余文件在 pack 中被截断,依据 PR 正文:各平台阈值更新;allreduce 移出 moe_forward custom op;non-custom-op fusion 匹配支持
```

## 性能声明

PR 正文未附具体 benchmark 数字;声明内容为:新增基准用于确定 flashinfer allreduce 的输入尺寸阈值;Hopper/Blackwell 上更新阈值并在 two-shot 更优时启用 two-shot 算法(结论基于该基准,数字未在正文中给出)。

## 验证与回归风险

正文声明:新增 non-custom-ops fusion 的测试与 Qwen3 MoE 的 e2e 测试(具体测试文件在截断部分,未能核对)。回归影响:阈值变化改变 Hopper/Blackwell TP 部署下 allreduce 融合路径的启用区间;MoE 模型的 allreduce 现暴露给 fusion pass(行为变化);依赖 FlashInfer comm 模块提供 `trtllm_allreduce_fusion`。

## 关联

#24604(本 PR 基于之上)、#24252(第二部分:Introduce compile ranges)。同分类 Parallelism / Communication 区域其他相关 PR 未在素材中列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
