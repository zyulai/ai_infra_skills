# PR #61465: [Bugfix] Require cudnn-frontend >=1.31.0: cuDNN could replay an unloaded kernel from a CUDA graph

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @PeganovAnton |
| 合并 | 2026-10-19 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | bug, ready, ci/build, nvidia |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);diff 本身可用(1 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/61465 |

## 一句话结论

把 `nvidia-cudnn-frontend` 下限从 1.19.1 提到 1.31.0,修复 decode CUDA graph 重放已卸载 cuDNN JIT kernel 的正确性 bug——最坏情况是无报错的静默错误输出。

## 动机

cudnn-frontend <1.31.0(缺 NVIDIA/cudnn-frontend#1195 "Keep execution plans alive for the CUDA graphs recorded from them")在最后一个 cuDNN owner 释放后会卸载 JIT 编译 kernel 的库,而 CUDA graph 中仍有启动该 kernel 的节点;重放时 dispatch 到占据该地址的任意代码。受影响栈 vllm/vllm-openai:v0.29.0-ubuntu2404(实际解析 cudnn-frontend 1.28.0,虽声明 floor 1.19.1)同时具备两个条件:FlashInferFP8ScaledMMLinearKernel 调 `bmm_fp8(backend="auto")` 允许 cuDNN runner;FlashInfer 以 `functools.lru_cache(maxsize=2048)` 按 shape 缓存 cuDNN graph 且逐出冷条目,而这些 GEMM 已捕获进 decode CUDA graph。可观察后果:illegal-address fault、同步时 CUDA error,或最严重的——完全无报错的静默错误输出。

## 优化点

一行依赖变更:`requirements/cuda.txt` 中 `nvidia-cudnn-frontend>=1.19.1` 改为 `>=1.31.0`(1.31.0 为含上游 graph retention 修复的首个版本,使 graph 在还能启动 kernel 时持有引用)。选择 frontend 而非 backend(cuDNN 9.27.0.42)修复的理由:vLLM 已声明 frontend floor、一行完成;pin `nvidia-cudnn-cu13` 会覆盖 torch 传递解析的 cuDNN。不经过 cudnn-frontend 的 cuDNN 调用者(torch convolutions、`SDPBackend.CUDNN_ATTENTION`)不受此 PR 保护,需等 torch 的 transitive cuDNN pin 到 9.27.0.42(当前不被认为受影响)。

## 关键改动

```text
requirements/cuda.txt: nvidia-cudnn-frontend>=1.19.1 -> >=1.31.0(注释说明 CUDA graph retention 修复)
```

## 性能声明

此为正确性/稳定性修复,无吞吐改进;代价测量(v0.29.0-ubuntu2404 镜像仅换 frontend wheel,每 arm 3 次取中位数):CUDA graph replay 45.04 µs 不变;capture 192.0→229.1 µs(每个被捕获 graph 在 warmup 时约 +37 µs,各次重复区间不重叠);plan build 7655→7710 µs(3 样本噪声内);相对 cuBLAS 的 max relative error 均为 0.000000。Eager per-call host time 无回归(1.31.0 中位数 95.80/94.88 µs vs 1.28.0 的 112.10/95.34 µs,区间重叠)。

## 验证与回归风险

硬件 RTX PRO 6000 Blackwell(sm_120)。frontend 修复前后各 96 cells(backends 9.19.0.56-9.26.0.51,变 graph 构造/workload/cache 尺寸/owner 生命周期):pre-fix 22/96 复现,PR #1195 后 0/96,且所有复现 cells 都在释放 owner 的 arm。backend wheel 对照:9.20.0.48 release arm 出现 SIGSEGV / CUDA error 718 / 3,948,544 of 5,275,648 元素错误(2,637,824 non-finite,`sync_status=0`,主机无任何报错),9.27.0.42 全干净;并核对 wheel 二进制(1.30.0 无 retention symbols,1.31.0 有 41 个)。未做端到端 serving 前后对比(作者明示为多小时随机测量);frontend fix 保留整个 ExecutionPlan 的额外内存未量化。回归风险:capture 时间小幅增加;#52225 根因直接确认(coredump + CUDA-GDB),#55571 likely fixed。

## 关联

#52225(根因调查的 issue)、#55571(likely fixed)、#57209(sm_12x 上 `bmm_fp8` pin cuBLAS,绕开 lane;与本 PR 互补而非替代,#61007 的 plan 构建开销由它处理)、#61007/#61069/#60119/#59768/#49926(明示不 claim);上游 NVIDIA/cudnn-frontend#1195;FlashInfer 0.7.2 的 capture-retention helper 亦缓解 GEMM 逐出路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
