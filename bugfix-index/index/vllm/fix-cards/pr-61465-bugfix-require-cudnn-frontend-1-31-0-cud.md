# PR #61465: [Bugfix] Require cudnn-frontend >=1.31.0: cuDNN could replay an unloaded kernel from a CUDA graph

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#61465](https://github.com/vllm-project/vllm/pull/61465) |
| 作者 | PeganovAnton |
| 合并日期 | 2026-10-19 |
| labels | bug, ready, ci/build, nvidia |
| 热度 | 👍 0 · 💬 29 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | general |

## 一句话结论
旧 cudnn-frontend 会在 CUDA graph 仍引用时卸载 JIT kernel,replay 落到被占用地址致错误输出;requirements floor 提到 1.31.0。

## 问题现象
受影响栈 `vllm/vllm-openai:v0.29.0-ubuntu2404` 同时满足两个条件:FlashInfer FP8 GEMM 走 cuDNN(`bmm_fp8(backend="auto")`)且被捕获进 decode CUDA graph;FlashInfer 用 `functools.lru_cache(maxsize=2048)` 按 shape 缓存 cuDNN graph 并驱逐冷条目。可观察结果:illegal-address fault、同步时报 CUDA error,或——对服务最关键——完全无报错的静默错误输出。#52225 表现为 Xid 13 `Out Of Range Register`;#55571 判为 likely fixed(`VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel` 或 `--enforce-eager` 都能消除)。96 格确定性复现:预修复 frontend 22/96 复现,修复后 0/96;某静默错误格 5,275,648 个元素中 3,948,544 个错误、`sync_status=0`。

## 根因
正文披露:cudnn-frontend #1195 之前,cuDNN 可在最后一个 cuDNN owner 消失时卸载 JIT 编译 kernel 的库,而 CUDA graph 仍含启动该 kernel 的 node;replay 时 dispatch 到占据该代码地址的内容。上游修复让 graph 在能 launch kernel 期间持有引用。GPU coredump 与 CUDA-GDB 独立确认故障 kernel,并直接观察到:LRU 驱逐在新图编译后 436µs 销毁了冷捕获图的 plan 与 runtime kernel,其 graph node 仍存活、自捕获 66 分钟无 host launch。

## 修复方案
`requirements/cuda.txt`:`nvidia-cudnn-frontend>=1.19.1` → `>=1.31.0`(首个包含 NVIDIA/cudnn-frontend#1195「Keep execution plans alive for the CUDA graphs recorded from them」的版本):

```diff
-nvidia-cudnn-frontend>=1.19.1
+# >=1.31.0 keeps execution plans alive for the CUDA graphs recorded from them
+nvidia-cudnn-frontend>=1.31.0
```

选 frontend 而非 backend(cuDNN 9.27.0.42):vLLM 已声明其 floor、一行改动,且不会覆盖 torch 传递解析的 cuDNN 版本;FlashInfer 0.7.2 的 capture-retention helper 在当前 main 另行缓解 GEMM 驱逐路径。

## 影响与回归风险
- 影响窗口:v0.29.0 image 实际解析 cudnn-frontend 1.28.0(声明 floor 1.19.1),对该 image 的有效 bump 为 1.28.0 → 1.31.0;backend ≥9.22 时预修复 frontend 在默认 cache 大小下不复现(cache 持有第二引用),仅凭后端版本无法判断部署是否暴露。
- 回归面:仅依赖 floor 提升;覆盖所有经 cudnn-frontend 的 cuDNN 消费者(flashinfer_cudnn NVFP4 linear、cuDNN ViT 与多模态 encoder attention)及 #57209 未覆盖的 sm_100/sm_121;torch convolutions 与 `SDPBackend.CUDNN_ATTENTION` 不经 frontend,不受影响。实测 capture 每图多约 37µs(warmup 一次性),replay 与数值不变;frontend 修复会多保留整个 ExecutionPlan,内存代价未量化。

## 关联
- NVIDIA/cudnn-frontend#1195(上游修复)
- #52225(根因直接确认)、#55571(likely fixed)
- #57209(sm_12x 上绕开 cuDNN 车道的另一改动)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
