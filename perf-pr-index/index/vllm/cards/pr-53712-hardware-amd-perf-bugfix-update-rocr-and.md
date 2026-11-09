# PR #53712: [Hardware][AMD][Perf][Bugfix] Update ROCr and clr in base image

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mawong-amd |
| 合并 | 2026-08-26 |
| 分类 | Model-specific Fast Path \| tier: major |
| Label | bug, rocm, ci/build |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/53712 |

## 一句话结论

更新 ROCm base image 的 ROCr/CLR，gfx950 上 DeepSeek V4 并发 1 时 TPOT 改善约 20%。

## 动机

正文说明该 PR 更新 ROCm 7.2.3 vLLM base image 中的 ROCR 与 CLR，以回移植两类 bugfix：一是复杂计算图在 graph replay 下某些 workload 会 segfault，二是某些 stream 依赖下的 kernel dispatch latency。后者（dispatch latency 修复）在部分 workload 上带来可观的 decode 性能改善，尤其是 `--async-scheduling` 与 ModelRunner V2 场景。

## 优化点

改动是基础镜像中运行时组件的版本更新，不涉及 vLLM 侧算子或调度逻辑重写：把 ROCR 与 CLR 更新到含上述 backport 修复的版本，从而消除 graph replay 的 segfault 并降低 kernel dispatch latency。性能收益来自运行时修复而非代码路径改动，PR 正文未列出具体版本号或构建参数之外的实现说明。

## 关键改动

```text
docker/Dockerfile.rocm_base: 更新 ROCR / CLR，回移植 graph replay 与 kernel dispatch latency 的 bugfix
```

## 性能声明

正文给出两组 decode 收益（均为 "at concurrency 1, shrinking to ... at concurrency 64"）：DeepSeek V4，ISL=1024、OSL=512、TP=8、gfx950，最好情况约 20% TPOT improvement，并发 64 时收窄到 9%；Kimi K3，ISL=1024、OSL=512、TP=8、gfx950，最好情况约 13% TPOT improvement，并发 64 时收窄到 4%。测试条件：硬件 gfx950（AMD），TP=8，ISL=1024 / OSL=512；workload 涉及 --async-scheduling 与 ModelRunner V2；精度 PR 未注明。

## 验证与回归风险

Test Plan 写明在 gfx950 与 gfx942 上对多个模型与配置测量 GSM8K，结果与运行时改动前一致，并会触发完整 AMD CI；Test Result 为 "There should not be any regressions in AMD CI caused by this PR"。可能受影响的路径为 ROCm 平台下依赖 graph replay 与 kernel dispatch 的 workload，正文未给出其他硬件的评估。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
