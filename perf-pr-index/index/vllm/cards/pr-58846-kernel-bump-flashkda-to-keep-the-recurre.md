# PR #58846: [Kernel] Bump FlashKDA to keep the recurrent state in fp32

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @simon-veitner-redhat |
| 合并 | 2026-09-26 |
| 分类 | Attention / Kernel(major) |
| Label | ready, ci/build |
| Diff | diffstat 缺失(素材包未提供 +x/-x);diff 仅含 1 行 GIT_TAG 变更 |
| 链接 | https://github.com/vllm-project/vllm/pull/58846 |

## 一句话结论

FlashKDA pin 升级到 fp32 recurrent state 版本,GLM-5.3-Flash 长上下文 tool-call 损坏显著缓解(42k 探针 67.6%→87.5%),内核时间 geomean −4.6%。

## 动机

旧 pin(b59532f)的 FlashKDA K2 kernel 把 KDA recurrent state 存为 bf16,每 16-token tile 后 round 一次;长 prefill(GLM-5.3-Flash)下误差跨 tile 与 chunked-prefill 边界累积,表现为长上下文 tool-call 输出损坏。上游 vllm-project/FlashKDA#13 在 tile 间保持 fp32 state,并补上缺失的 TMA store waits、修复 sm_10x 上的寄存器压力回归。

## 优化点

vLLM 侧唯一改动是 cmake/external_projects/flashkda.cmake 的 GIT_TAG bump(b59532f1… → 17a037d9…,即 FlashKDA#13 在 dev 分支的 merge)。op schema、flash_kda.h、源文件列表与 cutlass submodule 在两个 commit 间完全一致,无接口变化。

## 关键改动

```text
cmake/external_projects/flashkda.cmake: FetchContent GIT_TAG 更新(唯一改动)
```

## 性能声明

RedHatAI/GLM-5.3-Flash-NVFP4,TP2,B200(sm_100f, cu130)与 H200:内核时间 B200 geomean 快 4.6%(vs 旧 pin);e2e serving 吞吐在旧 pin ±1.3% 内(噪声级,vllm bench serve,random 数据集,16 输出 token,2k–32k 输入,并发 1–64)。精度:对 fp64 参考的终态相对误差 4k–128k token 为 0.37–0.39%(原 0.85–1.24%),输出相对误差 0.47%(原 0.48%);42k tool-call 探针 B200 224/256(87.5%)正确 vs 原 173/256(67.6%),34k 为 253/256 vs 256/256;H200 42k(模拟 MoE 隔离 FlashKDA)31/32 vs 21/32。

## 验证与回归风险

FlashKDA tests/test_fwd_full 580/580 on B200(对 torch_ref 的逐位比较,覆盖 fixed/varlen/batched 至 1M token);长上下文 tool-call 损坏探针(34k/42k,salted cold prefill);vllm bench serve A/B。风险:外部依赖 pin 变更影响所有 FlashKDA 使用方(KDA 类模型,如 GLM-5.3-Flash);fp32 state 可能带来少量带宽/寄存器开销,但实测内核反而更快、e2e 在噪声内。

## 关联

vllm-project/FlashKDA#13(内容来源)、#12(此前 pin);vLLM 侧 pack 未列出关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
