# PR #23882: Deepseek V4

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hnyls2002 |
| 合并 | 2026-05-08 |
| 分类 | MoE(tier: major) |
| Label | documentation, high priority, quant, dependencies, deepseek, speculative-decoding, sgl-kernel, blackwell, npu, benchmark, run-ci, diffusion, jit-kernel, bypass-fastfail |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 超长,素材仅保留 17 个源码类文件,另注明 "153 file(s) dropped" |
| 链接 | https://github.com/sgl-project/sglang/pull/23882 |

## 一句话结论

把 fork 上的 DeepSeek V4 完整支持(c128 压缩 attention、hybrid SWA、MXFP4 MoE、CustomAllReduceV2)rebase 进 main。

## 动机

DeepSeek V4 的支持此前长期停留在 fork 分支,与 main 已累计分叉 2880 个 commit。PR 正文即 rebase 进度日志:按 ~100 commit 一批拆成 29 个批次逐一合入,最终 29/29 完成,目标是让 V4 的模型实现、kernel 与调度改动整体进入上游,而不是继续维护一份难以演进的分叉。

## 优化点

这个 PR 的"性能内容"分散在一批机制上:(1) JIT kernel 新增 DeepSeek V4 专属的 c128 压缩 kernel,decode 端在 128-chunk 上用 online softmax 维护 `[max, sum, kv]` 运行态,prefill 端按 compress/write 两个 plan 分 pass 处理,支持 PDL;(2) hybrid SWA 池改用 memory-based sizing,把 `DSv4MemoryCalculator` 收敛进新的 `DSv4PoolConfigurator`,并复用 main 的 `MemoryPoolConfig` 框架;(3) nixl PD 迁移用泛化 `send_state` 统一 SWA/NSA/Mamba 状态传输;(4) 保留 fork 的 `CustomAllReduceV2` JIT dispatch(优先于 main #19880 版本)以及 `sqrtsoftplus` fused topk、`SGLANG_OPT_*` 一系列可选开关(`SGLANG_OPT_SWIGLU_CLAMP_FUSION`、`SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK` 等)。

## 关键改动

```text
python/sglang/jit_kernel/csrc/deepseek_v4/c128.cuh: c128 压缩 decode/prefill kernel(online softmax + PDL)
python/sglang/jit_kernel/csrc/deepseek_v4/c128_online.cuh: 在线 128-chunk 压缩,decode 维护 [max,sum,kv] 状态
python/sglang/srt/mem_cache/... model_runner_kv_cache_mixin.py + pool_configurator.py: DSv4PoolConfigurator 吸收内存计算
python/sglang/srt/disaggregation/nixl/conn.py: 泛化 send_state 取代 _send_mamba_state dispatcher
python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py: 保留 fork JIT CustomAllReduceV2
```

## 性能声明

正文未给出任何 benchmark 数字(素材正则未检出 perf claim),PR 未注明测试条件。机制层面的量化收益均未在本 PR 内声明。

## 验证与回归风险

- `test_dsv4_pd_disagg_nixl.py` 改为 fully-symmetric topology(P/D 均开 dp-attention + deepep + EAGLE MTP),因为 V4 上 NSA c4/c128 indexer ring 在 spec 状态下会翻倍(8→16 / 128→256),非对称 P/D 布局不匹配,PD-MTP 非对称尝试已被 revert。
- `CustomAllReduceV2` 与 main #19880 版本存在实现分叉(nvlink 预检查、piecewise cuda graph 处理等 main 有而 fork 无),留有 TODO 待 reconcile。
- rebase 涉及 29 个批次、上百处 keep-dsv4/accept-main 决策,涉及 FP8/MXFP4 量化、ROCm(fp8_dtype e4m3fnuz)、NPU、Blackwell 等多路径;diffstat 缺失,实际变更规模需查 PR 页面。

## 关联

同区域:PR #27954(dsv4 MLA decode q-heads pad 到 64)、PR #38440(DSpark compact mode)、PR #24880(HiSparse DSV4 P2D DRAM)。正文引用的 main 侧 PR:#19880、#20330、#20343、#22384、#18939、#19266 等。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
