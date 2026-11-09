# PR #55671: [FlashInfer] Migrate autotune warmup to flashinfer.autotune_v2 managed cache

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @YangXu1990uiuc |
| 合并 | 2026-10-17 |
| 分类 | Attention / Kernel（tier: major） |
| Label | documentation, ready, nvidia, verified, warmup |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/55671 |

## 一句话结论

单 rank 部署把 FlashInfer autotune warmup 迁到 autotune_v2 managed cache，GSM8K 86.66%（PR 不主张吞吐提升）。

## 动机

PR 之前 #57635 的 warmup 采用 vLLM 自管的 per-rank legacy 文件（`autotune_configs_dp<dp>_rank<rank>.json`）与 group 级缓存同步。FlashInfer 提供 managed autotune_v2 缓存，但混合缓存命中（部分 rank 命中、其余进入 timing collectives）在最终 barrier 前可能挂死，FlashInfer #6185（2026-10-16 合并）在库内协调 pre-profiling hit 决策。该 PR 只在单 rank 门控下启用 managed cache，把多 rank 启用留到 pin 带修复的 FlashInfer release 并验证之后。

## 优化点

在两套 API 同时存在且部署为 single-rank、single-node、PP=1、非 elastic EP 时，自动选择 managed v2；其余部署保留 #57635 的 legacy 缓存同步（per-rank 文件、group 级校验、PP stage 调优组）。`use_flashinfer_autotune_v2(runner)` 逐项检查 `enable_elastic_ep`、`pipeline_parallel_size > 1`、`has_flashinfer_autotune_v2()`、`get_node_count() != 1`、`world_size > 1`，任一不满足即回 legacy。共享通用 warmup 执行（HiSparse、ReplaySSM、Kimi），保留 MRv2 稀疏 warmup 路线、speculative-drafter bucket 覆盖与 `round_up=True`，随 main 移除单独 BF16-only pass。`resolve_flashinfer_autotune_v2_root()` 不再把 config hash / flashinfer 版本 / arch 折进路径，由 FlashInfer 自行哈希环境。PR 明确这是 cache-lifecycle migration，不主张 universal deadlock prevention、zero warm-start profiling 或 improved serving throughput。

## 关键改动

```text
vllm/model_executor/warmup/flashinfer_autotune_cache.py: 新增 use_flashinfer_autotune_v2() 单 rank 门控与 resolve_flashinfer_autotune_v2_root()（managed store 路径）。
vllm/utils/flashinfer.py: has_flashinfer_autotune_v2 API 探测（legacy control 用 lambda: False 覆盖该函数回退）。
vllm/model_executor/warmup/kernel_warmup.py、flashinfer_sparse_mla_warmup.py: 共享 v1/v2 通用 warmup 执行，保留 MRv2 稀疏 warmup 路线。
docs/usage/security.md: 更新 VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR 说明，列出 managed（~/.cache/flashinfer/autotune/）与 legacy（$VLLM_CACHE_ROOT/...）默认值。
tests/model_executor/test_flashinfer_autotune_warmup.py: 新增 lifecycle 参数化用例（v1/v2/elastic/多节点、HiSparse/ReplaySSM/Kimi 等）。
```

```text
def use_flashinfer_autotune_v2(runner: "GPUModelRunner") -> bool:
    from vllm.distributed.parallel_state import get_node_count, get_world_group
    from vllm.utils.flashinfer import has_flashinfer_autotune_v2

    if runner.vllm_config.parallel_config.enable_elastic_ep:
        return False
    if runner.vllm_config.parallel_config.pipeline_parallel_size > 1:
        return False
    if not has_flashinfer_autotune_v2():
        return False
```

## 性能声明

PR 正文明确声明 "This is a cache-lifecycle migration, with no claim of universal deadlock prevention, zero warm-start profiling or improved serving throughput" 与 "No serving-performance claim is made."，故无吞吐/加速性能数字；给出的定量为正确性与生命周期结果：Targeted warmup suite 78 passed；六文件 pre-commit 与 mypy 通过；B200 TP1 Qwen3-8B-FP8 managed 冷/热启 3/3；B200 TP2+EP Qwen3-30B-A3B-FP4（FlashInfer CUTLASS）legacy 路径 4/4；B200 PP2 Qwen3-8B-FP8 legacy 2/2；managed warm-cache reuse "40 winners plus manifest unchanged"；历史 9/9 GPU lifecycle cases（Buildkite #94030）。Full GSM8K（1319 questions、5-shot、256 output tokens）：managed warm 1143/1319（**86.66%**），same-runtime legacy 1157/1319（**87.72%**），差 **-1.06 percentage points**，近似 paired 95% 区间 **[-2.41, +0.28]**，正文注明仅一对 runs、不证明精度等价或稳定回归。测试条件：B200、Model Runner v2、Torch 2.13.0、FlashInfer Python/cubin 0.7.0.post1（离线 `LLM.generate` 测试，非 HTTP serving benchmark）；当前 head 的 hosted CI pending，GPU/model 结果未在现 head 与依赖 pin 上 rerun。

## 验证与回归风险

验证：78 个 CPU warmup 用例、六个 PR 文件的 pre-commit 与 mypy、B200 TP1/TP2+EP/PP2 GPU 用例、Buildkite #94030 的 9/9 GPU lifecycle cases。风险：multi-rank managed tuning 保持禁用（需 FlashInfer release + pin + 验证才能重开）；混合 cache 命中在 finalization 前可能挂死，依赖 FlashInfer #6185；SM120 sparse-MLA CPB/crossover 校准是独立缓存（`FLASHINFER_AUTOTUNE_DIR`），managed winner reload 不分发其文件或进程内状态；部分非默认 per-op measurement policy 在 warm start 仍会 re-profile；当前 head 的 hosted CI pending，GPU 结果为历史证据。

## 关联

#57635（per-rank legacy 文件、group 校验与 PP 调优组，本 PR 保留其迁移）、FlashInfer #6185（pre-profiling hit 决策修复）、draft #61403（measurement-policy 实验）、FlashInfer #3573（EP/DP-aware profiling 与最终同步）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
