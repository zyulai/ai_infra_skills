# PR #56377: [Bugfix] Disable sequence parallelism / async TP under batch invariance and add a TP regression test

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56377](https://github.com/vllm-project/vllm/pull/56377) |
| 作者 | LioEinaudi |
| 合并日期 | 2026-09-27 |
| labels | bug, documentation, ready |
| 热度 | 👍 1 · 💬 11 |
| 分类 | Parallelism / Communication |
| bug 类型 | regression |

## 一句话结论
VLLM_BATCH_INVARIANT=1 未关掉 SP/async TP,reduce-scatter 归约顺序依赖 batch 致 bs=1 与 bs=64 logprobs 不一致;配置层强制关闭并补回归测试与文档。

## 问题现象
4x RTX PRO 6000 + Qwen3-1.7B 实测:SP 开启时,bs=1 与 bs=64 之间有 1522/1536 个采样 token 的 logprobs 不同,greedy 输出改变(17/64 条);SP 关闭时 1536 个全部一致。正文另注明:当前 `enable_sp` 会使 VllmConfig 回落到 V1 model runner(MRV2 尚不支持 SP),故 SP-on 数据均来自 V1;强制的 V1 SP-off 对照是 bitwise 稳定的,效应确实来自 SP 路径。

## 根因
正文明确披露:`VLLM_BATCH_INVARIANT=1` 并未阻止 `pass_config.enable_sp` / `fuse_gemm_comms` 被启用。`SequenceParallelismPass` 在编译范围超过 `sp_min_token_num` 时把 `all_reduce + rms_norm` 重写为 `reduce_scatter + rms_norm + all_gather`;该 reduce-scatter 走普通 PyNccl 路径,归约顺序取决于哪个 rank 拥有某个 token 的 chunk——即取决于 batch 组成,因此不满足 batch invariance。把 SP reduce-scatter 本身做成 batch-invariant(像 MoE combine 的 `reduce_scatterv` 那样 owner 无关的归约顺序)留作后续,以便日后在 batch invariance 下重新启用 SP。

## 修复方案
- `vllm/config/vllm.py` `__post_init__`:在 breakable-CUDA-graph 判定之前新增守卫——`VLLM_BATCH_INVARIANT` 且 `enable_sp` 或 `fuse_gemm_comms` 为真时,`warning_once` 提示后强制 `pass_config.enable_sp = False; pass_config.fuse_gemm_comms = False`;

```diff
+        if envs.VLLM_BATCH_INVARIANT and (
+            pass_config.enable_sp or pass_config.fuse_gemm_comms
+        ):
+            logger.warning_once(
+                "Disabling sequence parallelism and async TP "
+                ...
+            pass_config.enable_sp = False
+            pass_config.fuse_gemm_comms = False
```

- 新增多 GPU 回归测试(TP=4,prompt 覆盖 `sp_min_token_num` 两侧)置于 `tests/v1/determinism/`,在 main 无守卫时失败、加守卫后通过(guard 生效时 bs=1 vs bs=64 logprobs bitwise equal);
- `docs/features/batch_invariance.md` 补充限制说明。

## 影响与回归风险
- 回归面:同时设置 `VLLM_BATCH_INVARIANT=1` 且显式开启 `enable_sp`/`fuse_gemm_comms` 的用户会被强制关闭并收到一次告警(预期行为变化);守卫在配置层、与 runner 无关;正文说明 SP/batch-invariance 的交互应在 SP 落入 MRV2 后重新测量。测试硬件为 4x NVIDIA RTX PRO 6000 Blackwell (sm_120),vLLM main @ 7470082f5 + 本补丁,torch 2.13.0+cu130。

## 关联
- https://github.com/vllm-project/vllm/issues/56370 (Fixes;含复现脚本与数据)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
