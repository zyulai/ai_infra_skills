# PR #14848: Revert "[Model] Mamba2 Prefill Performance Tweaks: Fixing Flurry of Unnecessary Memory Copies  (#14778)"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#14848](https://github.com/vllm-project/vllm/pull/14848) |
| 作者 | tlrmchlsmth |
| 合并日期 | 2025-03-15 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
Mamba2 prefill 性能优化(#14778)导致 Bamba-9B gsm8k 评测全错;整体回退恢复正确性。

## 问题现象
ibm-ai-platform/Bamba-9B 以 `lm_eval --model vllm --model_args pretrained=ibm-ai-platform/Bamba-9B,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8 --tasks gsm8k --limit 100 --batch_size auto` 评测:main 分支 flexible-extract / strict-match 的 exact_match 均为 0(± 0);本 PR 回退后为 0.22(± 0.0416)/ 0.32(± 0.0469)。

## 根因
正文未披露具体根因机制,仅以评测对比表明回归来自被回退的 #14778(main 全错、回退后恢复)。从 diff 看,回退恢复的是 `mamba_mixer2.py` `forward_cuda` 中逐 idx 的 `ssm_state` 清零与 `varlen_state` 拷贝循环,移除了 #14778 引入的 `torch.vmap` 批量零初始化/批量拷贝写法(由 diff 推断)。

## 修复方案
整体回退 commit fe66b34728e5d383e3d19aefc544eeee808c99fb。`vllm/model_executor/layers/mamba/mamba_mixer2.py` 的 `forward_cuda`:`has_initial_states` 分支恢复为对 `~has_initial_states` 的索引逐个 `ssm_state[idx].zero_()`;SSM 状态更新恢复为 `for i, idx in enumerate(...)` 逐样本 `copy_(varlen_state[i])`。

```diff
-            batched_copy = torch.vmap(
-                lambda idx, source_state: mamba_cache_params.ssm_state[
-                    idx].copy_(source_state))
-            batched_copy(
-                mamba_cache_params.state_indices_tensor.unsqueeze(dim=-1),
-                varlen_state)
+            # update ssm states
+            # - varlen state is a (batch, nheads, headdim, dstate) tensor
+            for i, idx in enumerate(mamba_cache_params.state_indices_tensor):
+                mamba_cache_params.ssm_state[idx].copy_(varlen_state[i])
```

## 影响与回归风险
- 影响窗口:正文未写(回归由 #14778 引入,本 PR 为其回退)。
- 回归面:Mamba2(mamba_mixer2.py)prefill 的 SSM 状态零初始化与更新路径恢复为逐样本 Python 循环,放弃 vmap 向量化带来的内存拷贝优化(性能回退换取正确性)。

## 关联
- #14778(被回退的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
