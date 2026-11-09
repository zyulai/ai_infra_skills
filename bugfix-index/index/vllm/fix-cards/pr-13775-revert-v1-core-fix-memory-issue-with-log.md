# PR #13775: Revert "[V1][Core] Fix memory issue with logits & sampling"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#13775](https://github.com/vllm-project/vllm/pull/13775) |
| 作者 | ywang96 |
| 合并日期 | 2025-02-24 |
| labels | v1 |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Sampler / Detokenization |
| bug 类型 | regression |

## 一句话结论
回滚 #13721:其 sampler 预热/cumem 方案与 LoRA 存在未解决问题,恢复原 profile_run 采样逻辑。

## 问题现象
本 PR 是回滚,正文未描述新的运行时现象。触发背景是 #13721("[V1][Core] Fix memory issue with logits & sampling")已合入但被认定未达可合入状态——正文原话:"This PR is not ready to be merged yet since we haven't resolved the issue with cumem and LoRA."。即该内存修复在 cumem 与 LoRA 相关场景存在问题,正文未给出具体报错或复现步骤。

## 根因
正文披露:#13721 尚未解决 cumem(所用内存分配机制)与 LoRA 之间的问题,因此被判定 not ready,整体回滚。至于 cumem 与 LoRA 相互影响的具体机制,正文未披露。从 diff 看,#13721 的做法是把 dummy 采样预热抽成 `_dummy_sampler_run` 并在 `capture_model` 之后调用,以预分配最大形状的 logits/sampling 缓冲、避免内存碎片(该 PR 的 commit 说明),回滚将这套预热路径整个移除(由 diff 推断)。

## 修复方案
回滚 PR #13721,涉及两个文件:
- `vllm/v1/worker/gpu_model_runner.py`:删除独立的 `_dummy_sampler_run` 方法,把 dummy 采样逻辑还原内联回 `profile_run`(`prompt_token_ids` 恢复为 `torch.ones_like(logits, dtype=torch.int64)`),并在 `del` 时连同 `logits`、`dummy_metadata` 一起释放;
- `vllm/v1/worker/gpu_worker.py`:删除 `compile_or_warm_up_model` 中 `capture_model` 之后对 `_dummy_sampler_run` 的预热调用(该调用原意为预分配采样相关内存缓冲)。

```diff
-        # Warm up sampler and preallocate memory buffer for logits and other
-        # sampling related tensors of max possible shape to avoid memory
-        # fragmentation issue.
-        self.model_runner._dummy_sampler_run(
-            hidden_states=self.model_runner._dummy_run(
-                num_tokens=self.scheduler_config.max_num_seqs))
```

## 影响与回归风险
- 影响窗口:#13721 合入(2025-02)到本回滚(2025-02-24)之间的 V1 版本。
- 回归面:改动位于 V1 的 profile_run 与 warm-up 路径。回滚后 #13721 想解决的 logits/sampling 内存碎片问题回归为未修复状态,LoRA + cumem 场景则回到已验证可用的行为;`_dummy_sampler_run` 这一独立方法随之消失,依赖它的外部代码(若有)会失效,但按 diff 该方法仅被本次删除的两处调用。

## 关联
- vllm-project/vllm#13721(被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
