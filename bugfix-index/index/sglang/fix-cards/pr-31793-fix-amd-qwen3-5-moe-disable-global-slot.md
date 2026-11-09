# PR #31793: [Fix][AMD] Qwen3.5 MoE: disable global-slot shared-expert fusion under per-rank EP backends (MoRI + dp-attention init crash)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#31793](https://github.com/sgl-project/sglang/pull/31793) |
| 作者 | yichiche |
| 合并日期 | 2026-07-26 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 5 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
MI355X 上 Qwen3.5-397B-FP8 用 `--moe-a2a-backend mori --enable-dp-attention` 启动即崩:`FusedMoE.__init__` 断言 `(513-8)%8=1!=0` 失败;在 per-rank EP 后端 + moe_ep_size>1 时禁用 Qwen 全局槽共享专家融合,回退独立 shared-expert MLP。

## 问题现象
正文披露:在 MI355X/ROCm 上以 `--moe-a2a-backend mori --enable-dp-attention` 部署 **Qwen/Qwen3.5-397B-A17B-FP8**(512 routed experts + 1 fused shared expert)时,模型初始化阶段崩溃,所有 scheduler rank 在 `FusedMoE.__init__` 中 abort:

```
File ".../srt/layers/moe/fused_moe_triton/layer.py", line 227, ...
    assert (num_experts - num_shared_slots) % self.moe_ep_size == 0
AssertionError
```

开启 `--enable-dp-attention` 后 `moe_ep_size` 被提升为 `tp_size`(8);MoRI 属 deepep-class,`has_per_rank_fused_shared_slots()` 为 True,`num_shared_slots = 1*8 = 8`,而 `(513 - 8) % 8 = 1 != 0`,断言必失败。影响面是该模型在 MoRI + dp-attention 组合下完全无法启动(issue #31336)。

## 根因
正文披露:MoRI dispatcher(`token_dispatcher/moriep.py`)用**连续 per-rank 块**掩码专家:`expert_mask[ep_rank * num_local_experts : + num_local_experts] = 1`,仅当 `num_experts == num_local_experts * ep_size`(即每 rank 一个共享槽)时才正确;DeepSeek-V2 正是按此构建 `num_experts`。而 Qwen 的 shared-expert fusion 是**全局槽**设计:`Qwen2MoeSparseMoeBlock` 构造 `TopK` 时不带 `num_fused_shared_experts`(topk.py 的 per-rank remap 不触发),改为把共享专家手动追加到单一全局 id(`_append_shared_to_topk_output`,base id `N == num_experts`)并配 per-token `shared_expert_gate`。该「全局槽 + gate」布局无法用 MoRI 的 per-rank 连续 tiling 表达,也不能复用 DeepSeek 无 gate 的 remap 机制。issue 中「把 MoRI 当 global(`num_shared_slots=1`)」的替代方案能过 `__init__` 断言,但会破坏 MoRI 连续 dispatch 掩码(`65*8=520 != 513`,rank 7 块越界)并静默错路由专家。

## 修复方案
改 `python/sglang/srt/models/qwen2_moe.py`:当 shared-expert fusion 开启 **且** `uses_per_rank_fused_shared_slots()`(per-rank EP 共享槽后端,如 DeepEP/MoRI)为真 **且** `moe_ep_size > 1` 时,置 `self.enable_shared_expert_fusion = False`,回退到独立的(正确的)shared-expert MLP 路径,并用 `logger.warning_once` 提示一次;`moe_ep_size == 1` 与非 per-rank 后端不受影响(fusion 保持开启)。这是 correct-by-construction 修法:fusion 关闭后 `num_experts = 512` 恰好整除平铺(`64*8`),共享专家由带 gate 的 dense `shared_expert` MLP 计算后叠加;带 per-token gate 的 fully-fused per-rank 共享专家需要专门的 remap+append kernel,留作后续工作。关键 diff 摘录:

```diff
+        if (
+            self.enable_shared_expert_fusion
+            and uses_per_rank_fused_shared_slots()
+            and get_parallel().moe_ep_size > 1
+        ):
+            logger.warning_once(
+                "Disabling Qwen shared-expert fusion: it uses a single global "
+                ...
+            )
+            self.enable_shared_expert_fusion = False
```

## 影响与回归风险
- 回归面:仅 Qwen2/Qwen3.5 MoE 系模型的 shared-expert fusion 启用判定。断言级验证(mocked MoRI, ep_size=8):修复前 `num_experts=513, num_shared_slots=8` 断言失败;修复后 fusion 禁用 → `num_experts=512, num_shared_slots=0`,断言通过且 MoRI 连续掩码恰好平铺 `[0, 512)`;`ep_size==1` 与非 per-rank 后端行为不变。GSM8K sweep(MI355X):baseline 0.925、fp8-KV 0.915、dp-attention+dp-lm-head 0.935 均健康;注意所有加 `--moe-a2a-backend mori` 的 run(R3/R4/R5)精度塌到 ~0.01、invalid 率高——正文说明这是独立的 MoRI all-to-all 正确性问题,与本 init-crash 修复无关,另案跟踪(见 #21886 的 ROCm-MoRI enablement),本 PR 是能触达并调试该问题的前置。

## 关联
- https://github.com/sgl-project/sglang/issues/31336 (修复的 init crash issue)
- https://github.com/sgl-project/sglang/issues/21886 (MoRI 正确性问题所属的 ROCm-MoRI enablement)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
