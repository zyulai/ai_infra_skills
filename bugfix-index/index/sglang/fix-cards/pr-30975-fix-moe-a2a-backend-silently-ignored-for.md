# PR #30975: Fix --moe-a2a-backend silently ignored for LongCat-2.0 (moe_topk missing from gate)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#30975](https://github.com/sgl-project/sglang/pull/30975) |
| 作者 | whn09 |
| 合并日期 | 2026-07-16 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
LongCat-2.0 的 top-k 属性名 moe_topk 不在 Scheduler 识别列表里,initialize_moe_config 从不执行,--moe-a2a-backend deepep 被静默忽略;把 moe_topk 加入列表修复。

## 问题现象
用 `--moe-a2a-backend deepep` 服务 LongCat-2.0 时,该 flag 被静默忽略:模型实际以 `StandardDispatcher` 运行,而非请求的 DeepEP 专家并行 all-to-all。作者在 H200 上运行时确认:尽管传了 `--moe-a2a-backend deepep` 且日志打印了 "DeepEP MoE is enabled",`get_moe_a2a_backend()==NONE`、`dispatcher==StandardDispatcher`。修复后在 4x p5en H200(tp16/ep16)验证 `get_moe_a2a_backend()==DEEPEP` 并构建 `DeepEPMoE`。

## 根因
正文披露:根因在 Scheduler。`initialize_moe_config()`(唯一从 server_args 设置全局 `MOE_A2A_BACKEND` 的地方)仅在模型 config 暴露固定元组 `moe_topk_attrs` 中的 top-k 属性名之一时才被调用;LongCat-2.0 的属性名是 `moe_topk`(无下划线,见 `configs/longcat_flash.py`),不在元组中——元组里是带下划线的 `moe_top_k`。于是 `initialize_moe_config()` 从不执行 → 全局 `MOE_A2A_BACKEND` 保持 `None` → `get_moe_a2a_backend()` 回落 `MoeA2ABackend.NONE` → `get_moe_impl_class()` 判定 `is_deepep()==False`,构建 `FusedMoE` + `StandardDispatcher` 而非 `DeepEPMoE`。

## 修复方案
`python/sglang/srt/managers/scheduler.py` 的 `moe_topk_attrs` 元组加入 `"moe_topk"`,使 LongCat-2.0 被识别为 MoE 模型从而执行 `initialize_moe_config()`,尊重 `--moe-a2a-backend`。

```diff
         moe_topk_attrs = (
             "num_experts_per_tok",
             "num_experts_per_token",
             "top_k_experts",
             "moe_top_k",
+            "moe_topk",
         )
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅扩展属性名识别;不改变已被识别模型的 dispatch/compute,无模型数学变化,所有既有属性名保留。正文另提醒:LongCat-2.0(top-k=12)启用 DeepEP 还需 UCCL/DeepEP 低延迟 internode launcher 的 `kNumMaxTopK >= 12`,默认值 9 会 abort——属 EP 后端的独立问题,不在本 PR 范围。

## 关联
正文未列出其他 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
