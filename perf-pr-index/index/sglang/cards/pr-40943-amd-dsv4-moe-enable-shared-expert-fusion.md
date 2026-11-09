# PR #40943: [AMD][DSV4] moe: enable shared-expert fusion on the grouped-topk path (megamoe)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @karverma-amd |
| 合并 | 2026-09-28 |
| 分类 | MoE (major) |
| Label | amd, deepseek, run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 为 topk.py 两行参数修正 |
| 链接 | https://github.com/sgl-project/sglang/pull/40943 |

## 一句话结论

修正 grouped-topk 调用点传给 gate 的融合 shared expert 计数,让 DSV4 在 ROCm megamoe 后端可启用 shared-expert 融合,decode TPOT 降约 8%。

## 动机

`select_experts` 已经计算 `num_fused_shared_experts_for_gate`(在 per-rank fused shared-slot / aiter 路径上为 0,因为 shared expert 由 `_post_process_topk_ids` 事后追加),但只在非 grouped 的 gate 调用点使用;两个 grouped 调用点(`grouped_topk` / `biased_grouped_topk`)仍传原始 `num_fused_shared_experts`,导致 gate 双重发射 shared 标记:routed expert 被挤掉一个槽位(top-6 变 top-5),且该标记落在 id `num_experts` 上,被 DeepEP per-rank remap 移到专家空间之外(EP8 下 384→392,合法 id 为 0..391)。DeepSeek-V4 走 grouped `biased_grouped_topk`,正好命中;FlyDSL MegaMoEv2 后端(#35619)上越界 id 使 dispatch 直接 fault(`HSA_STATUS_ERROR_EXCEPTION 0x1016`)——这正是 #35619 在 ROCm 上必须 `--disable-shared-experts-fusion`、shared expert 只能跑独立 per-layer GEMM 的原因。

## 优化点

两行修复:`select_experts` 调用 `grouped_topk` 与 `biased_grouped_topk` 时,把 `num_fused_shared_experts` 参数改为 `num_fused_shared_experts_for_gate`,使 gate 不再为事后追加的 shared expert 预留路由槽位。该改动在 per-rank fused shared-slot / aiter 路径之外是 no-op(两值相等),CUDA / 非 fused-shared-slot 路径行为不变。修复后 `--enforce-shared-experts-fusion` 可在 FlyDSL MegaMoEv2 上正常启动与推理,dispatch/combine 以 topk=7、local_experts=49(48 routed + 1 fused shared per rank)运行,shared expert 被真正融合进 megamoe 的 dispatch/combine 链,替代每层一次的独立 shared-expert GEMM。

## 关键改动

```text
python/sglang/srt/layers/moe/topk.py: grouped_topk / biased_grouped_topk 两个调用点改传 num_fused_shared_experts_for_gate(共两行)
```

```python
-                num_fused_shared_experts=num_fused_shared_experts,
+                num_fused_shared_experts=num_fused_shared_experts_for_gate,
```

## 性能声明

测试条件:MI355X gfx950,DeepSeek-V4-Pro-0813,TP8 / DP8 attention / EP8,`--moe-a2a-backend megamoe`,FlyDSL MegaMoEv2(a8w4),base 为 20260922 镜像 + #35619 bind-mount;A/B 为 shared-off(`--disable-shared-experts-fusion`)vs shared-on(`--enforce-shared-experts-fusion`)。

- 吞吐/交互 A/B(random 数据集,ISL/OSL 8192/1024,num-prompts = 4×conc):conc 256 时 Mean TPOT 74.2→**67.9 ms**(**−8.4%**),per-rank decode throughput **+5.5%**;conc 512 时 TPOT 90.8→90.1 ms(持平);聚合 output tok/s(conc256 2814.5→2775.9、conc512 3974.6→3873.3)在 ISL 8192(prefill/TTFT 主导)下处于噪声内。
- 准确率:GSM8K 200 题 5-shot:shared-off 0.940 / shared-on **0.970**,0 invalid,同一 fp4-MoE run-to-run 波动带内、无回退。
- 机制佐证:shared-on 臂确认 dispatch/combine 以 topk=7、local_experts=49 运行,无 0x1016 fault。

## 验证与回归风险

无新增测试;验证为上述启动/服务冒烟、GSM8K 与 bench_serving A/B。CI 状态:PR Test (Base) 沙漏(排队/进行中),Extra 与 AMD ROCm 10 失败。风险面:改动极小但触及所有 grouped-topk 模型(DeepSeek 系)的 gate 语义;在 `num_fused_shared_experts_for_gate == num_fused_shared_experts` 的常规路径上为精确 no-op,仅 per-rank fused shared-slot 路径行为改变;依赖 #35619 的 MegaMoEv2 集成与一组环境变量(SGLANG_USE_AITER、SGLANG_AMD_USE_FLYDSL_MEGA_MOE 等)。

## 关联

前序/依赖:#35619(Aiter MegaMoEv2 集成,因本 bug 在 ROCm 上要求 disable shared-experts fusion,本 PR 移除该限制)。同区域:DeepSeek-V4 MoE / megamoe dispatch-combine、aiter fused MoE 系列(pack 未列其他编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
