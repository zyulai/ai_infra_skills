# PR #42011: [AMD][V4.1][*/N] Fix shared-expert fusion accuracy and speed up MoE routing on ROCm

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#42011](https://github.com/sgl-project/sglang/pull/42011) |
| 作者 | kkHuang-amd |
| 合并日期 | 2026-10-02 |
| labels | quant, amd, deepseek, run-ci, jit-kernel |
| 热度 | 👍 0 · 💬 1 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
ROCm 上 DSV4.1-Flash 共享专家融合 requant 把 e2m1 中点向零取整导致约 5% 精度损失;改为就近偶舍入,并把 decode 路由融合成单 kernel。

## 问题现象
DSV4.1-Flash 在 ROCm 上以 `--enforce-shared-experts-fusion` 运行时的两个问题:一是共享专家 requant 精度损失——旧的 cast 把 e2m1 中点(midpoint)向零取整,使共享专家整体缩小约 5%;二是 decode 路由性能不佳——decode 时路由回退到 reduce + topk + append 的多步路径。正文给出实测:GSM8K 1319 5-shot、TP2 开融合,精度 0.891 → 0.906(3 次),与不融合(0.904/0.905)持平;带 router 改动为 0.897/0.901/0.900。性能:MI355X TP2 + DSpark 每 decode step,router 省 0.41 ms、sorting 省 0.13 ms;AgentX c1 11,371.9 tok/s/GPU、P90 349.5(+2.6%/+4.8%),c2 11,819.8、P90 324.6(+2.8%/+10.7%)。

## 根因
正文在 Modifications 中披露了精度问题的根因:旧的 FP8→MXFP4 requant cast 把中点值向零取整(rounded midpoints toward zero),而 FP8 源权重恰好大量落在 e2m1 的中点上,导致整个矩阵系统性偏低,共享专家缩小约 5%。decode 路由方面,正文说明旧路径 decode 会回退(fall back)到 reduce + topk + append 的分离实现。

## 修复方案
- `python/sglang/srt/layers/quantization/fp8_utils.py`:新增 `_quantize_block_fp8_weight_to_mxfp4_rne`,在 HIP(`_is_hip`)下替代旧路径:scale 按 aiter `dynamic_mxfp4_quant` 的方式取(amax 尾数在 1.75 处舍入再 `floor(log2) - 2`),量化码按「就近舍入、平局取偶数尾数」计算;其他平台保持原 requant,输出不变。
- `python/sglang/kernels/ops/moe/rocm_router_gate.py`:`_router_gate_kernel` 新增 `NUM_SHARED`/`shared_id_base`,在同一次 launch 中直接写出融合共享专家列(shared slot 的 id 为 `shared_id_base + s`、weight 1.0),输出宽度变为 `topk + num_shared`,decode 不再回退到 reduce + topk + append。
- `python/sglang/srt/layers/moe/topk.py`:`biased_topk_jit_kernel_impl`/`select_experts`/`_post_process_topk_ids` 传递 `num_shared_append`,并相应调整 `_aiter_append` 与 routed experts 捕获逻辑(路由已写出共享列时不再重复 append,避免共享 id 写两次挤掉真实 routed expert)。
- `python/sglang/srt/environ.py`:新增 `SGLANG_AITER_MOE_SORTING_DISPATCH_POLICY`(默认 2 = multi-phase,0 恢复 aiter auto policy),`moe_runner/aiter.py` 传入该策略,小 batch 用 aiter multi-phase MoE 排序(diff 注释:auto 在 ~24 token 以下选 oneshot,gfx950 上要 11-16 us,multi-phase 只要 6-7 us,输出一致)。
- 更新 requant 单测(新增 `test_ties_round_to_even`)并给 `test_rocm_router_gate.py` 增加用例。

```diff
+    if NUM_SHARED > 0:
+        # fused_append_shared_experts: shared slot s holds id shared_id_base + s, weight 1.0
+        is_shared = lane >= TOPK
+        weights = tl.where(is_shared, 1.0, weights)
+        ids = tl.where(is_shared, shared_id_base + lane - TOPK, ids)
+    out_mask = lane < TOPK + NUM_SHARED
```

## 影响与回归风险
- 回归面:仅 ROCm/HIP 路径——FP8 块量化权重转 MXFP4 的 requant 结果改变(更接近真实值,但与旧版权重码位不同)、`rocm_router_gate` 输出宽度与调用约定变化、aiter MoE sorting 默认策略从 auto 变为 multi-phase(可用环境变量恢复);`_post_process_topk_ids`/`select_experts` 的共享专家 append/捕获判定逻辑重写,EPLB remap 场景新增 `not _eplb_remap_enabled()` 分支条件。其他平台 requant 输出不变。

## 关联
正文未列出
