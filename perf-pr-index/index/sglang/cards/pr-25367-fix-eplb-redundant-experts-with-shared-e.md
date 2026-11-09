# PR #25367: Fix EPLB redundant experts with shared expert fusion and Waterfill

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @xutizhou |
| 合并 | 2026-05-21 |
| 分类 | Model-specific Fast Path(major) |
| Label | run-ci |
| Diff | pack 未提供 diffstat(additions/deletions/files 均缺失),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/25367 |

## 一句话结论

修复 EPLB 冗余 expert 与 shared expert fusion/Waterfill 的交互:Waterfill 使用物理 expert 数,DeepSeek V3 EP16 Waterfill 吞吐 +2.27%/+3.30%。

## 动机

开启静态 EPLB 冗余 expert(`ep_num_redundant_experts`)并配合 shared expert fusion 时存在两处错配:TopK id 在 Waterfill 之前已被重映射为物理 expert id,但 DeepEP 交错布局的重映射和 Waterfill balancer 仍以逻辑 routed expert 数(`num_routed_experts` / `router_logits.shape[1]`)计算 per-rank expert 数,导致 shared expert 槽位重映射错误;同时 fused MoE 权重加载时,checkpoint 的 shared expert 权重位于逻辑 expert 边界之后,而本地 fused MoE 权重把它们存放在物理 routed expert 之后,需要正确映射回物理 shared 槽位。

## 优化点

三处配套修复。其一,`model_runner.py` 构造 `DeepEPWaterfillBalancer` 时传入 `num_routed_experts + ep_num_redundant_experts`(物理 routed expert 数),使 Waterfill 的 shared-expert 槽位重映射把冗余 expert 计入 per-rank expert 数。其二,`topk.py` 的 `_remap_topk_for_deepep` 参数从 `n_routed_experts` 改为 `num_physical_routed_experts`,`num_local_routed = num_physical_routed_experts // ep_size`;调用点 `_post_process_topk_ids` 优先取 `expert_location_dispatch_info.num_physical_experts`(无 dispatch info 时回退 `router_logits.shape[1]`)。其三,`fused_moe_triton/layer.py` 的 `weight_loader` 用逻辑 expert 边界识别 fused shared expert 权重(`shared_expert_id = expert_id - num_logical_experts`),当 EPLB metadata 激活且 DeepEP 后端时,把 checkpoint shared 权重映射到各 rank 的物理 shared 槽位(`rank * num_local_experts + _num_local_routed + shared_expert_id`),否则保持原有 `num_global_routed + shared_expert_id` 布局。

## 关键改动

```text
python/sglang/srt/model_executor/model_runner.py: Waterfill balancer 传入
  含冗余的物理 routed expert 数
python/sglang/srt/layers/moe/topk.py: _remap_topk_for_deepep 改用
  num_physical_routed_experts
python/sglang/srt/layers/moe/fused_moe_triton/layer.py: weight_loader 按
  逻辑边界识别 shared 权重并映射到物理 shared 槽位
```

## 性能声明

PR 正文 Testing 段:DeepSeek V3 EP16、redundant=16 与 redundant=32 的内部 A/B 对比,Waterfill 吞吐分别 +2.27% 与 +3.30%,MMLU 无回退。测试硬件、workload 参数与评测配置 PR 未注明;此外仅给出 py_compile、git diff --check、pre-commit 等静态检查。

## 验证与回归风险

PR 正文未附 accuracy 数值(MMLU 仅文字声明"did not regress"),未列新增单测,pack diff 中也无测试文件。风险集中在 expert id 语义:三处改动把"逻辑数"换为"物理数"的假设依赖调用顺序(logical->physical 重映射发生在 DeepEP 重映射之前),若某条路径(如未启用 EPLB metadata、或非 DeepEP 后端)顺序不同,`expert_location_dispatch_info.num_physical_experts` 回退到 `router_logits.shape[1]` 可能再次错配;`weight_loader` 的 shared 权重映射分支对 `require_global_experts + deepep` 与其他后端给出两种物理布局,加载错误会直接导致 expert 权重错位。影响面:EPLB + shared expert fusion + DeepEP(Waterfill)组合路径,不影响无冗余 expert 的部署。

## 关联

同区域:#24515(LPLB,linear programming load balancer)、#22734(MSCCL integration)。EPLB/Waterfill 的前序 PR 编号在素材中未给出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
