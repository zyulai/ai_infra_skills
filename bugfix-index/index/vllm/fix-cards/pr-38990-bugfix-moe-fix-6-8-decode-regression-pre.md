# PR #38990: [Bugfix][MoE] Fix 6-8% decode regression: prefer multi-stream shared expert overlap

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38990](https://github.com/vllm-project/vllm/pull/38990) |
| 作者 | voipmonitor |
| 合并日期 | 2026-04-05 |
| labels | bug, ready, ready-run-all-tests |
| 热度 | 👍 2 · 💬 7 |
| 分类 | MoE |
| bug 类型 | regression |

## 一句话结论
#35153 让 TP-only MoE 无条件走 EXTERNAL 顺序执行 shared experts,丢掉 aux stream 重叠,decode 吞吐回退 6-8%;调整判断顺序优先 MULTI_STREAM_OVERLAPPED 修复。

## 问题现象
TP-only(无 EP/EPLB)多卡部署的 MoE 模型 decode 吞吐较此前回退 6-8%,由 #35153 引入(正文注明:该回归不影响任何官方 vLLM release)。影响任意 MoE 模型(DeepSeek-V3、GLM-5 等)在多 GPU TP 无 EP 下的部署。基准(GLM-5 MoE,8x RTX PRO 6000 Blackwell,TP=8,C=1,无 MTP):ctx 0 为 36.8→41.8 tok/s(+14%),ctx 16k 为 34.8→37.8(+9%),ctx 32k 为 33.8→36.8(+9%);修复后 0.19.1 优于 0.18.1,并与 SGLang(cutlass MoE、无 custom AR)持平。

## 根因
正文披露:`SharedExperts._determine_shared_experts_order()` 中 `_has_external_experts` 检查排在 aux-stream 重叠检查之前。TP-only 配置下 `_has_external_experts` 返回 True(EPLB 与 flashinfer 双侧 kernel 都未启用),于是无条件选择 `EXTERNAL`——shared experts 在主 stream 上先于 routed experts 顺序执行;`MULTI_STREAM_OVERLAPPED`(shared experts 在 aux stream 上并行,即 0.18.x 的行为)永远不会被选中,即使 aux CUDA stream 可用。

## 修复方案
改 `vllm/model_executor/layers/fused_moe/runner/shared_experts.py`:把属性重命名为 `_use_external_experts`,语义取反(现在表示"应禁用重叠、退回 EXTERNAL"),并在其开头加入 `if self._use_dp_chunking: return False`;`_determine_shared_experts_order()` 改为先判断 `_use_external_experts`,不满足时继续走 MULTI_STREAM_OVERLAPPED 等路径:
```python
-        if self._has_external_experts and not self._use_dp_chunking:
+        if self._use_external_experts:
             return SharedExpertsOrder.EXTERNAL
```

## 影响与回归风险
- 回归面:`SharedExpertsOrder` 的选择逻辑,覆盖所有 MoE runner 的 shared experts 执行路径。TP-only 场景由 EXTERNAL 恢复为 MULTI_STREAM_OVERLAPPED;EPLB(非 allgather_reducescatter 后端)与 flashinfer 双侧 kernel 场景仍走 EXTERNAL(正确性考量保持);DP chunking 场景不再选 EXTERNAL。正文测试计划注明 FLASHINFER_MLA + FLASHINFER_CUTLASS MoE 后端验证不变,但"运行既有 MoE 测试确认无正确性回归"一项在正文中尚未勾选。

## 关联
- #35153(引入回归)
- [#37113](https://github.com/vllm-project/vllm/issues/37113)(Related)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
