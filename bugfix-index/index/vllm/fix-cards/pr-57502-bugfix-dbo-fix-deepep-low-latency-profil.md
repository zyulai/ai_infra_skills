# PR #57502: [Bugfix][DBO] Fix DeepEP low-latency profiling crash with DP+EP+DBO

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57502](https://github.com/vllm-project/vllm/pull/57502) |
| 作者 | micah-wil |
| 合并日期 | 2026-09-18 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
DP+EP+DBO 下 padding mask 在收缩到单 microbatch 后才算,尾部 ubatch 拿到空 mask,DeepEP combine 崩溃。

## 问题现象
`tests/v1/distributed/test_dbo.py::test_dbo_dp_ep_gsm8k[deepep_low_latency]` 在 ROCm(MI300X)上、DP+EP+DBO 配置下,engine 初始化阶段(内存 profiling)即开始崩溃,报错为:

```
AttributeError: 'NoneType' object has no attribute 'recv_hook'
ValueError: torch.cat(): expected a non-empty list of Tensors
```

影响面:走 DeepEP low-latency 的 DBO(data parallel + expert parallel + dual-batch overlap)组合,profiling 阶段直接失败。

## 根因
正文披露:自 vllm-project/vllm#5607 之后,V1 `GPUModelRunner._dummy_run` 中 MoE padding mask(`is_padding`)是在 `num_tokens_padded` 已为 DBO forward-context 记账收缩到单个 microbatch 之后才计算的。于是 mask 只覆盖一个 microbatch,DBO ubatch wrapper 为尾部 microbatch 切到越界(空)的 mask,导致 DeepEP low-latency combine 在两个 microbatch 之间失同步而崩溃。

## 修复方案
修改 `vllm/v1/worker/gpu_model_runner.py` 的 `_dummy_run`:
- 在单 ubatch 调整**之前**按完整 padded token 数计算 `is_padding`,profile run 时让 dummy tokens 真正走 experts(而不是全部当 padding),与 ModelRunner V2 行为一致,使 MoE 内存被正确测量、MoE kernel 被实际执行:

```diff
+            num_tokens_unpadded = num_tokens_padded if is_profile else 0
+            is_padding = self._prepare_padding_mask(
+                num_tokens_unpadded, num_tokens_padded
+            )
+
             if ubatch_slices_padded is not None:
                 # Adjust values to reflect a single ubatch.
...
-            is_padding = self._prepare_padding_mask(0, num_tokens_padded)
```

正文验证:MI300X(2 卡,DP=2,EP,DBO,DeepEP 用 ROCm/DeepEP 构建)上 `test_dbo_dp_ep_gsm8k[deepep_low_latency]` 与 `[deepep_high_throughput]` 均 PASSED。

## 影响与回归风险
- 影响窗口:#5607 合入到本修复之间的版本,DP+EP+DBO + DeepEP low-latency 组合在 profiling 时可能崩溃。
- 回归面:改动仅位于 `_dummy_run` 的 padding mask 计算与 profile 语义,影响所有走 `_dummy_run` 的场景:profile 时 dummy token 会流经 experts(内存 profiling 读数可能比以前更接近真实负载);非 profile 路径 `is_padding` 全 True 与原先一致。DBO ubatch 切分逻辑不变,只是每个 microbatch 都拿到正确尺寸的 mask。

## 关联
- vllm-project/vllm#5607(引入时序问题的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
