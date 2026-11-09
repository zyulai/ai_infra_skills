# PR #34673: [Bugfix][MoE Kernel] Fix incorrect routing selection for models without expert groups (e.g., MiniMax-M2.1)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34673](https://github.com/vllm-project/vllm/pull/34673) |
| 作者 | wwl2755 |
| 合并日期 | 2026-02-18 |
| labels | bug, nvidia, ready |
| 热度 | 👍 0 · 💬 4 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
无专家分组的 sigmoid 模型(如 MiniMax-M2.1)被误判为 DeepSeekV3 路由,flashinfer TRTLLM kernel 启动即崩;按 `num_expert_group` 判定改为 Unspecified。

## 问题现象
在 `MiniMaxAI/MiniMax-M2.1` 上启动服务,worker 直接崩溃:

```
RuntimeError: Worker failed with error 'Check failed: (args->n_group != 0) is false: n_group should not be zero for DeepSeekV3 routing'
```

影响面:所有 sigmoid 打分且 top_k > 1、但没有专家分组(`num_expert_group=None`)的 MoE 模型,使用 flashinfer TRTLLM kernel 时触发。修复后服务器正常拉起。

## 根因
PR 正文披露:此前 `get_routing_method_type()` 对任何 sigmoid 打分且 top_k > 1 的模型一律返回 `DeepSeekV3`,不管该模型是否真的使用分组专家路由(grouped expert routing)。MiniMax-M2.1 这类模型没有专家分组(`num_expert_group=None`),flashinfer TRTLLM kernel 的 DeepSeekV3 路由要求 n_group 非零,于是崩溃(check `args->n_group != 0` 失败)。

## 修复方案
`vllm/model_executor/layers/fused_moe/config.py` 中 `get_routing_method_type()` 签名扩展 `num_expert_group` 与 `has_e_score_bias` 两个参数,并按分组情况判定:

```diff
+    if has_e_score_bias:
+        if (num_expert_group or 0) > 0 and scoring_func == "sigmoid":
+            return RoutingMethodType.DeepSeekV3
+        else:
+            return RoutingMethodType.Unspecified
     if scoring_func == "sigmoid":
         if top_k == 1:
             return RoutingMethodType.Llama4
         else:
-            return RoutingMethodType.DeepSeekV3
+            return RoutingMethodType.Unspecified
```

调用方同步更新:`fused_topk_bias_router.py` 传 `num_expert_group=None, has_e_score_bias=True`;`fused_topk_router.py` 传 `num_expert_group=None, has_e_score_bias=False`;`grouped_topk_router.py` 移除构造时的硬编码(`sigmoid → DeepSeekV3`),改为在 property 中调用统一函数,传入真实的 `self.num_expert_group` 与 bias 是否存在。另外删除了 `tests/kernels/moe/test_flashinfer.py` 中为 #34477 添加的回归测试 `test_flashinfer_blockscale_fp8_none_expert_group`(该测试显式传 `routing_method_type=RoutingMethodType.DeepSeekV3`,与新判定逻辑不再一致)。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围;涉及引入该映射逻辑的 flashinfer TRTLLM MoE 路径(关联 #34494)。
- 回归面:所有通过 `get_routing_method_type()` 获取路由类型的 router(fused_topk、fused_topk_bias、grouped_topk)及其下游 flashinfer TRTLLM kernel 选择;sigmoid + top_k>1 + 有分组 + 有 bias 的模型(DeepSeekV3 类)仍返回 DeepSeekV3,行为不变;无分组的 sigmoid 模型从 DeepSeekV3 变为 Unspecified(后者意味着 FLASHINFER_TRTLLM kernel 不再被选中,按 grouped_topk_router 中原注释这是有意的排除行为)。softmax 路径与 Llama4(top_k==1 sigmoid)路径不变。

## 关联
- https://github.com/vllm-project/vllm/issues/34622
- https://github.com/vllm-project/vllm/issues/34477
- https://github.com/vllm-project/vllm/pull/34494

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
