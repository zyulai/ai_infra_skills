# PR #16801: [BugFix] Accuracy fix for llama4 int4 - improperly casted scales

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#16801](https://github.com/vllm-project/vllm/pull/16801) |
| 作者 | LucasWilkinson |
| 合并日期 | 2025-04-18 |
| labels | bug, ready |
| 热度 | 👍 3 · 💬 3 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
Llama4 INT4 MoE 三处问题:scales fp16/bf16 被不安全转 float、`FusedMoE` 缺 `params_dtype` 属性报错、router 权重被转半精度不兼容 kernel;一并修复恢复精度。

## 问题现象
正文未描述具体的复现步骤与错误输出样例,仅概括为 Llama4 INT4(int4 量化 MoE)的精度问题与相关报错。涉及的现象包括:scale 为 fp16/bf16 时被不安全地按 float 指针读取;运行时报 `AttributeError: 'FusedMoE' object has no attribute 'params_dtype'`;以及 Llama4 router 权重类型与 `moe_wna16_gemm` kernel 不兼容。

## 根因
PR 正文列出三点(标题点明核心是 scales 被不当 cast):
1. scales 实际是 fp16/bf16 类型,却被不安全地 cast 成 float 读取(`data_ptr()` 直接转 `const float*`,无类型校验)。
2. `FusedMoE` 对象缺少 `params_dtype` 属性(构造时未保存),导致 `AttributeError`。
3. Llama4 的 `custom_routing_function` 把 router scores 从 float 转回 `hidden_states.dtype`(如 bf16),而 kernel( namely `moe_wna16_gemm`)假定 router 权重是 float。

关于第 1 点的数值危害机制(读 fp16 位再按 float 解释),正文未展开,细节由 diff 佐证。

## 修复方案
三处修改:

1. `csrc/moe/moe_wna16.cu`:`topk_weights_ptr` 改用带类型校验的 `.data_ptr<float>()` 取指针(未提供时为 `nullptr`):

```diff
-  const float* topk_weights_ptr;
+  const float* topk_weights_ptr = nullptr;
   if (topk_weights.has_value())
-    topk_weights_ptr = (const float*)topk_weights.value().data_ptr();
+    topk_weights_ptr = (const float*)topk_weights.value().data_ptr<float>();
```

同文件把输出清零从 kernel 内逐元素写 0 改为 host 侧 `output.zero_()`(原来 kernel 只在 `blockIdx.z == 0` 时写 0,与输出缓冲初始化方式耦合)。
2. `vllm/model_executor/layers/fused_moe/layer.py`:构造函数在解析 `params_dtype` 后保存 `self.params_dtype = params_dtype`,消除 `AttributeError`。
3. `vllm/model_executor/models/llama4.py`:`custom_routing_function` 不再把 router scores 转回 `hidden_states.dtype`,保持 float 返回以匹配 kernel 约定:

```diff
-        router_scores = torch.sigmoid(router_scores.float()).to(
-            hidden_states.dtype)
+        # psuedo-standard is that the router scores are floats
+        router_scores = torch.sigmoid(router_scores.float())
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围;受影响的是 Llama4 INT4 量化及使用 `moe_wna16_gemm` 且 router/topk 权重类型非 float 的场景。
- 回归面:`moe_wna16.cu` 是通用 MoE wna16 kernel,类型校验更严格(非 float 的 `topk_weights` 会报错而非静默误读);输出初始化逻辑改变,依赖 kernel 内清零行为的调用路径需走 host 侧 `output.zero_()`;router scores 保持 float 后,期望半精度 router 权重的下游代码需适配。正文未附精度数值对比或测试结果。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
