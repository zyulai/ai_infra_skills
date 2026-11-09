# PR #36725: [Bug][MoE] Fix TRTLLM NVFP4 Routing Kernel Precision

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36725](https://github.com/vllm-project/vllm/pull/36725) |
| 作者 | robertgshaw2-redhat |
| 合并日期 | 2026-03-23 |
| labels | bug, nvidia, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
TRTLLM NVFP4 MoE 的 routing bias 被无谓降精度到 bf16,而内核支持 fp32 且模型发布即 fp32,改为原精度直传以修正路由精度。

## 问题现象
TRTLLM NVFP4 MoE 路由(routing)精度问题:路由相关的 bias 此前在 bf16 下运行,而内核本身支持 fp32,与模型发布时的精度不一致(正文仅以三点说明,未给出修复前的具体错误数值)。正文以 lm_eval 验证修复后精度:`nvidia/DeepSeek-R1-NVFP4`(num_concurrent=1000)上 gsm8k exact_match flexible-extract 0.9530(±0.0058)、strict-match 0.9507(±0.0060)。

## 根因
正文披露:routing 是以 bf16 运行的,而该内核支持 fp32,应当与模型发布时使用的精度保持一致——即 `e_score_correction_bias`(路由修正 bias)在传入 flashinfer 内核前被额外降精度到 bf16,引入了不必要的精度损失。从 diff 描述改动所暗示的机制:降精度转换发生在 `apply` 内构造 `routing_bias` 时,`router_logits` 对 DeepSeekV3 路由方式本就转成 fp32,bias 却仍被压到 bf16,两者精度不一致(由 diff 推断)。

## 修复方案
`vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py` 的 `apply`:删除 bias 的 bf16 转换,`e_score_correction_bias` 以原始 dtype 直接传给内核:

```python
-        # Prepare routing bias into kernel format.
-        routing_bias = e_score_correction_bias
-        if routing_bias is not None:
-            routing_bias = routing_bias.to(torch.bfloat16)
+        # Prepare router logits for kernel format.
         router_logits = (...)
         return flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
             routing_logits=router_logits,
-            routing_bias=routing_bias,
+            routing_bias=e_score_correction_bias,
```

`router_logits` 的条件性 fp32 转换保持不变。

## 影响与回归风险
- 回归面:仅 TRTLLM NVFP4 fused MoE 专家路径中传入 `trtllm_fp4_block_scale_moe` 的 `routing_bias` dtype——由 bf16 变为 checkpoint 原精度(如 fp32);依赖内核对 bias dtype 的接受能力(正文断言内核支持 fp32),若 flashinfer 版本较旧不接受非 bf16 bias 可能报错;`routing_method_type == Llama4` 等分支不受影响。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
