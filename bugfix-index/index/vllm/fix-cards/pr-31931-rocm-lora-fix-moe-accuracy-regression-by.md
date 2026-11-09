# PR #31931: [ROCm][LoRA] Fix MoE accuracy regression by preserving float32 router weight scaling

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#31931](https://github.com/vllm-project/vllm/pull/31931) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-01-08 |
| labels | rocm, ready |
| 热度 | 👍 0 · 💬 9 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论

#31676 把路由权重乘法移到精度转换之后,ROCm 上 MoE 出现非确定性路由;恢复在 float32 阶段乘路由权重修复。

## 问题现象

ROCm 上 MoE 精度回归(由 #31676 引入),表现为非确定性的专家路由,在 LoRA mixed-adapter 测试中尤其明显。修复后 `test_olmoe_lora_mixed` 在 ROCm 通过。

## 根因

正文披露:#31676 为把 bias 移到反量化之后而重排了累加后的操作顺序,但同时也把 `MUL_ROUTED_WEIGHT` 移到了 `.to(compute_type)` 精度转换之后,导致路由权重乘法发生在 bf16/fp16 而不是 float32。这种精度损失在 ROCm 上因 ROCm 与 CUDA Triton 后端混合精度处理方式的差异而产生不同输出。正文给出了前后代码对照:修复前先在 float32 累加器上乘 `moe_weight` 再转 bf16,修复前(错误顺序)先转 bf16 再与 float32 的 `moe_weight` 相乘。

## 修复方案

修改 `vllm/model_executor/layers/fused_moe/fused_moe.py` 的 Triton `fused_moe_kernel`:`MUL_ROUTED_WEIGHT` 分支移回精度转换之前,在 float32 累加器上加载并乘 `moe_weight`;保留 #31676 的 bias 在反量化之后相加的行为不变。

```diff
+    # Router weight multiplication MUST happen in float32 before precision
+    # conversion for numerical stability (especially critical on ROCm).
+    if MUL_ROUTED_WEIGHT:
+        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
+        accumulator = accumulator * moe_weight[:, None]
+
     if use_int8_w8a16:
         accumulator = (accumulator * b_scale).to(compute_type)
```

## 影响与回归风险

- 影响窗口:#31676 合并后至本修复前。
- 回归面:`fused_moe.py` 的 Triton `fused_moe_kernel`(CUDA 与 ROCm 共用)——所有走该 kernel 的 MoE 路径的乘路由权重顺序;int8/fp8 量化分支与 bias 相加位置保持 #31676 后的行为不变。

## 关联

#31676

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
