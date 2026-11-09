# PR #6858: fix ep_moe_reorder kernel bugs

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#6858](https://github.com/sgl-project/sglang/pull/6858) |
| 作者 | BBuf |
| 合并日期 | 2025-06-04 |
| labels | (none) |
| 热度 | 👍 6 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
ep_moe_pre_reorder CUDA kernel 只支持 float32,改为按输入 dtype 模板分发并补非整除尾部循环。

## 问题现象
正文未给报错日志,Motivation 列出:原 CUDA kernel 实现只支持 float32,而推理输入通常是 bfloat16 或 float16;kernel 缺少按 dtype 的向量化长度处理,也未考虑不能整除的情形。附 H100 上 pre-reorder 性能对比(bs 64→4096,CUDA kernel 9.952→108.288,Triton kernel 15.584→147.552,CUDA 全面更快)。

## 根因
正文 Motivation 披露:kernel 将输入/输出指针硬编码为 float 并以 `16 / sizeof(float)` 计算向量化宽度,无法处理 bf16/fp16 输入;向量化主循环不处理非整除的尾部元素;当 `a1_scales_ptr != nullptr` 且 `use_per_token_if_dynamic` 为真时,scale 在 topk 循环内重复计算、产生冗余全局内存读取;另外 benchmark 脚本把输入张量分配错误地计入了计时,且缺少单元测试。

## 修复方案
- `sgl-kernel/csrc/moe/ep_moe_reorder_kernel.cu`:kernel 模板化,host 侧按输入 dtype 分发;`vec_size = 16 / sizeof(scalar_t)`;新增尾部标量循环处理非整除元素;per-token scale 提到 topk 循环外只算一次。
- `sgl-kernel/benchmark/bench_moe_ep_pre_reorder.py`:dtype 从 float32 改为 bfloat16,`alloc_tensors()` 移出计时函数。
- 新增 `sgl-kernel/tests/test_ep_moe_pre_reorder_kernel.py`:CUDA 与 Triton 结果对齐(fp16/bf16/fp32 × use_per_token_if_dynamic True/False,batch 32/64/128 × hidden 512/1024/2048 × topk 4/8,rtol/atol=1e-5)。

关键摘录:

```diff
+  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
+    ep_pre_reorder_cuda_kernel<scalar_t><<<grid, block>>>(
+        static_cast<scalar_t*>(input.data_ptr()),
+        static_cast<scalar_t*>(gateup_input.data_ptr()),
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:所有调用 `ep_moe_pre_reorder` 的 EP MoE pre-reorder 路径;新增尾部循环会改变 hidden_size 非 vec_size 整数倍时的输出(此前这些尾部元素不在向量化循环覆盖范围内,由 diff 推断);scale 提前按正文所述仅减少读取、逻辑等价;benchmark 计时口径变化(不再含张量分配)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
