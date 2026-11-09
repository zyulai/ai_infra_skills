# PR #49209: [Hardware][XPU] Register matmul and linear batch-invariant kernels for XPU

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tzielinski-habana |
| 合并 | 2026-09-02 |
| 分类 | Attention / Kernel |
| Label | documentation, intel-gpu, ready, v1 |
| Diff | diffstat 缺失(pack 未提供);diff 可用,涉及 determinism 模块、linear/embedding 层、engine args、测试与文档 |
| 链接 | https://github.com/vllm-project/vllm/pull/49209 |

## 一句话结论

为 XPU 补齐 batch-invariant 的 matmul/linear 内核并新增 Tensor Descriptor Triton GEMM(比指针版快约 3x),使 Intel GPU 也能开启 `VLLM_BATCH_INVARIANT`。

## 动机

这是 XPU batch invariance 系列的第 2/2 个 PR(前序 #41934)。上一个 PR 跳过了 matmul/linear,因为需要更多工作:作者在 RL 场景验证时,backward 路径因 "XPU" dispatch key 未注册 `aten::matmul`/`aten::linear` 而报错;同时希望引入基于 Tensor Descriptors 的新 Triton matmul——其在 XPU 上比基于指针的 persistent 实现快约 3x。

## 优化点

(1) 新增 `matmul_kernel_descriptor_persistent`:用 `tl.make_tensor_descriptor` 做 2D block 加载/存储,借助硬件 2D block I/O 与自动越界检查省去显式 mask,B 以 [N, K] 转置布局接收、`tl.dot(a, b.T)` 还原,persistent 调度按 NUM_SMS 分 tile;(2) `mm_batch_invariant` 在 XPU 上路由到该 descriptor 内核,并注册 `aten::matmul`、`aten::linear`(替换原 TODO),`linear.py`/`vocab_parallel_embedding.py` 的 batch-invariant 分支条件加入 `is_xpu()`;(3) 新增 `_get_descriptor_matmul_config`:按 M 分档调优(M<=16 decode 用 16×256 小 M 块宽 N 块;M<=64;M>64 用 64×128),BLOCK_SIZE_K 仅由 dtype 决定,保证 K 归约顺序与 batch 大小无关——batch invariance 语义不破;(4) XPU 上 oneDNN 的 softmax/log_softmax/mean/bmm 本就逐位确定,跳过这些 override;`VLLM_BATCH_INVARIANT=1` 且未指定 attention backend 时在 XPU 默认 TRITON_ATTN,并移除相关测试的 xfail。

## 关键改动

```text
vllm/model_executor/determinism/batch_invariant.py: descriptor persistent GEMM + XPU 算子注册
vllm/model_executor/determinism/batch_invariant_configs.py: 按形状/dtype 的 descriptor 配置
vllm/engine/arg_utils.py: XPU + batch-invariant 默认 TRITON_ATTN
vllm/model_executor/layers/linear.py、vocab_parallel_embedding.py: XPU 走 linear_batch_invariant
tests/v1/determinism/test_batch_invariance.py: 移除 XPU xfail;docs/features/batch_invariance.md 更新
```

```python
 def mm_batch_invariant(a, b, bias=None):
+    if a.device.type == "xpu":
+        return matmul_descriptor_persistent(a, b, bias=bias)
     return matmul_persistent(a, b, bias=bias)
...
+        _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, key)
+        _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, key)
```

## 性能声明

PR 正文唯一的性能数字是内核级:"that implementation has shown ~3x speedup on XPU vs. the one based on pointers"(descriptor 版相对指针版 persistent matmul),测试条件(具体 XPU 型号、dtype、形状)未注明。E2E 推理吞吐/延迟数字 PR 未提供;Test Result 仅写 "All tests pass"。

## 验证与回归风险

验证:XPU batch invariance 单元测试(移除 xfail 后全过);docs 更新给出 XPU 上 `VLLM_BATCH_INVARIANT=1` + `TRITON_ATTN` 的用法。风险面:descriptor 内核要求 contiguous 输入并临时转置 B(`b.t().contiguous()` 产生额外拷贝,PR 未量化其成本);`triton.set_allocator` 全局状态须在 torch.compile 区域外设置(代码已注释说明);CUDA 路径行为不变(softmax/bmm 等 override 仅注册于非 XPU);NUM_SMS 依赖 `num_compute_units`,`_NUM_SMS` 需在 Dynamo trace 前确定。该 PR 目标是确定性而非吞吐,~3x 是相对自家旧实现。

## 关联

前序:#41934(XPU batch invariance 1/2)。同批无其他 XPU/确定性卡。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
