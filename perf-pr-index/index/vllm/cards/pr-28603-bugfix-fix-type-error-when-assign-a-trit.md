# PR #28603: [BugFix] Fix type error when assign a trition kernel tensor to a torch.nn.Parameter

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @liuzijing2014 |
| 合并 | 2025-11-13 |
| 分类 | Attention / Kernel（tier: major） |
| Label | ready |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失）；实际 diff 仅 mxfp4.py 两行替换 |
| 链接 | https://github.com/vllm-project/vllm/pull/28603 |

## 一句话结论

修复 triton kernel tensor 直接赋给 `torch.nn.Parameter` 导致的 MXFP4 权重加载 TypeError，恢复 gpt-oss 类模型启动。

## 动机

PR #28064 引入回归：`Mxfp4MoEMethod.process_weights_after_loading` 把 triton kernel 产生的 `triton_kernels.tensor.Tensor` 直接赋给 `layer.w13_weight`/`layer.w2_weight`，而这两者是 `torch.nn.Parameter` 槽位，PyTorch 的 `__setattr__` 校验抛出 `TypeError: cannot assign 'triton_kernels.tensor.Tensor' as parameter 'w13_weight' (torch.nn.Parameter or None expected)`。错误发生在 `process_weights_after_loading` 阶段，WorkerProc 启动失败，使用该路径的模型（如 gpt-oss）完全无法启动，属于可用性阻断而非性能回归。

## 优化点

无机制层面的优化，是两行类型修复：把裸张量赋值改为用 `Parameter(w13_weight.data, requires_grad=False)` 包裹后赋给 layer，使赋值对象满足 `torch.nn.Module.__setattr__` 对 Parameter 槽位的类型要求，同时不复制数据（`.data` 共享存储）、保持无梯度语义。

## 关键改动

```text
vllm/model_executor/layers/quantization/mxfp4.py: process_weights_after_loading 中 SM90 cutlass interleave 分支的两行赋值修复
```

```python
-            layer.w13_weight = w13_weight
-            layer.w2_weight = w2_weight
+            layer.w13_weight = Parameter(w13_weight.data, requires_grad=False)
+            layer.w2_weight = Parameter(w2_weight.data, requires_grad=False)
```

## 性能声明

无优化前后对比的性能声明。正文中的压测输出为修复后功能冒烟结果：100/100 请求成功率 100.00%，QPS 3.43，Total TPS 24,290.44（Input TPS 23,624.57）；硬件/模型/workload 条件 PR 未注明。另附 aime25_gpt_oss 评测 exact_match 0.7833（stderr 0.0266），同样未给基线对照。

## 验证与回归风险

- 验证：修复后在 vLLM 上重新启动成功（"Run on vLLM again and able to start vLLM"）+ 上述冒烟压测与 aime25 评测通过；PR 未新增针对性单元测试（diff 中无测试文件）。
- 回归风险：仅影响 MXFP4 后端为 cutlass SM90 interleave 的加载路径；`requires_grad=False` 的 Parameter 包裹不改变权重数值与内存布局，风险面很小；其他 mxfp4 backend 分支（会 raise ValueError 的部分）未触碰。

## 关联

- 引入回归的 PR：#28064（本卡素材未含其正文，仅 PR 正文点名）。
- 同分类：本批 Attention/Kernel PR #57057、#51739；同区域 MXFP4 量化 PR #51473、#45739、#28892、#32520。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
