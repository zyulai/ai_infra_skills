# PR #34543: [Bugfix] Fix ROCm UVA CPU weight offloading broken by #32993

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34543](https://github.com/vllm-project/vllm/pull/34543) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-02-14 |
| labels | bug, ready, rocm |
| 热度 | 👍 0 · 💬 3 |
| 分类 | KV Cache / Memory |
| bug 类型 | regression |

## 一句话结论
#32993 的 UVA CPU 权重卸载只查 CUDA/XPU 平台,ROCm 上抛 ValueError;在 CUDA 分支补上 is_rocm() 判定。

## 问题现象
ROCm 上使用 UVA(Unified Virtual Addressing)CPU 权重卸载时抛出 `ValueError`——`get_accelerator_view_from_cpu_tensor` 的平台分支只认 CUDA 和 XPU,ROCm 落到 else 的 raise。

## 根因
正文披露:回归由 #32993 引入。该 PR 为 CPU 权重卸载添加 UVA 支持,但在 `get_accelerator_view_from_cpu_tensor` 中只写了 CUDA 与 XPU 的平台检查,漏掉 ROCm。C++ 层在 ROCm 上本来就能工作:`cuda_view.cu` 会被自动 hipify,且 op 注册没有 ROCm guard;缺的只是 Python 层的平台判断。

## 修复方案
修改 `vllm/utils/torch_utils.py` 的 `get_accelerator_view_from_cpu_tensor`:在已有 CUDA 分支上叠加 `current_platform.is_rocm()`——正文解释理由是 ROCm 与 CUDA 共享相同 device type 与 dispatch key,因此走同一个 `get_cuda_view_from_cpu_tensor` custom op:

```diff
-    elif current_platform.is_cuda():
+    elif current_platform.is_cuda() or current_platform.is_rocm():
         return torch.ops._C.get_cuda_view_from_cpu_tensor(cpu_tensor)
```

测试计划:`tests/kernels/core/test_uva.py`、`tests/basic_correctness/test_cpu_offload.py`、`tests/quantization/test_cpu_offload.py`。

## 影响与回归风险
- 影响窗口:正文未给出具体版本号;回归自 #32993 加入 UVA 支持起,至本修复合并前。
- 回归面:仅 `get_accelerator_view_from_cpu_tensor` 的平台分发;CUDA/XPU 路径不变,ROCm 从报错变为调用同一 C++ op。该 op 在 ROCm 上依赖 hipify 后的 `cuda_view.cu` 与未加 guard 的 op 注册(正文所述),若未来给该 op 加 ROCm guard,这里会再次失效(由 diff 推断)。

## 关联
- #32993(正文指明的回归引入 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
