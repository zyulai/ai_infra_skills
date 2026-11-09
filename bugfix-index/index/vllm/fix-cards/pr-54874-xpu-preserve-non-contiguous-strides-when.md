# PR #54874: [XPU] Preserve non-contiguous strides when pinning CPU tensors for UVA view

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54874](https://github.com/vllm-project/vllm/pull/54874) |
| 作者 | chaojun-zhang |
| 合并日期 | 2026-09-29 |
| labels | bug, intel-gpu, ci/build |
| 热度 | 👍 0 · 💬 42 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论

XPU 上从非 pinned CPU 张量取加速器视图前先做 contiguous(),丢掉非连续 stride,导致 GEMM 时反复隐式拷贝拖慢性能;改用 empty_strided 保 stride。

## 问题现象

从非 pinned 的 CPU 张量获取 XPU 加速器视图(UVA view)时,旧实现先调用 `cpu_tensor.contiguous()` 再固定内存,把非连续 strided 视图(如转置的权重 scale 张量 `scale_kn.t()`,shape [16,32]、stride (1,16))转换成连续副本,丢掉原始 stride 模式;之后对这些连续副本再 `.t()` 又变回非连续,在 GEMM 执行期间触发额外的隐式 `.contiguous()` 拷贝,造成性能退化。无崩溃、无错误输出,属性能问题。

## 根因

正文披露:`get_accelerator_view_from_cpu_tensor` 的 XPU 分支在为非 pinned 张量制作 pinned 副本时,先 `contiguous()` 再 `torch.empty_like(..., pin_memory=True)` 拷贝,pinned 副本总是连续布局,原始 stride 信息在这一步被丢弃;下游再转置时产生非连续张量,GEMM 路径被迫做隐式连续化拷贝。

## 修复方案

改 `vllm/utils/torch_utils.py` 的 XPU 分支:不再先 contiguous,而是按原张量的 size 与 stride 直接分配 pinned 内存并拷贝,然后返回视图:

```python
pinned = torch.empty_strided(
    cpu_tensor.size(), cpu_tensor.stride(),
    dtype=cpu_tensor.dtype, device="cpu", pin_memory=True,
)
pinned.copy_(cpu_tensor)
return torch.ops._C.get_xpu_view_from_cpu_tensor(pinned)
```

由此非连续 stride 在 pinned 副本上得以保留。配套:`tests/kernels/core/test_uva.py` 新增 `test_non_contiguous_strided_view`(仅 XPU,pinned/unpinned 两种参数,断言视图 stride 为 (1,16)、转置回 [32,16] 后连续、值与原 CPU 张量一致),并新增 `test_non_pinned_cpu_tensor` 说明非 pinned 张量的视图是创建时的私有副本、与后续 CPU 写解耦;测试文件由 CUDA 专用泛化为 `current_platform.device_type`;`.buildkite/intel_jobs/kernels_intel.yaml` 新增 UVA Kernel Tests CI 任务。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:仅 XPU 的非 pinned CPU 张量视图路径(方法其余分支与 CUDA alike 分支不变)。修复后返回的视图保留原 stride,依赖「返回视图必连续」假设的调用方需复查;pinned 副本仍与原 CPU 张量写操作解耦(私有拷贝,测试已固化该语义);测试文件从 cuda 硬编码改为平台泛化,涉及 Stream/Event API 换用 `torch.Stream`/`torch.Event` 与 accelerator 接口,对非 XPU 平台的测试行为有重构风险。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
