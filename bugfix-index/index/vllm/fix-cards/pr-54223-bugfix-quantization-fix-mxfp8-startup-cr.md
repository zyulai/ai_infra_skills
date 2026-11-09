# PR #54223: [Bugfix][Quantization] Fix MXFP8 startup crash on layers below mm_mxfp8 shape limits

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54223](https://github.com/vllm-project/vllm/pull/54223) |
| 作者 | samuelkim7 |
| 合并日期 | 2026-09-25 |
| labels | bug, minimax, nvidia, quantization, ready, rocm, verified |
| 热度 | 👍 2 · 💬 9 |
| 分类 | Quantization |
| bug 类型 | crash |

## 一句话结论
MXFP8 两个 FlashInfer 内核对任意形状都返回可实现,低于 mm_mxfp8 形状下限的层启动即断言崩溃,修复为在 can_implement 中做形状校验并回落 Marlin。

## 问题现象
`vllm serve Qwen/Qwen3.5-0.8B --quantization mxfp8` 在 B200 上 profile run 阶段崩溃:`AssertionError: mm_mxfp8 requires N >= 128, got N=32`,问题层是 GDN 的 `in_proj_ba`(权重 32 x 1024)。更大的 Qwen3.5 尺寸与 Qwen3-Next 在该层为 N=64 或 N=96,因此整个模型家族用 online MXFP8 都无法启动。正文对比测试(B200、vLLM 0.28.1rc1.dev49+gd1922cb5a、FlashInfer 0.6.17、CUDA 13.0):main 分支 52 秒后启动失败、无任何输出;本 PR 启动正常,`in_proj_ba` 由 Marlin 内核接管(pad 到 N=64,日志有 Marlin padding 警告),其余层仍用 FlashInfer 内核,输出 ` Paris.`。

## 根因
正文披露:`FlashInferCutlassMxfp8LinearKernel` 与 `FlashInferCutedslMxfp8LinearKernel` 的 `can_implement` 对所有层都返回 True;mm_mxfp8 的形状限制(N >= 128、K >= 128、K % 32 == 0)只是 `apply_weights` 内部的 assert,于是在 SM100 上低于限制的层仍被选中,而不是回落到 Marlin。此外 compressed-tensors、INC 与 online MXFP8 方案在 scheme 的 `__init__` 里就选内核,那时权重形状未知(ModelOpt 已在 #49381 后改为 `create_weights` 中选择)。

## 修复方案
- `Mxfp8LinearLayerConfig` 新增二元组 `(N, K)` 字段 `weight_shape`;`init_mxfp8_linear_kernel` 增加 `weight_shape` 必填参数。
- `vllm/model_executor/kernels/linear/mxfp8/flashinfer.py`:新增共享校验器 `_check_mm_mxfp8_shape`(K >= 128、K % 32 == 0、N >= 128),由两个 FlashInfer 内核的 `can_implement` 调用;CUTLASS 内核额外加 SM12x 的 `N % 32 != 0` 拒绝;`apply_weights` 中的 assert 移除(无内核代码改动)。
- `marlin.py`:`can_implement` 拒绝 N <= 0、K <= 0、K % 32 != 0。
- compressed-tensors、INC、online 三个方案把内核选择从 `__init__` 移到 `create_weights`(即 `Fp8LinearMethod` 已用的每层模式),传 `layer.weight.shape`;ModelOpt 在 BMM 重选点取末尾两维传入。

```python
def _check_mm_mxfp8_shape(weight_shape):
    N, K = weight_shape
    if K < 128: return False, f"mm_mxfp8 requires K >= 128, got K={K}."
    if K % MXFP8_BLOCK_SIZE != 0: return False, ...
    if N < 128: return False, f"mm_mxfp8 requires N >= 128, got N={N}."
    return True, None
```

新增 CPU 选择测试 `tests/kernels/quantization/test_mxfp8_kernel_selection.py`(164 passed / 11 skipped)。

## 影响与回归风险
- 回归面:所有 MXFP8 量化路径的内核选择——低于 mm_mxfp8 形状限制的层现在回落到 Marlin(可能略慢但能跑);`Mxfp8LinearLayerConfig` 构造签名变更影响全部 15 处构造点与相关测试;SM12x 上 N % 32 != 0 的层改由其他内核承担(@stecasta 在 DGX Spark 上测得相对全 Marlin 的吞吐收益);Marlin 对非法形状的拒绝是新语义。

## 关联
#49381(已把 ModelOpt 选择移入 create_weights,本 PR 基于其上)、#51808、#52275

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
