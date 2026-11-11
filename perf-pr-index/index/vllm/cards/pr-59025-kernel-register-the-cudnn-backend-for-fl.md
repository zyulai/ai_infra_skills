# PR #59025: [Kernel] Register the cudnn backend for FlashInfer BF16 mm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @schopde-nvidia |
| 合并 | 2026-11-10 |
| 分类 | Attention / Kernel(major) |
| Label | performance, ready, nvidia, verified |
| Diff | +36/-5,1 file(素材未提供 diffstat,按 pack diff 计数) |
| 链接 | https://github.com/vllm-project/vllm/pull/59025 |

## 一句话结论

为 FlashInfer BF16 mm 注册 cudnn backend,补上 consumer Blackwell(sm_120/121)无 unquantized BF16 FlashInfer 路径的空白;窄输出宽度(N<=128)下比 cuBLAS 快最多约 6x。

## 动机

之前 unquantized BF16 linear 只接了 cute-dsl backend,而它被限制在 sm_10x(FlashInfer 对 cute-dsl 在 sm_120 抛 `BackendSupportedError`,该限制本身是正确的)。但 `mm_bf16` 还有 cudnn backend 可以在 sm_120/121 运行——因此 consumer Blackwell(sm_120/121)此前对 unquantized BF16 完全没有 FlashInfer 路径可选。

## 优化点

把 shape/layout 检查从 cute-dsl 的 predicate 中拆出为共享的 `_bf16_mm_shape_supported`;cute-dsl 单独保留 sm_10x gate;在 `_FLASHINFER_BF16_BACKENDS` 注册新条目 `flashinfer_cudnn`,其 `can_implement`(`_can_use_flashinfer_cudnn_bf16`)要求 sm_120 family 且输出宽度 `N <= _FLASHINFER_BF16_MAX_N = 128`。N 上限来自 GB10(sm_121,K=5120)的 sweep:N<=128 时 FlashInfer 赢 5-6x,N=512 时输 2-4x,所以不做无差别路由——混合宽度模型上把所有 unquantized GEMM 都导过去会净亏。sm_10x 在同一 regime 实测 0.90-1.13x 无收益,故不启用。默认 selection 完全不变(`linear_backend` 未设置仍解析为 `default_unquantized_gemm`),仅显式 `--linear-backend flashinfer_cudnn` 可达(该值此前已是 NVFP4 接受的 `LinearBackend` 值)。

```python
_FLASHINFER_BF16_MAX_N = 128

def _can_use_flashinfer_cudnn_bf16(x, weight, bias):
    # sm_10x showed no benefit in the same regime (measured 0.90-1.13x).
    if not (current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)):
        return False
    if weight.shape[0] > _FLASHINFER_BF16_MAX_N:
        return False
    return _bf16_mm_shape_supported(x, weight, bias)
```

## 关键改动

```text
vllm/model_executor/layers/utils.py: 唯一改动文件——拆出共享 _bf16_mm_shape_supported;注册 flashinfer_cudnn backend(仅 sm_120 family,N<=128)
```

## 性能声明

GB10(sm_121,K=5120)sweep,表中数值为 cuBLAS_us / flashinfer_us(>1 表示 FlashInfer 赢),M=1/2/4/8/16/32:N=96 行为 1.09/6.14/6.15/1.26/1.28/1.20;N=128 为 1.01/5.38/1.26/1.35/1.42/1.51;N=192 为 0.85/4.31/1.17/1.26/1.38/1.31;N=256 为 1.14/1.38/1.26/1.37/0.89/0.96;N=512 为 0.27/0.37/0.41/0.53/0.63/0.75。峰值收益在 M=2-4、N<=128(约 5.4-6.2x),宽 N 反向(最差 0.27x)。无 e2e 吞吐数字;PR 未注明测试所用的精度上下文(BF16 mm 本身)之外的更多条件。

## 验证与回归风险

验证较薄:PR 仅给出 `ast.parse` 语法检查与"默认 selection 不变"的断言(linear_backend 未设置时行为不变;新条目仅显式 opt-in 可达),未附 GPU 单测。回归风险面:sm_120/121 上显式 opt-in 的 unquantized BF16 linear 路径(新启用 cudnn backend);N<=128 的边界只由 GB10、K=5120 单一 sweep 支撑,其他 K 值或硬件未在 PR 注明;与 #59021 的 `--unquantized-linear-backend` 组合使用时的行为依赖该 flag 的实现。

## 关联

#57318(GDN 专用 custom op,把 `in_proj_ba` 路由到 FlashInfer;其输出宽度 `2 * num_v_heads` 在所有已发布 Qwen GDN 模型 <=96,本 PR 落地后经 `--linear-backend flashinfer_cudnn` 自动覆盖,该 bespoke op 可简化或移除);#59021(`--unquantized-linear-backend` flag)。本批同分类:#66037(FlashInfer sparse MLA)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
