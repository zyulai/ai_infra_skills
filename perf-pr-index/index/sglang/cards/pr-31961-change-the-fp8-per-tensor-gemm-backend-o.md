# PR #31961: Change the FP8 per-tensor GEMM backend on SM120 to cuBLAS

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @b8zhong |
| 合并 | 2026-07-21 |
| 分类 | Quantization (tier: major) |
| Label | quant, run-ci |
| Diff | pack 未提供 diffstat(diff 完整可见,改 2 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/31961 |

## 一句话结论

SM120 上 FP8 per-tensor GEMM 从 sgl-kernel CUTLASS 换到 flashinfer `bmm_fp8`(cuBLAS),M=1 时 TPS 55 → 65.59(约 20%)。

## 动机

SM120(Blackwell 消费/工作站级)此前不在 flashinfer `bmm_fp8` 路径的启用范围,FP8 per-tensor static linear 走 sgl-kernel 的 CUTLASS GEMM。PR 的动机极简:"Just call bmm_fp8 of Flashinfer API (which call cuBLAS)"——用现成 cuBLAS 后端替换自维护 kernel,把 SM120 纳入与 SM100 相同的路径。

## 优化点

机制是启用条件放宽而非新 kernel:`ModelOptFp8LinearMethod.__init__` 中 `enable_flashinfer_bmm` 从 `is_sm100_supported() and is_flashinfer_available()` 改为 `(is_sm100_supported() or is_sm120_supported()) and is_flashinfer_available()`,使 SM120 也走 flashinfer `bmm_fp8`(底层 cuBLAS),不再调用 sgl-kernel;`apply_fp8_linear_bmm_flashinfer` 的 docstring 同步从 "(SM10X only)" 改为 "(SM100/SM120 Blackwell)"。取舍说明:PR 正文承认 CUTLASS 或许还有 5-10% 性能空间,但认为不值得维护自定义 kernel 与代码。trace 显示新路径用的是 cuBLAS 的 SM89 kernel(SM89 与 SM120 架构相近);未来 Torch 升级带新 cuBLAS 后会生成 `nvjet_sm120_qqtst_mma`(TMA 等进一步优化)。

## 关键改动

```text
python/sglang/srt/layers/quantization/modelopt_quant.py: enable_flashinfer_bmm 增加 is_sm120_supported() 条件
python/sglang/srt/layers/quantization/fp8_utils.py: apply_fp8_linear_bmm_flashinfer 适用范围说明更新为 SM100/SM120
```

```python
self.enable_flashinfer_bmm = (
    is_sm100_supported() or is_sm120_supported()
) and is_flashinfer_available()
```

## 性能声明

- 测试条件:`sglang serve --model-path nvidia/Qwen3.6-27B-NVFP4 --reasoning-parser qwen3 --tool-call-parser qwen3_coder`;M=1(单请求)TPS 从 55 提升到 65.59,约 20%("55 -> 65.59 TPS (around 20%) for M = 1");pack 未注明 GPU 具体型号(SM120 语境)与其他 workload 参数
- Profile 对比:CUTLASS FP8 GEMM vs cuBLAS 后端的两张 trace 截图,无具体 µs 数字
- CUTLASS 相对 cuBLAS "maybe further 5-10% perf improvement"(PR 正文原话,为不切换的代价估计,非实测)

## 验证与回归风险

Accuracy Tests 一节为空,PR 未提供精度测试数据;CI Base/Extra 在 PR 时点均为失败状态(pack 记录)。回归面:仅影响 `enable_flashinfer_bmm=True` 时的 ModelOpt FP8 per-tensor static linear 路径,SM120 新增走 flashinfer bmm_fp8(SM100 行为不变);数值上从 CUTLASS 换到 cuBLAS 会改变 GEMM 实现的舍入,PR 未量化;依赖 flashinfer 版本提供 SM120 支持与 cuBLAS 后端。

## 关联

同区域:ModelOpt FP8 quantization 的 flashinfer bmm_fp8 路径(SM100 先行);pack 正文未列出关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
