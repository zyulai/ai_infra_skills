# PR #17369: [Fix] Revert back to using CUTLASS `mm_fp4` backend

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#17369](https://github.com/sgl-project/sglang/pull/17369) |
| 作者 | b8zhong |
| 合并日期 | 2026-01-31 |
| labels | quant, run-ci |
| 热度 | 👍 0 · 💬 9 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
FP4 GEMM 后端为 auto 时会选中 cudnn 后端且表现更差,回退为 auto 时固定使用 cutlass。

## 问题现象
`mm_fp4` 后端在 cudnn >= 9.15.xx 且 CUDA 13 的环境下会被自动选中;作者手动安装 `nvidia-cudnn-cu13==9.18.0.77` 后表现"仍然更差"(still seems to be worse),即该自动选择会把 API 指到不正确的后端。问题在 flashinfer 侧以 issue #2375 报告。正文未描述具体报错日志或精度错误现象,主要以性能/后端选择错误呈现。

## 根因
正文披露:mm_fp4 的自动选择逻辑在"cudnn >= 9.15.xx 且 CUDA 13"条件下会挑中 cudnn 后端,而该后端表现比预期更差,即使升级到 cudnn 9.18.0.77 也未改善,导致 API 选中不正确的后端。这是此前引入自动后端选择行为后的回归,故本 PR 将行为回退到更早的行为(earlier behaviour)。

## 修复方案
改 `python/sglang/srt/layers/quantization/modelopt_quant.py` 的 `fp4_gemm`:当 FP4 GEMM runner backend 为 auto 时,不再透传给 flashinfer 让其自行挑选,而是显式固定为 `"cutlass"`;仅当用户显式配置了非 auto 后端时才使用 `get_flashinfer_backend()` 的映射结果。

关键摘录:
```diff
-        backend = fp4_backend.get_flashinfer_backend()
+        backend = (
+            fp4_backend.get_flashinfer_backend()
+            if not fp4_backend.is_auto()
+            else "cutlass"
+        )
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围(与引入 flashinfer mm_fp4 自动后端选择之后的版本相关)。
- 回归面:改动仅触及 `modelopt_quant.py` 中 `fp4_gemm` 的后端选择分支。所有走 auto 后端的 modelopt FP4 量化 GEMM 将固定使用 cutlass;显式配置了具体后端的用户行为不变。等 cudnn 侧问题解决前,auto 用户无法再自动尝试 cudnn/cublas 等其他后端。

## 关联
- https://github.com/flashinfer-ai/flashinfer/issues/2375

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
