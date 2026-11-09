# PR #46518: [Kernel][MoE] Allow FlashInfer MXINT4 MoE for gated SiLU

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @kjiang249 |
| 合并 | 2026-06-24 |
| 分类 | MoE(tier: major) |
| Label | performance, ready, nvidia |
| Diff | diffstat 缺失(素材未提供 +/− 统计);diff 素材完整可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/46518 |

## 一句话结论

解除 FlashInfer TRT-LLM MXINT4 MoE 对 gated SiLU 的误拒,让 Kimi-K2.5 类模型能选中该后端。

## 动机

TrtLlmMxint4ExpertsMonolithic._supports_activation() 原来只接受 MoEActivation.SWIGLUOAI,拒绝 MoEActivation.SILU,导致 vLLM 不会为 Kimi-K2.5 这类 gated-SiLU MoE 层选择 FlashInfer TRT-LLM MXINT4 后端。而底层 FlashInfer trtllm_mxint4_block_scale_moe 本身支持标准 gated SiLU 计算,属 Python 侧能力判定过严。

## 优化点

纯后端选择/能力元数据修复,不改任何 GPU kernel:_supports_activation() 改为同时接受 MoEActivation.SILU 与 MoEActivation.SWIGLUOAI(FlashInfer 把标准 gated SiLU 命名为 "SwiGLU",在 vLLM MoE 配置中对应 SILU/silu_and_mul;SWIGLUOAI 作为别名保留以对齐其他 FlashInfer 后端);其余激活(如 RELU2_NO_MUL)继续拒绝。新增回归单元测试覆盖该判定。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/experts/trtllm_mxint4_moe.py: _supports_activation 放宽到 (SILU, SWIGLUOAI)
tests/kernels/moe/test_marlin_vs_trtllm_mxint4.py: 新增 activation 判定回归测试
```

```python
@staticmethod
def _supports_activation(activation: MoEActivation) -> bool:
    # FlashInfer MxInt4 names the standard gated SiLU path "SwiGLU".
    return activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI)
```

## 性能声明

PR 未给出 kernel 或端到端的性能对比数字(仅单元测试通过)。端到端运行验证:Kimi-K2.5 TP8 serve(VLLM_USE_FLASHINFER_MOE_INT4=1)日志确认选中 "Using 'FLASHINFER_TRTLLM' WNA16 MoE backend";GSM8K flexible 0.9727 / strict 0.9727。属可用性/正确性验证,收益体现为启用 MXINT4 MoE 后端这一选择本身。

## 验证与回归风险

测试:tests/kernels/moe/test_marlin_vs_trtllm_mxint4.py::test_trtllm_mxint4_activation_supports_vllm_gated_silu(断言 SILU/SWIGLUOAI 通过、RELU2_NO_MUL 拒绝)+ 上述 TP8 端到端 serve。风险很小:仅影响后端选择判定,不匹配的激活仍被拒;可能改变 Kimi-K2.5 等模型默认落到的 MoE kernel,数值路径切换依赖 FlashInfer kernel 精度。

## 关联

同分类 MoE 卡:#33506(同样调整 FlashInfer MoE 的 activation 支持判定,非 gated FP8/NVFP4)、#65542(其验证引用了 test_marlin_vs_trtllm_mxint4.py)。无前序/后续 PR 声明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
