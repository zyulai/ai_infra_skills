# PR #36100: [ROCm] Fix fused_moe_fake signature mismatch and other AITER bugs

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36100](https://github.com/vllm-project/vllm/pull/36100) |
| 作者 | ChuanLi1101 |
| 合并日期 | 2026-03-23 |
| labels | ready, rocm, v1 |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
`rocm_aiter_fused_moe` 的 fake 实现缺 4 个参数,torch.compile 下按 schema 传参会 TypeError 崩溃;补齐签名并顺带修 AITER 相关文案/拼写/注册路径问题。

## 问题现象
使用 ROCm AITER 的 `rocm_aiter_fused_moe` 自定义 op 时,在 `torch.compile` / FakeTensor 模式下发生 **TypeError 崩溃**:fake 实现不接受真实实现有的参数。正文明确给出触发路径:`mxfp4.py` 第 1162-1177 行的 MXFP4 CK backend 代码路径会调用
```python
rocm_aiter_ops.fused_moe(..., hidden_pad=..., intermediate_pad=..., bias1=layer.w13_bias, bias2=layer.w2_bias)
```
这些参数传给 op 时,fake 实现直接抛 TypeError。

## 根因
正文披露:`rocm_aiter_fused_moe` 的 fake/meta 实现(`_rocm_aiter_fused_moe_fake`)缺少真实实现有的 4 个参数(`hidden_pad`、`intermediate_pad`、`bias1`、`bias2`)。由于 `direct_register_custom_op` 是从 impl 签名推导 op schema 的,schema 允许调用方传这些参数;torch.compile / FakeTensor 模式下 PyTorch 会以相同参数调用 fake,而 fake 不接受,于是 TypeError。其余为小问题(复制粘贴错误信息、注释布局写错等),正文逐一列出。

## 修复方案
1. 主修复:`vllm/_aiter_ops.py` 给 `_rocm_aiter_fused_moe_fake` 补上 4 个缺失参数:
```python
+    hidden_pad: int = 0,
+    intermediate_pad: int = 0,
+    bias1: torch.Tensor | None = None,
+    bias2: torch.Tensor | None = None,
```
测试计划确认签名现与 `_rocm_aiter_fused_moe_impl` 完全一致(全部 19 个参数)。
2. `vllm/v1/attention/backends/rocm_aiter_fa.py`:两处 `AiterFlashAttentionImpl` 抛错信息误写为 `FlashAttentionImpl`,改正;KV cache 布局注释由 `[num_blocks, num_heads, page_size, head_dim]` 改为 `[num_blocks, page_size, num_heads, head_dim]` 以匹配实际索引;`only support`→`only supports`、`deocde`→`decode`。
3. `vllm/model_executor/layers/quantization/quark/quark_moe.py`:日志消息中变量标签 `use_mxfp4_aiter_moe` 改为 `use_rocm_aiter_moe`。
4. `_aiter_ops.py`:方法名拼写 `triton_fp4_gemm_dynamic_qaunt`→`triton_fp4_gemm_dynamic_quant`。
5. `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`:删除本地 `@cache` 的 `is_rocm_aiter_fp4_asm_gemm_enabled`,改用 `rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled()`,并移除不再需要的 `cache`/`envs` 导入(diff 可见,配套上面改名)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:ROCm AITER 路径。fake 签名补齐只新增带默认值的参数,不改变 op schema 与已有序列化行为;主要收益是 MXFP4 CK 后端在 torch.compile 下不再崩溃。`triton_fp4_gemm_dynamic_quant` 改名与 quark_ocp_mx 改用 `rocm_aiter_ops` 判断会影响引用旧拼写的调用点(需同步,diff 内已同步);rocm_aiter_fa 仅注释/报错文案变化,无逻辑改动。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
