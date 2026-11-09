# PR #14553: Fix FP8 KV Triton type issue and add regression test

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14553](https://github.com/sgl-project/sglang/pull/14553) |
| 作者 | harvenstar |
| 合并日期 | 2025-12-07 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 8 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
B200 上 fused FP8 KV cache 路径把 0-D tensor 的 k_scale/v_scale 传给 Triton kernel,`1.0/k_scale` 触发 IncompatibleTypeError 打挂 Llama-3.1-FP4 CI(#14093 因此被回退);改为 GPU 上预算逆 scale 并以指针传入修复。

## 问题现象
B200 上 TRTLLM MHA 的 fused FP8 KV cache 路径出现 Triton `pointer<fp32>` vs `float32` 类型错误,打挂 Llama-3.1-FP4 CI;该问题曾导致 #14093 在 #14550 被回退。原始错误发现于 CI run sgl-project/sglang run 19985523381 / job 57335123826(pr=14493)。

## 根因
正文披露:
- `layer.k_scale`/`v_scale` 是 `torch.nn.Parameter`(0-D tensor);
- Triton 把 tensor 参数解释为指针;
- kernel 内表达式 `1.0 / k_scale` 因此触发 `IncompatibleTypeError`。

## 修复方案
改 `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py`:

- wrapper 新增 `_to_tensor_scale()` 显式处理 None/Tensor/scalar 三种情况,把 scale 统一为 GPU 上的 0-D fp32 tensor;逆 scale `1.0/k_scale` 改在 GPU 上用 tensor 运算算好,再以 `inv_k_scale_ptr/inv_v_scale_ptr` 指针传入 kernel,kernel 内改为 `tl.load(inv_k_scale_ptr)`。diff 注释说明:这样可避免 `float(k_scale)` 触发 GPU→CPU 同步在 CUDA graph capture 期间抛 `cudaErrorStreamCaptureUnsupported`;`use_provided_scale=False` 时传 dummy 指针(`k_3d`),kernel 用常量 1.0,Triton 常量折叠后不会访问该指针,避免 capture 期间新建 GPU tensor。
- 新增回归测试 `test_fp8_kv_kernel_accepts_tensor_scales`,另加 CUDA graph capture(带 scale / 不带 scale)两个兼容性测试。

关键 diff 摘录(kernel 侧):

```diff
-        inv_scale = 1.0 / k_scale if use_provided_scale else 1.0
+        if use_provided_scale:
+            inv_scale = tl.load(inv_k_scale_ptr)
+        else:
+            inv_scale = 1.0
```

验证:`pytest test/manual/test_trtllm_fp8_kv_kernel.py -vv` 全部 11 个 kernel 测试(含新增回归测试)通过;`ATTN_BACKEND=trtllm_mha pytest test/srt/test_llama31_fp4.py::TestLlama31FP4B200::test_gsm8k -vv` 在 B200 通过。

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围;受影响场景为 B200 + TRTLLM MHA + FP8 KV cache 且 k_scale/v_scale 为 0-D tensor 的组合。
- 回归面:仅 `trtllm_fp8_kv_kernel.py` 的 scale 传参与 kernel 签名,per-tensor scaling 语义保持;逆 scale 从 kernel 内除法移到 wrapper GPU 计算,数值结果不变但多出两个极小的 0-D tensor 分配(scale 未提供时用 dummy 指针规避);CUDA graph capture 兼容性由新增测试覆盖。其他量化路径无改动。

## 关联
- PR #14093(被回退的原始改动): https://github.com/sgl-project/sglang/pull/14093
- PR #14550(回退 #14093): https://github.com/sgl-project/sglang/pull/14550
- CI 失败日志: https://github.com/sgl-project/sglang/actions/runs/19985523381/job/57335123826?pr=14493

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
