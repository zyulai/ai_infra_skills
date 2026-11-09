# PR #4359: [FIX] fix incorrect output when enable both deepgemm and torch compile

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#4359](https://github.com/sgl-project/sglang/pull/4359) |
| 作者 | AniZpZ |
| 合并日期 | 2025-03-13 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
同时开启 deepgemm 与 torch.compile 时输出错误;把 deepgemm GEMM 封装为 torch custom op 并走 `torch.ops` 调用修复。

## 问题现象
同时启用 deepgemm(JIT deepgemm)与 torch.compile 时,模型输出不正确。PR 正文附有修复前后输出对比截图(H20 平台测试):修复前输出明显异常,修复后恢复正常。正文未描述具体日志或更多触发细节,仅说明该问题在 H20 平台上复现并验证。

## 根因
正文披露:原始 deepgemm ops 与 torch.compile 不兼容("We found original deepgemm ops is not compatible with torch compile"),直接在编译图中调用 deep_gemm 的 GEMM 函数会产生错误输出。

具体机制(由 diff 推断):torch.compile 追踪 `fp8_kernel.py` 中直接调用的 `deep_gemm.gemm_fp8_fp8_bf16_nt` 时,无法正确处理该外部函数对输出张量 C 的原地写入,导致编译后执行的结果不正确;将其注册为带 `mutates_args=["C"]` 与 fake impl 的 custom op,可让编译器把它当作不透明算子处理,保留原地写语义。

## 修复方案
改动集中在 `python/sglang/srt/layers/quantization/fp8_kernel.py`:

1. 新增条件注册(`supports_custom_op()` 为真时)一个名为 `deep_gemm_fp8_fp8_bf16_nt` 的 torch custom op,真实实现内部调用 `deep_gemm.gemm_fp8_fp8_bf16_nt((A, As), (B, Bs), C)`,fake impl 为空操作,并声明 `mutates_args=["C"]`。
2. 在 block-wise fp8 GEMM 分支中,把原来的直接调用改为经 custom op 调用:

```diff
     if _is_cuda and C.dtype == torch.bfloat16 and _enable_jit_deepgemm:
-        deep_gemm.gemm_fp8_fp8_bf16_nt((A, As), (B, Bs), C)
+        if supports_custom_op():
+            torch.ops.sglang.deep_gemm_fp8_fp8_bf16_nt(A, As, B, Bs, C)
+        else:
+            deep_gemm.gemm_fp8_fp8_bf16_nt((A, As), (B, Bs), C)
```

不支持 custom op 的环境保持原调用路径不变。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:仅影响 `_is_cuda && C.dtype == torch.bfloat16 && SGL_ENABLE_JIT_DEEPGEMM` 生效时的 block fp8 GEMM 路径;非 CUDA、非 bf16 输出或未启用 JIT deepgemm 的路径走原 triton kernel,不受影响。若 custom op 注册失败或 fake impl 与真实实现签名不一致,torch.compile 下可能出现新问题;作者已在 H20 平台验证。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
