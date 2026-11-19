# PR #63272: [ROCm][Quantization][Perf] Fuse the block Hadamard input rotation into AITER's MXFP4 activation quant

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mjkvaak-amd |
| 合并 | 2026-11-16 |
| 分类 | Quantization（tier: major） |
| Label | documentation, rocm, ready, ci/build, quantization |
| Diff | pack 标注 +None/-None, None files；pack 末尾无「N file(s) dropped」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/63272 |

## 一句话结论

ROCm 上 `mxfp4_hadamard32` 层把 32 宽 block Hadamard 输入旋转融进 AITER ASM MXFP4 GEMM 的 HIP 激活量化（ROCm/aiter#6734），把 #62750 PyTorch 旋转在 Qwen3.8-2.4T 上造成的 6–12% serving 吞吐损失收回（fused 距 plain RTN mxfp4 仅 0.1–0.6%）。

## 动机

#62750 的 Hadamard 输入旋转以单独 PyTorch kernel 运行（`block_hadamard_rotate` 再交给被包装 method），对量化 + GEMM 增加 70–170% 每层耗时，在 Qwen3.8-2.4T 上损失 6–12% serving 吞吐。AITER 的 ASM MXFP4 GEMM 走 HIP 激活量化 kernel，可以把旋转搬进量化内部（上游 ROCm/aiter#6734 提供 `hadamard_rotate` 参数）。非重复性：检索 open PR "hadamard mxfp4"/"hadamard rotation aiter"，#65398 只加该 online 方案的 MoE 变体、不做旋转融合。

## 优化点

- **`InputHadamardFusable` mixin**（`input_hadamard.py`）：`fuse_input_hadamard(layer, size) -> bool`，被包装 method 可接管输入旋转；`InputHadamardLinearMethod` 仍旋转权重，并在被包装 method fusable 且返回 True 时跳过输入旋转（`layer.input_hadamard_fused` 分支，server 以 `... rotates its input by the 32-wide Hadamard matrix itself.` 每 method 记一次日志）。移除了原 TODO。
- **`AiterMxfp4LinearKernel.fuse_input_hadamard`**：仅在 ASM GEMM 路径且 size==32 时接管（`self.is_hadamard_fused = self.use_asm_gemm and size == 32`），向 HIP quant 传 `hadamard_rotate=True`；Triton 路径保留 PyTorch 旋转（非 ASM 路径 assert 拒绝）；`hadamard_rotate` 与预量化 `x_scales` 互斥（assert）。
- **继承链**：Online MXFP4（`Mxfp4OnlineLinearMethod`）、`QuarkLinearMethod`、Quark OCP MX（`QuarkOCP_MX`）继承 mixin 并转发到各自 `MxFp4LinearKernel`；`MxFp4LinearKernel` 基类默认 `fuse_input_hadamard -> False`。
- **附带修复**：`gemm_with_dynamic_quant` 的 fake impl 签名与 op 漂移（`x_scales` 参数顺序/类型），torch.compile 会按 fake 输出尺寸读越界，测试 `test_gemm_with_dynamic_quant_fake_dtype` 用 FakeTensorMode 校验 dtype/shape。
- ASM 路径仅对 AITER 已调优 a4w4 config 的 (N, K) 生效：`out_proj` 今天已调优；ROCm/aiter#7441 补上上表数字包含的 `in_proj_qkvz` (N=4608, K=8192) config。

## 关键改动

```python
# vllm/model_executor/kernels/linear/mxfp4/aiter.py
+        hadamard_rotate: bool = False,
     ) -> torch.Tensor:
+        """With `hadamard_rotate`, the HIP quant of the ASM path rotates each
+        32-element block of `x` by the orthonormal Sylvester Hadamard matrix..."""
+        assert not (hadamard_rotate and x_scales is not None), ...
+        quant_kwargs = {"hadamard_rotate": True} if hadamard_rotate else {}
...
-                    if M >= 32:
-                        x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=True)
-                    else:
-                        x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=False)
+                    x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=M >= 32, **quant_kwargs)
...
+    def fuse_input_hadamard(self, size: int) -> bool:
+        # Only the HIP activation quant of the ASM path rotates, in 32-blocks.
+        self.is_hadamard_fused = self.use_asm_gemm and size == 32
+        return self.is_hadamard_fused
# vllm/model_executor/layers/quantization/input_hadamard.py
+class InputHadamardFusable:
+    def fuse_input_hadamard(self, layer, size) -> bool: raise NotImplementedError
# InputHadamardLinearMethod.apply:
-        hadamard: torch.Tensor = layer.input_hadamard
-        x = block_hadamard_rotate(x, hadamard)
+        if not layer.input_hadamard_fused:
+            hadamard: torch.Tensor = layer.input_hadamard
+            x = block_hadamard_rotate(x, hadamard)
         return self.quant_method.apply(layer, x, bias)
```

## 性能声明

（环境：8×MI355X，Qwen3.8-2.4T-A95B Quark MXFP4，TP8，nightly `df3bd094` + 各臂 diff，AITER 0.1.25 + ROCm/aiter#6734 移植；所有臂把 GDN 投影（in_proj_qkvz、out_proj）与 shared experts 放 online MXFP4（322 层），仅 GDN 投影方案不同；serving 为 2 次重复均值（重复间差最高 1.6%），1024/1024 random，CONC 4/64/256。）

- **Serving（output tok/s）**：RTN `mxfp4` 373.0 / 1807.1 / 4028.3（c4/c64/c256）；PyTorch rotation 326.9（−12.4%）/ 1694.4（−6.2%）/ 3787.7（−6.0%）；fused（本 PR）371.3（−0.5%）/ 1796.6（−0.6%）/ 4024.2（−0.1%）。Mean TPOT：RTN 10.45/34.09/60.91 ms；PyTorch 11.94/36.23/64.86；fused 10.50/34.21/61.09。
- **每层 quant+GEMM（µs，单 GPU，CUDA graph）**：`in_proj_qkvz` plain → PyTorch → fused：M=1 7.2/15.3/7.4；M=16 9.5/17.1/9.2；M=256 14.7/28.1/15.0；M=1024 29.3/65.1/29.6；M=8192 170.6/459.1/175.7。`out_proj`：5.3/10.9/5.0；5.4/12.6/5.6；8.0/17.6/8.5；16.9/32.5/17.0；83.5/155.5/88.0。即 fused 使 quant+GEMM 增加 0–5%，PyTorch 旋转增加 70–170%（Claims 口径）。
- **精度**：相同输入下 fused 与 PyTorch 旋转对未量化 GEMM 的相对误差在 M 1024–16384 每处五位一致；最高 0.07% 输出元素不同（fp32 butterfly vs bf16 matmul 后同一次 bf16 舍入）；#62750 在 Qwen3.8-2.4T 真权重上的结果（GSM8K 12 reps 96.34 → 96.13，fixed/broken +125/−158，p=0.057；GPQA Diamond 5 reps 92.3 → 93.1，+20/−12，p=0.22）可沿用。
- 正文还给出 "up to 1.6%"（重复间差异）与 "Up to 0.07%"（元素差异比例）口径。

## 验证与回归风险

单元测试（MI355X，stock nightly `be47189f` + AITER 0.1.29）：`test_input_hadamard.py`、`test_online_mxfp4.py`、`test_quantization_config_args.py`、`test_mx_linear_batch_invariant.py` 共 91 passed、4 skipped（含/不含 `VLLM_ROCM_USE_AITER=1`）。新增：wrapper 跳过旋转且不双重旋转的用例（`layer.applied_input` 原样）；online MXFP4 按 emulation/aiter-triton/aiter-asm 参数化并断言 `input_hadamard_fused` 与 fused 预期一致（asm 用 (1536, 3072) 触发 ASM）；fake dtype 一致性测试。`test_input_hadamard.py` 在 ASM 停止融合、Triton/emulation 路径开始融合或 wrapper 双重旋转时失败；`test_online_mxfp4.py` 在 fake `gemm_with_dynamic_quant` 再次漂移时失败。回归风险：serving 用 dummy weights，精度数字取自 #62750 未融合路径，融合等价性依赖逐元素对比；在 Quantization jobs 恢复前（#66900、#66672）无 CI job 跑这些测试，届时 MI355 job 覆盖融合路径（AITER 用例自行开启 AITER，因 CI 不设 `VLLM_ROCM_USE_AITER`）；融合仅覆盖 ASM 路径与已调优 (N, K)，其余形状与 Triton 路径保持 PyTorch 旋转。

## 关联

- 依赖上游 ROCm/aiter#6734（HIP quant 的 fused rotation）；ROCm/aiter#7441（in_proj_qkvz 的 a4w4 tuned config）。
- #62750（引入 PyTorch 旋转与精度结论）；#65398（online 方案的 MoE 变体，未融合旋转）；#66900、#66672（Quantization CI jobs 重启）。
- pack 未提供其他关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
