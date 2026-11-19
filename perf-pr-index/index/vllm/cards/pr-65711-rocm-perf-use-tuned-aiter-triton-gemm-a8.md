# PR #65711: [ROCm][Perf] Use tuned AITER Triton `gemm_a8w8` for per-tensor FP8 decode GEMMs on gfx950

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @amd-jrosas |
| 合并 | 2026-11-16 |
| 分类 | Quantization(tier: major) |
| Label | rocm, ready |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/65711 |

## 一句话结论

把 per-tensor FP8(static activation scales)线性层在 5-256 token 的 decode 批上、且 AITER 已有 tuned config 的形状,路由到 AITER 的 Triton `gemm_a8w8`;Llama 3.1 8B FP8 在 MI355X 上 decode TPOT 最多降 8.1%、吞吐最多升 8.4%,其余批大小保持原 `ROCmFP8ScaledMMLinearKernel`(hipBLASLt)路径。

## 动机

ROCm 上所有 AITER FP8 kernel 都需要 per-token scales,per-tensor checkpoint 此前 decode 只能用 hipBLASLt;而 tuned 的 Triton `gemm_a8w8` 在这些 decode 批上更快。依赖 ROCm/aiter#7347(v0.1.28.post1 发布)。FNUZ(gfx942)被排除。

## 优化点

- 新 kernel `AiterTritonPerTensorFp8ScaledMMLinearKernel` 子类化 `ROCmFP8ScaledMMLinearKernel`;`MAX_M = 256`。batch-size 检查放在 custom op 内部(`5 <= A.shape[0] <= max_m` 且无 bias 才走 Triton),批外自动回退且 torch.compile 不会按它特化。
- `can_implement` 要求 per-tensor static activation/weight scales、BF16 输出、且 `rocm_aiter_ops.is_triton_gemm_a8w8_tuned(N, K, MAX_M)`(沿 #61278 的 tuned-config 门控;探测 8 到 max_m 的每个 M_LEQ bucket)。无新环境变量。
- 加载时把 static per-tensor scale 各扩成向量一次:`aiter_x_scale_vec`(expand 到 MAX_M)与 `aiter_w_scale_vec`(expand 到 N 列),因为 `gemm_a8w8` 读每行 x / 每列 w 的 scale。
- `_triton_gemm_config_is_tuned` 增加 `M` 参数;注册新 custom op `rocm_aiter_triton_per_tensor_fp8_gemm` 及 fake impl。

## 关键改动

```python
# vllm/_aiter_ops.py
def _rocm_aiter_triton_per_tensor_fp8_gemm_impl(A, B, out_dtype, As, Bs, bias, x_scale_vec, w_scale_vec, max_m):
    # Checked inside the op so torch.compile does not specialize on M.
    if bias is None and 5 <= A.shape[0] <= max_m:
        from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8

        return gemm_a8w8(A, B.t(), x_scale_vec, w_scale_vec, dtype=out_dtype, backend="triton")

    from vllm.model_executor.kernels.linear.scaled_mm.rocm import (
        rocm_per_tensor_float_w8a8_scaled_mm_impl,
    )

    return rocm_per_tensor_float_w8a8_scaled_mm_impl(A, B, out_dtype, As, Bs, bias)
```

```python
# vllm/model_executor/kernels/linear/scaled_mm/aiter.py
class AiterTritonPerTensorFp8ScaledMMLinearKernel(ROCmFP8ScaledMMLinearKernel):
    # Largest batch sent to Triton; larger batches run on hipBLASLt.
    MAX_M = 256
```

```diff
+    @staticmethod
+    @functools.cache
+    def is_triton_gemm_a8w8_tuned(n: int, k: int, max_m: int) -> bool:
+        if not current_platform.is_rocm():
+            return False
+        # Probe every M_LEQ bucket from 8 up to max_m.
+        probes = [8]
+        while probes[-1] < max_m:
+            probes.append(probes[-1] * 2)
```

## 性能声明

- Llama 3.1 8B FP8,MI355X(gfx950),同 vLLM 安装卸下本改动对比,`VLLM_ROCM_USE_AITER=1`,每格 10 × concurrency prompts 的 3 次计时均值,并发 4 保持 wvSplitKQ 作 control:
  - 1k/1k:TPOT 3.95 → 3.63 ms(-8.1%)@c16、4.70 → 4.45(-5.3%)@c32、6.39 → 6.23(-2.5%)@c64、10.79 → 10.35(-4.0%)@c128、19.33 → 19.08(-1.3%)@c256;total tok/s 7862 → 8523(+8.4%)@c16、+5.4% @c32、+2.6% @c64、+4.2% @c128、+1.3% @c256;c4 为 +0.2% / -0.2%。
  - 8k/1k:c16 TPOT 8.52 → 8.01 ms(-6.1%)、TTFT 2745.6 → 2290.2 ms(-16.6%)、吞吐 12859 → 14075(+9.5%);c32 TPOT -4.5%、c64 -2.9%、c128 -4.1%、c256 -0.7%;c4 TTFT +1.2%(5.1 ms),两种 ISL 下 c4 的 TTFT 增幅都在请求延迟的 0.2% 以下、吞吐不变。
- TTFT 三次运行散布(11/06 更新):如 8k/1k c16 baseline 2745.6 ± 102.2 ms(3.7%)vs PR 2290.2 ± 230.4 ms(10.1%)。
- 精度:GSM8K 5-shot 每 benchmark cell 后以该 cell 的 `num_concurrent` 运行,1k/1k flexible-extract 均值 0.7664 → 0.7664(0.0 pt)、strict-match 0.6941 → 0.6908(-0.3 pt);8k/1k 均值 0.7679 → 0.7657 / 0.7003 → 0.6946;GPQA-Diamond(0-shot CoT, 198 题, 7 次均值)0.2697 → 0.2762(+0.6 pt);作者注明每分数标准误约 ±1.2 pt,±3.3 pt 内为噪声。

## 验证与回归风险

- 新测试文件 `tests/kernels/quantization/test_rocm_aiter_triton_per_tensor_fp8.py`:49 passed;覆盖 can_implement 拒绝场景(per-token/per-channel/dynamic、fp32 输出、untuned shape 1232)、默认 kernel 选择、Llama 8B 四形状 × 10 个 batch size 与 `torch._scaled_mm` 对比(atol/rtol 2e-2)、custom op 的 opcheck;`VLLM_GPU_SYNC_CHECK=error` 无 sync。
- 回归风险:仅当 AITER 有 tuned config 的形状才切换,untuned shape 明确回落 `ROCmFP8ScaledMMLinearKernel`;Triton 路径只在无 bias 时启用(带 bias 回 hipBLASLt impl);gfx942(FNUZ)不受影响。

## 关联

- 依赖 ROCm/aiter#7347(v0.1.28.post1);tuned-config 门控沿用 #61278;正文明示未发现其他把 per-tensor FP8 路由到 `gemm_a8w8` 的 open PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
