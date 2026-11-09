# PR #29918: [AMD] Gate broken CK block-FP8 GEMM shapes to aiter-triton-GEMM to fix ROCm 7.0 Qwen3.5 accuracy

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29918](https://github.com/sgl-project/sglang/pull/29918) |
| 作者 | yctseng0211 |
| 合并日期 | 2026-07-03 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
gfx95+ROCm 7.0 的 CK block-FP8 GEMM 回退路径在超阈值 M 时返回 NaN,Qwen3.5 prefill 被污染致 gsm8k 0.244,受影响 shape 按 CK 安全 M 上限改走 Triton。

## 问题现象
Qwen3.5-397B-A17B-FP8 精度测试在 AMD gfx950(MI35x)+ ROCm 7.0 上失败:GSM8K 掉到约 0.24,无效输出率约 67%;相同代码/模型/参数在 ROCm 7.2 上通过(约 0.95)。触发条件:prefill(chunked 上限 16384 tokens)的 M 超过受影响 shape 的 CK 阈值,Qwen3.5 的 dense FP8 attention 投影正好命中这些 shape。

## 根因
正文披露:在 gfx95 上,bpreshuffle CK block-scale FP8 GEMM 在 ROCm < 7.2 被禁用(ROCm 7.0 hipcc 误编译,见 PR23319);dense FP8 linear 路径随之回退到非 bpreshuffle 的 `ck_gemm_a8w8_blockscale`,该内核在 ROCm 7.0 gfx95 上对特定 shape 在超过每-shape M 阈值后返回 NaN:

| Shape (n, k) | CK 正确至 | NaN 出现于 | ROCm 7.2 CK |
| --- | --- | --- | --- |
| (2560, 4096) | M = 2048 | M ≥ 4096 | 正确 |
| (4096, 1024) | M = 4096 | M ≥ 8192 | 正确 |

NaN 污染被 prefill 的序列,产生垃圾/无效输出。最小单测(相同 fp8 输入,CK vs Triton vs fp32 参考实现)证实 ROCm 7.0 上 CK 返回 NaN 而 Triton 全 M 匹配 fp32 参考;ROCm 7.2 上同一 CK 调用全 M(含 16384)正确。

## 修复方案
仅作用于受影响环境(gfx95 + ROCm < 7.2),把超过实测 CK 安全 M 的受影响 shape 回退到数值正确的 Triton FP8 block GEMM,小 M(如 decode)保留更快的 CK 路径,其余 shape 与 ROCm ≥ 7.2 / 非 gfx95 一律不动。
`fp8_utils.py` 新增保守安全上限表,并在 `aiter_w8a8_block_fp8_linear` 的 `_use_aiter_gfx95` 分支叠加判断:

```diff
+_AITER_GFX95_CK_W8A8_MAX_SAFE_M = {
+    (2560, 4096): 2048,
+    (4096, 1024): 4096,
+}
...
     elif _use_aiter_gfx95:
-        use_triton = use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k)
+        _ck_safe_m = _AITER_GFX95_CK_W8A8_MAX_SAFE_M.get((n, k))
+        use_triton = use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k) or (
+            _ck_safe_m is not None and input_2d.shape[0] > _ck_safe_m
+        )
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本范围;问题环境为 ROCm 7.0 + gfx95(ROCm 7.2 中已修复,不受影响)。
- 回归面:仅 gfx95+ROCm<7.2 上 (2560,4096)、(4096,1024) 两个 shape 的 dense w8a8-block FP8 linear;大 M 走 Triton(小 M 仍走 CK),未列出的 shape 与其他 ROCm/硬件的派发决策保持原样。验证:tp=8、dcp=2、`--attention-backend triton` 下 GSM8K(1319 题)ROCm 7.0 从 0.244/Invalid 0.666 → 0.950/约 0.01,ROCm 7.2 参考 0.945/0.009。CI:Base/Extra 红(原帖标注)。

## 关联
- [PR23319](https://github.com/sgl-project/sglang/pull/23319)(正文引用:ROCm 7.0 hipcc 误编译 bpreshuffle CK 的依据)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
