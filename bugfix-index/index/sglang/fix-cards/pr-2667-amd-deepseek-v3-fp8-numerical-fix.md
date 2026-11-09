# PR #2667: AMD DeepSeek_V3 FP8 Numerical fix

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#2667](https://github.com/sgl-project/sglang/pull/2667) |
| 作者 | HaiShaw |
| 合并日期 | 2024-12-30 |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论

AMD/ROCm 下 DeepSeek-V3 FP8 权重与 bmm 未适配 e4m3fnuz 格式,补权重归一化、scale 调整与 bmm 结果缩放。

## 问题现象

DeepSeek-V3 FP8 在 AMD(MI300X / ROCm)上存在数值精度问题,需专项修复。正文未描述修复前的具体错误表现或错误率,仅给出修复后的验证结果:gsm8k 200 题 8-shot 基准 Accuracy 0.920、Invalid 0.000;MI300X dry 性能(全部未调优)prefill 13291.01 token/s、decode 中位延迟 0.10719 s。

## 根因

正文披露:deepseek_v2.py 中多处需要对权重做额外归一化(转到 e4m3fnuz)、调整缩放系数,并对最终 bmm 结果做缩放——即 AMD 平台的 fp8 数值格式与既有 CUDA 路径处理方式不匹配,原实现缺少这些适配步骤。结合代码内注释(由 diff 推断):ROCm/HIP 侧 fp8 张量为 `torch.float8_e4m3fnuz` 格式,而原代码只处理 `torch.float8_e4m3fn`,`bmm_fp8` 不支持 fnuz 格式;且 fnuz 表示下的 scale 数值需乘 2.0 才能对应。

## 修复方案

改动集中于 `python/sglang/srt/models/deepseek_v2.py`:

1. `forward_absorb`:`w_kc` / `w_vc` 为 `float8_e4m3fnuz` 时不再走 `input_to_float8` + `bmm_fp8`,改为转 bf16 后 `torch.bmm` 并乘 `w_scale`,原 `float8_e4m3fn` 分支保留为 `elif`;
2. `load_weights`:权重 dtype 判断扩展到 `float8_e4m3fnuz`;HIP 下先经 `normalize_e4m3fn_to_e4m3fnuz` 归一化权重与 `weight_scale_inv`,再做 `block_quant_to_tensor_quant`;
3. `w_scale` 回退路径(`weight_block_size` 不存在时)在 HIP 下乘 2.0。

关键 diff 摘录:

```diff
+        if self.w_kc.dtype == torch.float8_e4m3fnuz:
+            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
+            q_nope_out = torch.bmm(
+                q_nope.to(torch.bfloat16).transpose(0, 1),
+                self.w_kc.to(torch.bfloat16) * self.w_scale,
+            )
+        elif self.w_kc.dtype == torch.float8_e4m3fn:
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:改动 DeepSeek-V2/V3 在 HIP 平台的 fp8 权重加载路径与 MLA `forward_absorb` 的 bmm 计算路径;非 HIP(CUDA)路径通过 `is_hip_` 分支保持原逻辑。代码内原有注释自述:`bmm_fp8` 只支持 per-tensor scale,故对 `kv_b_proj` 做 requantize,这可能影响 fp8 模型精度(原文既有说明,非本次引入)。

## 关联

- [PR #2637](https://github.com/sgl-project/sglang/pull/2637)(正文注明本 PR 配套其核心改动)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
