# PR #47910: [Bugfix] Patch Hopper MXFP4 OOB scales reads leading to NaN

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47910](https://github.com/vllm-project/vllm/pull/47910) |
| 作者 | yzong-rh |
| 合并日期 | 2026-07-07 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
Hopper MXFP4 swizzle 后 scale 按 (8,128) tile 读取越界产生 NaN;swizzle 前 pad K 到 4 的倍数修复,镜像上游 Triton 修复。

## 问题现象
Hopper 上 MXFP4 scales 越界(OOB)读取导致 NaN,#47303 给出 E2E 复现。GPT-OSS 20B GSM8K 评测:加内存投毒时修复前 accuracy=0.0%(0/100)×5 轮(avg_output_len=256.0),修复后 100.0%(100/100)×5 轮(avg_output_len=29.0);另一评测 accuracy 0.016-0.021 → 0.323-0.333。单测结果:33 passed, 117 skipped, 16 warnings。

## 根因
正文披露:(E, M, K) = (..., 5760, 90) 经 swizzle 变为 (E, M / 32, K * 32) → (..., 184, 2880),按 (8, 128) tile 读取,2880:2944 这一段越界。Triton 的 `convert_layout` 在 Hopper 上已补零保证 K = 90 可被 2 整除,但还需 K 为 4 的倍数使 K * 32 能被 128 整除。修复镜像上游 triton-lang/triton#10800,升级 Triton 3.8.0+ 内核后可移除。

## 修复方案
`vllm/model_executor/layers/quantization/utils/mxfp4_utils.py` `_swizzle_mxfp4` 的 Hopper 分支:swizzle 前 pad scale,使 K(num scale groups)为 0 mod 4:

```diff
+            # Patches #47303: pad K (num scale groups) to 0 mod 4
+            # TODO: Remove once we upgrade to Triton 3.8.0+ kernels
+            if scale.numel() > 0:
+                K = scale.shape[-1]
+                pad_k = -K % 4
+                scale = torch.nn.functional.pad(scale, (0, pad_k))
```

## 影响与回归风险
- 影响窗口:正文未说明引入版本;注明升级到 Triton 3.8.0+ 内核后可清理此补丁。
- 回归面:`_swizzle_mxfp4` 的 Hopper(设备 capability 90)分支:scale 张量尾部被 pad 至多 3 个元素,改变传给内核的 scale 形状;Blackwell 等其他平台分支不受影响。

## 关联
- #47303(被 patch 的 issue)
- triton-lang/triton#10800(镜像的上游修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
