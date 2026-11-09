# PR #25733: [Bug] Fix V4-Pro NaN on Blackwell by converting fp8_einsum input scale to ue8m0

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#25733](https://github.com/sgl-project/sglang/pull/25733) |
| 作者 | yhyang201 |
| 合并日期 | 2026-05-19 |
| labels | deepseek, run-ci |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
DeepSeek-V4-Pro 在 Blackwell 上解码输出乱码,因 deep_gemm ue8m0 打包内核不掩尾数位,修复为对激活 scale 预先 `ceil_to_ue8m0`。

## 问题现象
DeepSeek-V4-Pro 在 Blackwell GPU(B300/B200)上自回归解码时产生乱码文本(garbage text);具体表现为 `fp8_einsum` 在 batch=1(单 token 解码)时产生 NaN。

## 根因
PR 正文披露:`deep_gemm` 的 `transpose_and_pack_fp32_into_ue8m0` CUDA 内核存在打包 bug——从 fp32 提取指数时不清除(掩码)尾数位,导致非 2 的幂的 scale 值被打包损坏;损坏的 scale 使 `fp8_einsum` 在 batch=1 单 token 解码时输出 NaN。作者另附详细分析:[gist](https://gist.github.com/yhyang201/2fce5750e44970af419f9669e10e994c)。

## 修复方案
改动位于 `python/sglang/srt/models/deepseek_v4.py` 的 forward:在把激活 scale 传给 `deep_gemm.fp8_einsum` 之前,先调用 `deep_gemm.ceil_to_ue8m0(o_s)`,保证打包前尾数位为零,规避上游打包 bug。scale 张量很小(例如形状 `(2, 32)`),开销可忽略。

关键 diff 摘录:

```diff
                 o.reshape(T * G, D).contiguous(),
                 group_size=128,
             )
+            o_s = deep_gemm.ceil_to_ue8m0(o_s)
             output = torch.empty(T, G, R, device=o.device, dtype=torch.bfloat16)
             deep_gemm.fp8_einsum(
                 "bhr,hdr->bhd",
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围,未写;受影响平台为 Blackwell(B300/B200)上运行 DeepSeek-V4-Pro 且走 `fp8_einsum` 路径的场景。
- 回归面:仅 `deepseek_v4.py` 中该一处 einsum 调用点的 scale 预处理。所有 scale 值被向上取整到 2 的幂(ue8m0 可精确表示的值),数值上是对量化 scale 的保守放大(由 diff 推断);不改变其他调用 `fp8_einsum` 或其他模型路径。未修复时 batch=1 解码直接 NaN,该场景下数值影响为修复性。

## 关联
- [详细分析 gist](https://gist.github.com/yhyang201/2fce5750e44970af419f9669e10e994c)(正文附)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
