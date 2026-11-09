# PR #10558: [ROCm] Fix fp8 quantization accuracy issue.

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10558](https://github.com/sgl-project/sglang/pull/10558) |
| 作者 | sogalin |
| 合并日期 | 2025-09-18 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
ROCm 下 Grok1 跑 FP8 量化出现精度问题,改为在 HIP 分支使用 aiter 后端的量化 kernel,GSM8K 精度恢复到 0.823。

## 问题现象
在 ROCm(HIP)环境运行 Grok1 模型时,FP8 量化出现精度问题(Motivation: "See FP8 quant accuracy issue with running Grok1 model.")。正文未给出具体错误日志或数值偏差,仅说明该问题影响 ROCm 上 FP8 量化的推理正确性。

## 根因
正文未披露根因,仅写道 "Using aiter's solution"(采用 aiter 已有的解决方案),即认为原量化 kernel 实现有精度问题,应以 aiter 后端的量化 kernel 替代。

从 diff 看,改动点位于 `apply_fp8_linear` 中 `_is_hip and weight_scale.numel() == 1`(HIP 且 per-tensor 权重 scale)的分支:原先调用 `ops.scaled_fp8_quant` 做激活量化,现改为调用 `scaled_fp8_quant`(非 ops 前缀的实现),配合 `use_per_token_if_dynamic=use_per_token_if_dynamic`。(由 diff 推断)该分支上的 `ops.scaled_fp8_quant` 实现与动态 per-token 量化路径不匹配,导致激活 scale 计算不正确进而产生精度误差;周边注释也提及该区域存在 per-tensor activation scaling 支持缺失与 torch.compile 报错(weight_scale.numel()==1 而 x_scale.numel()>1)等已知问题,最终方案是补齐 per-tensor activation scaling 支持。

## 修复方案
修改 `python/sglang/srt/layers/quantization/fp8_utils.py` 的 `apply_fp8_linear`:在 HIP 且 weight_scale 为标量的分支,把激活量化调用从 `ops.scaled_fp8_quant` 替换为 `scaled_fp8_quant`(即 aiter 后端的实现),保留 `input_2d, input_scale, use_per_token_if_dynamic` 等参数不变。

关键 diff 摘录:

```diff
                 if _is_hip and weight_scale.numel() == 1:
-                    qinput, x_scale = ops.scaled_fp8_quant(
+                    qinput, x_scale = scaled_fp8_quant(
                         input_2d,
                         input_scale,
                         use_per_token_if_dynamic=use_per_token_if_dynamic,
```

作者用 GSM8K 验证修复后精度:`python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000`,Accuracy: 0.823,Invalid: 0.001,Latency: 19.075 s,Output throughput: 10162.474 token/s。

## 影响与回归风险
- 影响窗口:正文未说明受影响的版本/提交范围,不写。
- 回归面:改动只影响 HIP(ROCm)且 weight_scale.numel()==1 的 FP8 linear 路径(典型为 Grok1 这类 per-tensor 权重 scale 的模型);CUDA 路径与非 per-tensor scale 的 HIP 路径不受影响。若 aiter 后端在部分 ROCm 环境不可用或行为差异,可能影响该分支的量化结果;周边注释提到的 torch.compile 场景(weight_scale.numel()==1 且 x_scale.numel()>1)仍属已知待解决问题。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
