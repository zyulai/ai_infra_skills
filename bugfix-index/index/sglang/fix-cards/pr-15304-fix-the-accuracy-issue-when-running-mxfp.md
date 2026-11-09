# PR #15304: Fix the accuracy issue when running mxfp4 dsv3 model and enable ep

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15304](https://github.com/sgl-project/sglang/pull/15304) |
| 作者 | kkHuang-amd |
| 合并日期 | 2025-12-17 |
| labels | amd, run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
mxfp4 dsv3 开启 EP 时精度出错;在 moe 函数调用中补传 `expert_mask=layer.expert_mask_gpu` 修复。

## 问题现象
正文未描述具体现象与触发日志,仅称「Fix the accuracy problem」,即标题所示:mxfp4 dsv3 模型开启 EP 时出现精度问题。作者给出的验证配置(照抄关键项):`SGLANG_AITER_MLA_PERSIST=0`,launch_server 参数 `--tp-size 8`、`--ep-size 8`、`--dp-size 8`、`--enable-dp-attention`、`--attention-backend aiter`、EAGLE(`--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`)、`--kv-cache-dtype fp8_e4m3`、`--disable-radix-cache`;gsm8k(`--num-questions 2000 --parallel 2000`)结果 Accuracy 0.948、Invalid 0.000、Latency 69.667 s、Output throughput 1883.314 token/s。

## 根因
正文未披露根因,正文仅有一句「Fix the accuracy problem」与修复描述。

## 修复方案
正文 Modifications:「Add expert mask in moe function call」。diff 显示两条量化路径的 `apply()` 各增加一行实参:
- `python/sglang/srt/layers/quantization/mxfp4.py` 的 `apply()`
- `python/sglang/srt/layers/quantization/quark/quark_moe.py` 的 `apply()`

```diff
             doweight_stage1=False,
+            expert_mask=layer.expert_mask_gpu,
         )
```

即调用底层 moe 函数时传入 `expert_mask=layer.expert_mask_gpu`,使 EP 场景下计算被限制在本 rank 应处理的专家上(由 diff 推断)。

## 影响与回归风险
- 回归面:mxfp4 与 quark mxfp4 两条 MoE apply 路径开始向底层 moe 函数传递 `expert_mask=layer.expert_mask_gpu`;若对应 `layer` 对象没有该属性或其语义与底层实现不符,这两条路径会受影响(由 diff 推断)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
