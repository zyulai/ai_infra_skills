# PR #37762: [AMD] Fix DeepSeek-R1-MXFP4 accuracy with AITER FP8

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#37762](https://github.com/sgl-project/sglang/pull/37762) |
| 作者 | bingxche |
| 合并日期 | 2026-09-21 |
| labels | amd |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
AITER + FP8 的 BF16 query 特例被误用于所有 head 数,DeepSeek-R1-MXFP4 选错 kernel、GSM8K 精度几乎全丢;条件补上 12-head 检查恢复 FP8 query。

## 问题现象
DeepSeek-R1-MXFP4 在 TP2、TP4、TP4-MTP 配置下(AITER + FP8 MLA)选择了错误的 kernel,GSM8K 精度几乎全部丢失(正文附回归示例 action 链接)。Kimi-K3 TP8(12 本地 MLA head)是原本受益于该 BF16 异常的场景,不应受影响。

## 根因
PR 正文披露:#34647 为 Kimi-K3 TP8(具有 12 个本地 MLA head)加入了 BF16 query 特例,但该条件同时作用于所有其他 AITER + FP8 MLA 模型——缺少 head 数限定,导致 DeepSeek-R1-MXFP4 也落入 BF16 分支、选错 kernel。本次修复只补上缺失的 12-head 检查。

## 修复方案
`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py` 的 `_fused_rope_cat_and_cache`:BF16 query 特例增加 head 数条件,非 12-head 的模型恢复使用 FP8 query:

```diff
         q_nope_out.dtype
         if attn.kv_cache_dtype == "fp8_e4m3"
         and attn.current_attention_backend == "aiter"
+        and q_nope_out.shape[-2] == 12
         else kv_cache_dtype
     )
```

## 影响与回归风险
- 影响窗口:正文未给出版本范围;问题引入自 #34647(Kimi-K3 BF16 特例)。
- 回归面:AMD ROCm 上 DeepSeek MLA forward 的 query dtype 选择。行为变化:除 12-head 外的 AITER + FP8 MLA 模型从 BF16 query 回到 FP8 query 路径;12-head 场景保持 BF16 不变。
- 正文验证(8x MI355X、ROCm 7.2.4、全量 1319 题 GSM8K):DeepSeek-R1-MXFP4 TP2 / TP4 / TP4-MTP 各 0.948 accuracy、0.000 invalid;Kimi-K3 TP8 h12 FP8 Gluon 0.951 accuracy、0.001 invalid、无 Gluon failure 或 fallback(Kimi-K3 走独立 NoPE 路径,该 run 是对原 h12 Gluon 行为的非回归检查)。

## 关联
- #34647(引入 BF16 query 特例的 PR)
- [回归示例(CI job)](https://github.com/sgl-project/sglang/actions/runs/33489586349/job/99797792543)
- [验证 run(CI job)](https://github.com/sgl-project/sglang/actions/runs/35080647993/job/104743576749)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
