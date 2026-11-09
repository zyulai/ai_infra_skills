# PR #39804: [AMD] Fix DeepSeek-V4 accuracy by not passing num_token_non_padded to MoE topk

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#39804](https://github.com/sgl-project/sglang/pull/39804) |
| 作者 | At1a8 |
| 合并日期 | 2026-09-23 |
| labels | amd, deepseek, run-ci |
| 热度 | 👍 0 · 💬 7 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
ROCm 上向 DeepSeek-V4 MoE topk 传 `num_token_non_padded` 产生垃圾输出,改为 HIP 平台传 None。

## 问题现象
PR #39574 开始向 `DeepseekV2MoE` 的 `self.topk()` 传递 `num_token_non_padded` 后,在 ROCm 上 DeepSeek-V4-Pro TP8DP8 TBO 配置出现垃圾输出和严重的 GSM8K 精度回归。按正文给出的 #39574 CI(GSM8K 1319):
- TP+DP,开 TBO:main 0.481(修复后 0.954)
- TP+DP+MTP,开 TBO:main 0.101(修复后 0.949)
- TP+DP+EP,不开 TBO:main 0.100(修复后 0.949)
- TP+DP+EP,开 TBO:main 0.096(修复后 0.945)
- TP+DP+EP+MTP,不开/开 TBO:main 0.119 / 0.115(修复后 0.950 / 0.946)

正文未给出具体报错日志。

## 根因
正文未做更深层代码级根因分析,仅指出回归由 #39574 传入 `num_token_non_padded` 这一改动引起,在 ROCm 平台上导致 MoE 路由产生垃圾输出(即该参数在 HIP 路径上不可用,具体机制未披露)。

## 修复方案
改动位于 `python/sglang/srt/models/deepseek_v2.py` 的 `DeepseekV2MoE.forward`:
- HIP(`_is_hip`)平台:直接置 `num_token_non_padded = None`,不再把该值传入 moe routing;
- 非 HIP 路径:保持原逻辑不变(`forward_batch.moe_num_token_non_padded()`)。

关键 diff 摘录:

```diff
-        num_token_non_padded = (
-            forward_batch.moe_num_token_non_padded()
-            if forward_batch is not None
-            else None
-        )
+        if _is_hip:
+            num_token_non_padded = None
+        else:
+            num_token_non_padded = (
+                forward_batch.moe_num_token_non_padded()
+                if forward_batch is not None
+                else None
+            )
```

## 影响与回归风险
- 影响窗口:由正文可知,回归自 #39574 合并起、至本 PR 修复前;影响 ROCm 上 DeepSeek-V2/V4 系模型(经 `DeepseekV2MoE`)。
- 回归面:仅 HIP 平台的 DeepseekV2MoE 路由入口,恢复到不传 `num_token_non_padded` 的行为;padded token 掩码逻辑对 HIP 不再生效(由 diff 推断)。非 HIP(CUDA 等)路径完全未动。正文附 GSM8K 全量对比表:修复后各并行组合(TP、TP+DP、TP+EP、TP+DP+EP,含/不含 MTP 与 TBO)精度均回到 0.945–0.954。

## 关联
- [PR #39574](https://github.com/sgl-project/sglang/pull/39574)(引入回归的改动)
- CI: [run 35024279469](https://github.com/sgl-project/sglang/actions/runs/35024279469/job/104568062634)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
