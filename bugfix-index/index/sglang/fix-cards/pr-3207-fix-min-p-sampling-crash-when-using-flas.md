# PR #3207: Fix min_p sampling crash when using flashinfer backend

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#3207](https://github.com/sgl-project/sglang/pull/3207) |
| 作者 | zifeitong |
| 合并日期 | 2025-02-02 |
| labels | (none) |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
flashinfer 后端 min_p 采样时解包不存在的 success 返回值导致崩溃;改为单返回值并把 NaN 检测移回适用分支。

## 问题现象
使用 flashinfer 采样后端时,开启 min_p 采样会崩溃(标题:"Fix min_p sampling crash when using flashinfer backend")。正文补充:"flashinfer backed won't return success states for min_p"——即 flashinfer 路径的 min_p 采样不返回 success 状态。触发条件为 flashinfer 采样后端 + `need_min_p_sampling`。正文未提供具体报错栈与复现脚本(详见关联 issue #3201,pack 未含其内容)。

## 根因
正文未披露根因,仅说明现象。(由 diff 推断)`python/sglang/srt/layers/sampler.py` 的 flashinfer 分支中,`min_p_sampling_from_probs` 被写成双返回值解包 `batch_next_token_ids, success = min_p_sampling_from_probs(...)`,但该函数并不返回 success 状态,解包失败导致崩溃;且其后的 NaN 检测块引用的 `success` 在 min_p 分支下也无来源。

## 修复方案
改 `python/sglang/srt/layers/sampler.py`:

```diff
-                    batch_next_token_ids, success = min_p_sampling_from_probs(
+                    batch_next_token_ids = min_p_sampling_from_probs(
                         probs, uniform_samples, sampling_info.min_ps
                     )
...
-                if self.use_nan_detectioin and not torch.all(success):
-                    logger.warning("Detected errors during sampling!")
-                    batch_next_token_ids = torch.zeros_like(batch_next_token_ids)
+                    if self.use_nan_detectioin and not torch.all(success):
+                        logger.warning("Detected errors during sampling!")
+                        batch_next_token_ids = torch.zeros_like(batch_next_token_ids)
```

即 min_p 分支改为接收单一返回值;`success` NaN 检测块整体移入 else 分支(top_k_top_p joint filter,该分支才返回 success),缩进保持逻辑不变。 Modifications 一节为空,无更多说明。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅 flashinfer 采样后端路径。min_p 分支此后不再执行 NaN 检测(success 不可得),采样异常时不再有 "Detected errors during sampling!" 告警与置零兜底;非 min_p 分支行为不变;pytorch 采样后端不受影响。

## 关联
- Fixes #3201

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
