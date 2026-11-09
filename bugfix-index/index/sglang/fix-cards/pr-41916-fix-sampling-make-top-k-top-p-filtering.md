# PR #41916: [Fix] sampling: Make top-k/top-p filtering consistent across sampling paths

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#41916](https://github.com/sgl-project/sglang/pull/41916) |
| 作者 | alvion427 |
| 合并日期 | 2026-11-03 |
| labels | quant, speculative-decoding, bypass-fail-fast |
| 热度 | 👍 1 · 💬 30 |
| 分类 | Sampler / Detokenization |
| bug 类型 | general |

## 一句话结论
部分采样路径先 top-k 重归一再 top-p,截断点随 top-k 后剩余概率质量漂移、与其他路径不一致;统一为对原始分布先 top-p 后 top-k,并拒绝投机解码下的 min_p。

## 问题现象
SGLang 正常路径用原始概率分布计算 top-p 截断并与 top-k 求交,但 FlashInfer min-p 路径、共享的 EAGLE 家族/standalone/n-gram verifier、以及 DFlash/DSpark/UNO 共享的 target 分布构建器,是先 top-k 重归一再 top-p,同一组参数在不同路径给出不同过滤结果;UNO 的独立 compact top-k 路径行为还可能依赖同 batch 的其他请求。回归用例:概率 [0.35, 0.25, 0.20, 0.05, 0.03, ...]、top_k=4、top_p=0.82 时,token 3 的原始前缀概率质量为 0.80(<0.82)应保留,但先 top-k 后质量变成 0.80/0.85,token 3 被错误剔除。

## 根因
正文披露:这些路径把 top-p 应用在 top-k 重归一后的分布上,top-p 的效果取决于 top-k 后碰巧剩下的概率质量,不可预测且与其他采样路径不一致。

## 修复方案
- FlashInfer min-p 路径与共享 verifier 改为先 `top_p_renorm_prob` 后 `top_k_renorm_prob`(EAGLE 侧抽成 `_apply_joint_top_k_top_p` helper)。
- DFlash/DSpark/UNO 共享 target 构建器:dense 路径调换顺序;sparse 路径用 top-k 支持集的原始概率质量把 top-p 截断换算进 top-k 重归一分布;CUDA 无种子路径新增 fused `_adjust_top_p_for_top_k`(复用保留的 top-k 最大值,避免全词表 max 归约与指数中间量),种子/非 CUDA 保留 logsumexp;UNO compact 路径同样修正。
- `scheduler._handle_generate_request` 对内置投机解码拒绝 `min_p > 0`(请求 abort 并提示),因为 min_p 目前未在投机解码中生效。
- 各修正实现新增 CUDA 回归测试(`test_speculative_sampling.py`、`test_sampling_mask.py`);H200 上验证 18 passed, 6 subtests passed,各回归"修复后 Pass、修复前失败"。

关键 diff(`sampler.py` min-p 路径):
```diff
                 if sampling_info.need_min_p_sampling:
-                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
+                    # Apply top-p before top-k renormalization to preserve the original
+                    # nucleus and match the other samplers' joint support.
                     probs = top_p_renorm_prob(probs, sampling_info.top_ps)
+                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
```

## 影响与回归风险
- 影响窗口:正文未说明版本范围。
- 回归面:所有 top-k/top-p/min_p 组合的采样输出分布(FlashInfer min-p、EAGLE/standalone/n-gram verifier、DFlash/DSpark/UNO target 构建);微基准显示 DFlash sparse 开销 +17%-38%、UNO compact +32%-126%(top-p 关闭时基本持平);fused FP32 舍入可能改变截断边界成员(文档注明不承诺 bitwise 一致或与 trainer 对齐);投机解码 + min_p>0 现在被显式拒绝(NGRAM 下返回 HTTP 400 / stream abort)。
- 注:pack 正文在 20k 字符处被截断,以上内容取自截断前部分。

## 关联
- Supersedes:#36173
- fused-helper 验证汇总:https://github.com/sgl-project/sglang/pull/41916#issuecomment-6328764169

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
