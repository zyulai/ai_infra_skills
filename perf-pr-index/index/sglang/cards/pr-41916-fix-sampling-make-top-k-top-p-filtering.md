# PR #41916: [Fix] sampling: Make top-k/top-p filtering consistent across sampling paths

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @alvion427 |
| 合并 | 2026-11-03 |
| 分类 | Speculative Decoding |
| Label | quant, speculative-decoding, bypass-fail-fast |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 含 8 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/41916 |

## 一句话结论

统一各采样路径的 top-k/top-p 过滤顺序为"先对原始分布做 top-p 再做 top-k",修复投机解码验证目标分布中 nucleus 被错误截断的问题,并融合 CUDA 未 seed 路径的 top-p 重标定。

## 动机

SGLang 常规路径从原始概率分布计算 top-p 截断并与 top-k 支撑集求交,但部分路径先施加 top-k、把剩余概率重归一化、再施加 top-p——这等于对"top-k 之后碰巧剩下的概率质量"做 top-p,效果不可预测且与其他使用相同参数的路径不一致。受影响路径:FlashInfer min-p 路径、EAGLE 家族/standalone/n-gram 共享 verifier、DFlash/DSpark/UNO 共享的 target 分布构建器;UNO 独立的 compact top-k 路径也需同样修复(否则行为取决于 batch 中其他请求)。supersede #36173 并落实其最终 review 意见。

## 优化点

顺序修正:统一为"先 top-p 后 top-k"。FlashInfer min-p 路径、EAGLE verifier(提取为 `_apply_joint_top_k_top_p` 助手,含 NaN probe)、DFlash/DSpark/UNO 共享 target builder 的 dense 路径均按此重排。稀疏路径的难点是 top-p 原本在 full-vocabulary 上定义,而 compact 支撑集只有 top-k:修复把 top-p 截断翻译进重归一化后的 top-k 分布——用 top-k 支撑集的原始概率质量(`topk_mass = exp(logsumexp(topk_logits) - logsumexp(full_logits))`)把 top_p 缩放为 `top_p / topk_mass`(clamp 到 1)。CUDA 未 seed 路径进一步融合:新增 `@torch.compile(dynamic=True)` 的 `_adjust_top_p_for_top_k`,复用 top-k 保留的全局最大值(免一次全词表 max reduction)并融合移位求和(避免全尺寸 exponential 中间张量),替代 eager `logsumexp`;seeded 与非 CUDA 调用方保留 logsumexp 舍入,UNO 拒绝确定性推理。文档化融合 FP32 舍入可能改变截断边界成员(不承诺 bitwise 一致或与 trainer 对齐)。另外:内建投机解码现在拒绝 `min_p > 0`(scheduler 层对投机请求 abort 并返回 400,该参数此前并未真正应用)。

## 关键改动

```text
python/sglang/srt/speculative/dflash_utils.py: _adjust_top_p_for_top_k 融合内核 + sparse 路径按 topk_mass 调整 top-p + dense 路径顺序重排
python/sglang/srt/speculative/eagle_utils.py: _apply_joint_top_k_top_p 助手(top-p 先于 top-k)
python/sglang/srt/speculative/uno_utils.py: compact 支撑集使用 full-distribution top-p 语义
python/sglang/srt/layers/sampler.py: FlashInfer min-p 路径先 top_p_renorm 再 top_k_renorm
python/sglang/srt/managers/scheduler.py: 投机解码拒绝 min_p>0(abort + 400)
test/registered/kernels/ops/speculative/test_speculative_sampling.py: 每条修复路径的 CUDA 回归测试
```

## 性能声明

正确性回归用例(probs `[0.35,0.25,0.20,0.05,...]`,top_k=4,top_p=0.82):token 3 的原始前置质量为 0.80 < 0.82 应保留,但先 top-k 重归一化后质量变为 0.80/0.85 而被错误移除;修复后 5 条路径(FlashInfer min-p、EAGLE、dense/sparse target builder、UNO compact)全部保留 token 3,修复前全部失败。H200(PyTorch 2.14.1+cu130, FlashInfer 0.7.2, Triton 3.8.0,seed 41916,FP32 logits std=2,temp 0.8,top-k 50,top-p 0.9,draft width 6,CUDA graph replay 中位数)采样函数微基准(**此测量先于融合提交 `b94a95d`,后续有 fused-helper 验证摘要**):DFlash sparse 增量 B=1/16/64、vocab 32768 为 +19.65/+28.09/+63.17 us,vocab 128256 为 +38.40/+122.41/+315.81 us(896.05 -> 1211.86 us);UNO compact 对应为 +9.26/+18.40/+58.30 与 +21.13/+57.31/+205.58 us(163.58 -> 369.16 us)。top-p 关闭时 B=16/vocab 128256:DFlash 300.17 -> 300.31 us,UNO 57.71 -> 56.90 us。profiler:B=64/vocab 128256 下 DFlash sparse 的 eager logsumexp 为 9 内核约 350 us、约 188 MiB 临时张量(内核数 30->52);UNO compact 为 4 内核约 206 us(6->10)。融合收益:DFlash 的 top-p 内核从约 53 us 降到 5 us。端到端吞吐与延迟未测量。

## 验证与回归风险

`TestSamplingMaskCapture` 与 `test_speculative_sampling.py`:18 passed + 6 subtests(H200,Python 3.12.3,CUDA 13.0,PyTorch 2.13.0,FlashInfer 0.6.18,Triton 3.7.1,commit `50e6ca6b73cc`)。历史 HTTP 验证(commit `f0c0a64`,Qwen2.5-0.5B-Instruct):10 项全过——NGRAM 接受 `min_p=0`,`min_p=0.1` 非流式 400、流式 abort+400+[DONE](各跑两遍,后续合法请求成功),普通采样接受 `min_p=0.1`;其他投机模式与 PD 部署未覆盖。风险面:采样语义变化会改变输出分布(top-k/top-p 同开时支撑集变大);融合路径 FP32 舍入与 logsumexp 不同,可改变截断边界成员;seeded/非 CUDA 保留旧舍入,两种实现并存;CI Base 曾绿(`f0c0a64`)但最新 Base/Extra/AMD 均失败(bypass-fail-fast);成本增加 17-38%(DFlash)与 32-126%(UNO),换取跨路径一致性。

## 关联

Supersedes:PR #36173(同一问题的前次尝试)。受影响路径覆盖 EAGLE 家族、DFlash、DSpark、UNO、n-gram verifier 与 FlashInfer min-p;与 PR #33395 的 seeded 采样流(rejection sampling 确定性)在 seeded 分支上交汇。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
