# PR #7137: [Core] Use flashinfer sampling kernel when available

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @peng1999 |
| 合并 | 2024-08-19 |
| 分类 | Attention / Kernel(major) |
| Label | ready |
| Diff | diffstat 缺失(diff 全文已提供,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/7137 |

## 一句话结论

采样阶段改用 flashinfer 融合的 top-k+top-p 采样 kernel 替代基于排序的 torch 链路,qwen2-1.5b batch 2048 端到端输出速度 3821.83→4174.60 tok/s。

## 动机

vLLM 当时对 top-k/top-p 的实现是 `_apply_top_k_top_p`:基于 torch 排序逐级截断概率分布,多轮 sort/mask 往返,GPU 时间在 sampler 里占比可观。flashinfer 提供单 kernel 融合的 `top_k_top_p_sampling_from_probs`(拒绝采样),PR 作者称其远快于现行 sorting kernel,可整体削减采样器 GPU 时间。

## 优化点

在 `sampler.py` 里按 `VLLM_USE_FLASHINFER_SAMPLER` 环境变量与 flashinfer 可用性决定是否启用:启用后跳过 `_apply_top_k_top_p`(logits 不再预先截断),对 RANDOM 类采样改走 `_top_k_top_p_multinomial_with_flashinfer`,一次调用同时完成 top-k/top-p 过滤与多项式采样,uniform 随机数按 seq_group generator 分段填充以保持可复现。flashinfer 拒绝采样失败(`success` 不全为真)时回退到 `top_k_renorm_prob`/`top_p_renorm_prob`/`sampling_from_probs` 链路并告警。另将 `_multinomial` 中 `expand().contiguous().view()` 换成 `repeat_interleave` 以避免强制 GPU-CPU 同步。

## 关键改动

```text
vllm/model_executor/layers/sampler.py: 引入 flashinfer_top_k_top_p_sampling,新增 _top_k_top_p_multinomial_with_flashinfer 及失败回退
vllm/envs.py: 新增 VLLM_USE_FLASHINFER_SAMPLER(默认 False)
tests/samplers/test_sampler.py: 新增 test_flashinfer_fallback(回退结果一致性)
Dockerfile: flashinfer 0.1.3 → 0.1.4
.buildkite/test-pipeline.yaml: samplers 测试增加 VLLM_USE_FLASHINFER_SAMPLER=1 一轮
```

启用判断节选:

```python
if envs.VLLM_USE_FLASHINFER_SAMPLER and find_spec("flashinfer"):
    from flashinfer.sampling import (
        top_k_top_p_sampling_from_probs as flashinfer_top_k_top_p_sampling)
else:
    flashinfer_top_k_top_p_sampling = None
```

## 性能声明

端到端:input length=1024、batch size=2048、model=qwen2-1.5b。main:29.86 it/s,est. speed output 3821.83 tok/s;本 PR:32.61 it/s,est. output 4174.60 tok/s(输出速度约 +9%)。采样器单独加速对比以 GitHub 截图形式给出,正文无具体数字。PR 未注明 GPU 型号与精度等其余测试条件。

## 验证与回归风险

依赖 `tests/samplers/test_sampler.py`:新增 `test_flashinfer_fallback` 验证 flashinfer 采样与回退链路结果一致;CI 对 samplers 测试以默认与 `VLLM_USE_FLASHINFER_SAMPLER=1` 双跑。风险面:flashinfer 拒绝采样 32 轮上限(`max_top_k_round = 32`)内失败时走 renorm 回退,数值路径不同;env 默认关闭,行为变化 opt-in;`test_sampler.py` 中检查 top-k/top-p 行为的用例需 patch 掉 flashinfer 入口,否则语义改变。

## 关联

同区域为采样/后处理 kernel 的持续替换(flashinfer 系列接入);素材未列出同分类其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
