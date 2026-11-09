# PR #37134: [ROCm] Fix EAGLE spec-decode verify silently sampling greedy on HIP

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#37134](https://github.com/sgl-project/sglang/pull/37134) |
| 作者 | xiaobochen-amd |
| 合并日期 | 2026-09-16 |
| labels | amd, speculative-decoding, run-ci, jit-kernel |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
HIP 上 EAGLE verify 因 `_is_hip` 与 greedy 判据同分支而无条件走 argmax、静默忽略 temperature/top_p;改为启用 rejection sampling 时走纯 Triton 链式采样器。

## 问题现象
EAGLE spec-decode 的 verify 在 ROCm 上一直提交 `argmax`,完全忽略 `temperature` 与 `top_p`,且日志中没有任何提示;`temp > 0` 时解码退化为重复循环(repetition loops)。

## 根因
正文披露:`eagle_sample` 中的一行判据

```python
if sampling_info.is_all_greedy or _is_cpu or _is_npu or _is_hip or _is_xpu:
    target_predict = torch.argmax(next_token_logits, dim=-1)
```

`_is_hip` 与 `is_all_greedy` 并列,使 HIP 无条件走 greedy 路径。之所以这样写,是因为采样分支需要 `tree_speculative_sampling_target_only`、`top_k_renorm_prob`、`top_p_renorm_prob`,而这些 op 在 `sgl_kernel` 中只有 CUDA/MUSA 版本,HIP 上无法 import。

## 修复方案
rejection sampling 激活且 batch 非 all-greedy 时,HIP 改走 `reject_sampling.py` 的纯 Triton 链式采样器,并用 Triton renorm 替代 CUDA-only renorm op(diff 中 HIP 分支从 `renorm_triton` 导入 `top_k_renorm_probs_triton`/`top_p_renorm_probs_triton` 并别名;正文说明其形状与被替换 op 一致,`top_p >= 1.0` 短路直通以保持与 CUDA 逐位一致)。greedy 判据抽成纯 helper `_verify_uses_greedy`(`is_hip and not use_rejection_sampling` 才强制 greedy),非 HIP 平台语义不变。配套:1) `speculative_hook.py` 新增 HIP 自动启用 rejection sampling 的条件(EAGLE、无 token_map、topk=1、默认 accept threshold、非 deterministic;EAGLE3/token_map 场景保持关闭避免 crash);2) draft cuda graph runner 携带真实 per-request `top_ks` 而非 -1 占位;3) `sample_draft_proposal` 对 greedy 行(top_k==1)直接提议 argmax,并依赖 verify 前按同样 top_ks renorm 保持无偏;4) `eagle_worker_v2` 把 `draft_probs` 从 graph 私有内存池 clone 出来;5) `reject_sampling.py` kernel 对非概率 q(NaN/±inf/负数)拒绝接受并回退 residual 路径。新增 gate 真值表测试与 renorm 数值等价测试。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:所有平台的 verify gate(测试固定非 HIP 与补丁前谓词逐字节一致);HIP 上 EAGLE verify 从 argmax 变为采样,行为可见变化是有意的修复;draft graph runner 的 top_ks 缓冲与 draft proposal 采样路径;reject_sampling kernel 的接受判定(对异常 q 从「无条件接受」改为「拒绝并重采样」,方向上安全)。

## 关联
正文未列出(diff 注释中提及 #26358,不属于正文链接)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
