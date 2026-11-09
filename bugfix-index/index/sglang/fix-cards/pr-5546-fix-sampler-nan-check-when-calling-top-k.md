# PR #5546: Fix sampler nan check when calling top_k_top_p_sampling_from_probs

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#5546](https://github.com/sgl-project/sglang/pull/5546) |
| 作者 | yubofredwang |
| 合并日期 | 2025-04-20 |
| labels | (none) |
| 热度 | 👍 2 · 💬 2 |
| 分类 | Sampler / Detokenization |
| bug 类型 | correctness |

## 一句话结论
flashinfer 新 API 不再返回 success 张量,开启 NaN 检测时采样触发变量未赋值错误;改为传 `check_nan` 参数修复。

## 问题现象
main 分支代码在 `self.use_nan_detection` 为 true 时,调用 `top_k_top_p_sampling_from_probs` 会抛出 `variable referenced before assigned` 错误(变量在赋值前被引用)。正文未描述更多具体复现细节。

## 根因
正文披露:最新 flashinfer API 的 `top_k_top_p_sampling_from_probs` 不再返回 success 张量,而 main 分支的调用代码仍按旧约定处理——开启 NaN 检测时(`self.use_nan_detection` 为 true)走旧逻辑,导致变量在赋值前被引用而报错。同时作者指出已无需再检查 success:flashinfer 侧 PR(#912)已改进采样健壮性,现在保证一定能生成 token。

## 修复方案
- `python/sglang/srt/layers/sampler.py`:删除采样后基于 `self.use_nan_detection` 的 `logger.warning("Detected errors during sampling!")` 与 `torch.zeros_like` 兜底;改为在调用前计算 NaN 检查标志并作为参数传入:
```python
# Check Nan will throw exception, only check when crash_on_warnings is True
check_nan = self.use_nan_detection and crash_on_warnings()
batch_next_token_ids = top_k_top_p_sampling_from_probs(
    probs, sampling_info.top_ks, sampling_info.top_ps,
    filter_apply_order="joint",
    check_nan=check_nan,
)
```
- `sgl-kernel/python/sgl_kernel/sampling.py`:把 `top_p_sampling_from_probs`、`top_k_top_p_sampling_from_probs` 及其 internal 版本的返回签名从 `Tuple[torch.Tensor, torch.Tensor]` 改为 `torch.Tensor`(去掉 success 张量),并同步清理 `Tuple` 导入。
- 验证:`python3 -m unittest test_bench_one_batch.TestBenchOneBatch.test_bs1`,改动前后 Decode median throughput 分别为 172.50 与 172.60 token/s,正文称无延迟回归。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:flashinfer 采样后端下 `top_k_top_p_sampling_from_probs` 的调用路径(行为从「警告 + 置零」改为「`check_nan=True` 且 `crash_on_warnings()` 时直接抛异常」);sgl-kernel 采样接口返回签名变化,影响所有依赖该接口返回二元组的调用方。

## 关联
- [flashinfer-ai/flashinfer#912](https://github.com/flashinfer-ai/flashinfer/pull/912)(采样健壮性改进,token 生成有保证)
- flashinfer sampling.py 参考实现(正文附链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
