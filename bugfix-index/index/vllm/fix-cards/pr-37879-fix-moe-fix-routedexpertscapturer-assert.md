# PR #37879: fix(moe): fix RoutedExpertsCapturer assertion failure with DP>1 and MK path

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37879](https://github.com/vllm-project/vllm/pull/37879) |
| 作者 | Young-Leo |
| 合并日期 | 2026-04-12 |
| labels | ready |
| 热度 | 👍 2 · 💬 7 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
DP>1 且量化走 Modular Kernel 路径时,`RoutedExpertsCapturer.capture()` 固有断言把本地 token 当全量而失败,CUDA graph capture 崩溃;按 topk 形状动态区分两种 DP 派发路径修复。

## 问题现象
`--data-parallel-size > 1` 且量化方法走 Modular Kernel(MK)路径(`supports_internal_mk=True`)时,`RoutedExpertsCapturer.capture()` 抛 `AssertionError`,发生在 CUDA graph capture 阶段,导致启动失败(Fixes #37857)。

## 根因
正文披露:`DefaultMoERunner.forward()` 有两条 DP 派发路径——1) Naive dispatch(`supports_internal_mk=False`):所有 DP rank 的 token 在路由前拼接,capturer 期望 `topk_ids.shape[0]` 等于全部 DP rank 的 token 总数;2) Modular-kernel 路径(`supports_internal_mk=True`):DP combine 发生在 `quant_method.apply` 内部,因此触发 capture 的 `select_experts()` 只看到**本 DP rank 的本地 token**。原代码 `assert cumsum[-1] == topk_ids.shape[0]` 按 naive 语义写死,在 MK 路径下必失败。

## 修复方案
`vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` `capture()`:multi-dp 分支不再写死断言,改为动态比较 `topk_ids.shape[0]` 与 DP 布局:

```python
if n == total:
    # Naive dispatch: 用 cumsum 切出本 rank 段
    cumsum = torch.cumsum(num_tokens_dp, dim=0)
    end_loc = int(cumsum[self.dp_rank].item())
    start_loc = end_loc - token_num_per_dp
elif n == token_num_per_dp:
    # Modular-kernel path: topk 已是本 rank 本地 token
    start_loc = 0
    end_loc = token_num_per_dp
else:
    raise AssertionError(
        "RoutedExpertsCapturer: unexpected topk_ids batch dim ...")
```

即:naive 路径(n == total)按 cumsum 切片本 rank 部分;MK 路径(n == token_num_per_dp)直接捕获本地 token;形状与两种布局都不匹配时抛出带详细信息的 `AssertionError`。正文 Changes 描述为加 `logger.warning` 兜底以避免硬崩溃、跳过 capture;最终 diff(及新增测试 `test_routed_experts_capturer_dp_unexpected_batch_raises`)实现为 fail-fast 断言。新增 4 个单测覆盖无 DP、naive 拼接、MK 本地、异常形状四种情形。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 EPLB 相关的 routed experts 捕获路径;DP>1 + MK 的 capture 从崩溃变为正常记录本地 token;DP>1 + naive 路径行为等价(仍按 cumsum 切片);单 DP 路径不变;形状异常从笼统断言变为明确报错。

## 关联
- Fixes https://github.com/vllm-project/vllm/issues/37857

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
