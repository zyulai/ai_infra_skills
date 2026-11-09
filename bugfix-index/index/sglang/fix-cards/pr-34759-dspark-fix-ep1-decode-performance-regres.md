# PR #34759: [DSpark] Fix EP1 decode performance regression

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34759](https://github.com/sgl-project/sglang/pull/34759) |
| 作者 | ormandj |
| 合并日期 | 2026-08-14 |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论

#33465 让 DSpark 每个投机解码 step 都构造一个未被使用的 CUDA int32 标量,EP1 decode 回退约 7%~9%;改为 padding 关闭时返回 None。

## 问题现象

DSpark 投机解码在 MoE EP size 1(EP1)场景下 decode 性能回退。作者在 2× RTX PRO 6000 Blackwell (SM120)、TP2、DeepSeek-V4-Flash-0731 + DSpark、固定 16,384-token 输入、2,048 输出 token、temperature 0、128 scheduler steps per capture 的负载下测得每 scheduler step 平均耗时(两 rank 平均,毫秒,越低越好):

| C | Before #33465 | After #33465 | Fixed | 降幅 |
|---:|---:|---:|---:|---:|
| 1 | 14.744 ms | 15.681 ms | 14.605 ms | 6.86% |
| 8 | 32.979 ms | 35.399 ms | 32.119 ms | 9.27% |

修复后与 after-#33465 对照组使用相同源码栈、launch 配置与输入,仅叠加本补丁;每次 capture 均保留 127 组完整 target+draft graph replay,图内核数一致。

## 根因

PR #33465 使 `DraftBlockProposer` 在每个投机解码 step 构造一个 CUDA `int32` 标量。而 `num_token_non_padded` 只在 `enable_num_token_non_padded()` 为 true 时才被消费;对 MoE EP size 1 该谓词为 false,设备端分配完全未被使用,纯属每步开销(正文披露)。

## 修复方案

在 `python/sglang/srt/speculative/dspark_components/dspark_draft.py` 新增辅助函数,padding 关闭时直接返回 `None`,开启时保留原有 int32 设备标量;`_run_forward` 调用点改用该函数:

```diff
+def _make_num_token_non_padded(
+    num_tokens: int, device: str | torch.device
+) -> Optional[torch.Tensor]:
+    if not enable_num_token_non_padded():
+        return None
+    return torch.tensor(num_tokens, dtype=torch.int32).to(device, non_blocking=True)
```

同时为两条路径(禁用返回 None、启用保留标量值与 `torch.int32` dtype)补充单元测试。该改动不改变模型算术;作者在 Linux image build 中跑过 `test/registered/spec/dspark/test_dspark_dp_tier.py`,改动文件通过仓库固定版本的 Black/Ruff/isort 检查。

## 影响与回归风险

- 影响窗口:自 #33465 合并起、本 PR 之前的版本受影响(正文指明回归由 #33465 引入)。
- 回归面:改动仅涉及 DSpark draft forward 路径的 `num_token_non_padded` 传参。padding 关闭(如 EP1)路径现在传 `None`,若下游存在未按谓词消费该字段的代码需注意;padding 开启路径行为不变(仍为 int32 设备标量)。

## 关联

- [#33465](https://github.com/sgl-project/sglang/pull/33465)(正文指明为回归引入者)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
