# PR #29308: Fix CI broken by #28450

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29308](https://github.com/sgl-project/sglang/pull/29308) |
| 作者 | b8zhong |
| 合并日期 | 2026-06-25 |
| labels | (none) |
| 热度 | 👍 2 · 💬 2 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#28450 改变了 _fill_padded_rows 抛出的异常类型,单测仍断言 AssertionError 导致 CI 100% 失败,同步更新断言。

## 问题现象
#28450 合入后 CI 持续失败。正文指出该失败在原 PR 自身的 CI 中就已被捕获(https://github.com/sgl-project/sglang/actions/runs/28067472170/job/83096113232),并且「not a flake (it fails 100% of the time)」,同时告诫作者后续应确保至少 stage-b CUDA 能通过。受影响的是 `test/registered/moe/test_topk_padded_region.py` 中的 `test_invalid_pad_count_tensor_raises`;具体报错正文未贴出。

## 根因
正文未给出逐条根因分析,仅认定 CI 失败由 #28450 引入且为必现(100% 失败)而非 flake。

从 diff 看,修复把 `test_invalid_pad_count_tensor_raises` 的三处断言期望从 `AssertionError` 分别改为 `TypeError`、`ValueError`、`TypeError`,对应传入 python int、int32 张量、float32 标量张量的非法 pad 参数三种情形。可推断机制是:#28450 重构了 `_fill_padded_rows` 的参数校验,把原来基于 assert 的检查改为抛出 `TypeError`/`ValueError`,但对应单测的异常期望没有同步更新,导致该测试必现失败(由 diff 推断)。

## 修复方案
改动 `test/registered/moe/test_topk_padded_region.py` 的 `test_invalid_pad_count_tensor_raises`,把三处 `self.assertRaises(AssertionError)` 更新为新实现的异常类型:

```python
with self.assertRaises(TypeError):
    _fill_padded_rows(x, 4, 0.0)  # python int, not a tensor
with self.assertRaises(ValueError):
    _fill_padded_rows(
        x,
        torch.tensor([1, 2], device=self.DEVICE, dtype=torch.int32),
        0.0,
    )
with self.assertRaises(TypeError):
    _fill_padded_rows(
        x,
        torch.tensor(4.0, device=self.DEVICE, dtype=torch.float32),
        ...
```

仅改测试断言,不触碰被测实现。

## 影响与回归风险
- 影响窗口:正文未说明(自 #28450 合入后)。
- 回归面:仅影响 MoE topk padded region 的无效参数异常类型单测;生产代码零改动。测试现在与 #28450 的异常语义绑定,若后续再次调整 `_fill_padded_rows` 的校验异常类型需同步此测试。

## 关联
- [#28450](https://github.com/sgl-project/sglang/pull/28450)(引入 CI 失败的 PR)
- [原 PR 中已捕获失败的 CI job](https://github.com/sgl-project/sglang/actions/runs/28067472170/job/83096113232)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
