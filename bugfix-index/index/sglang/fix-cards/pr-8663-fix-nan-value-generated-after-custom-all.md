# PR #8663: Fix nan value generated after custom all reduce

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#8663](https://github.com/sgl-project/sglang/pull/8663) |
| 作者 | kkHuang-amd |
| 合并日期 | 2025-08-15 |
| labels | high priority |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
custom allreduce warm-up 用 `empty_like` 分配输出张量,未初始化内存产生 NaN;改 `zeros_like` 并把 CAR 测试拆到独立 AMD CI job。

## 问题现象
正文未描述具体现象与触发日志,问题即标题所示:custom allreduce 之后产生 NaN 值。正文引用前置 PR #8532 并说明:作者复核了 #8532 的改动,它只发生在 graph capture 阶段,不属于性能运行的关键路径。修复后验证:gsm8k(`--num-questions 2000 --parallel 2000`)Accuracy 0.948、Invalid 0.000、Latency 52.566 s。

## 根因
正文未系统披露根因分析。正文 Modifications 表明:需要把 `empty_like`/`empty` 改为 `zeros_like`/`zeros`,以确保张量值被初始化为零——即 NaN 来自未按零初始化的张量;该行为与正文所引用的 #8532 改动相关,且只发生在 graph capture 阶段。

## 修复方案
- `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`:`custom_all_reduce()` 的 warm-up 分支(模拟 out-of-place 分配模式)将 `torch.empty_like(input)` 改为 `torch.zeros_like(input)`。
- `.github/workflows/pr-test-amd.yml`:新增 `unit-test-backend-8-gpu-CAR-amd` job(linux-mi300-gpu-8 runner),把 CustomAllReduce 测试移到 AMD CI 的独立 job。
- 正文 Modifications 还提到 empty→zeros 的改动;pack 的 diff 中未见对应代码片段,仅含上述两处改动。

```diff
-                return torch.empty_like(input)
+                return torch.zeros_like(input)
```

## 影响与回归风险
- 回归面:仅 `custom_all_reduce` warm-up 返回值的初始化方式变化;该路径只在 graph capture 阶段使用,按正文观点不影响性能运行关键路径;零初始化会带来该分支的少量额外写开销(由 diff 推断);CI 层面新增独立 job,不改变其余测试布局。

## 关联
- [#8532](https://github.com/sgl-project/sglang/pull/8532)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
