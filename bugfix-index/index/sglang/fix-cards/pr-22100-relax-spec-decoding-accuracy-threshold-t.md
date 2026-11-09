# PR #22100: Relax spec decoding accuracy threshold to fix flaky test

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22100](https://github.com/sgl-project/sglang/pull/22100) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-04-04 |
| labels | speculative-decoding, run-ci |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
standalone spec decoding 的 GSM8K 测试用严格 `> 0.7` 断言,分数恰在边界即 flaky 失败;阈值放宽为 0.69 并改用 `>=`。

## 问题现象
`test_standalone_speculative_decoding.py` 的 `test_gsm8k` 在 CI 中间歇性失败,两次观测为:
- [Run 23974687870](https://github.com/sgl-project/sglang/actions/runs/23974687870/job/69930222693):`AssertionError: 0.7 not greater than 0.7`
- [Run 23973150935](https://github.com/sgl-project/sglang/actions/runs/23973150935/job/69925954502):`AssertionError: 0.69 not greater than 0.7`

正文统计多次 standalone spec decoding GSM8K 得分:0.635, 0.665, 0.690, 0.690, 0.710, 0.719, 0.720, 0.720, 0.730, 0.730, 0.740, 0.805, 0.810, 0.864;分数下限约 0.69,多数运行落在 0.69–0.74 区间,与 0.7 的严格阈值重叠,故边界分数会随机触发失败。

## 根因
正文披露:测试使用 `assertGreater(score, 0.7)`(严格大于),当 GSM8K 分数恰好落在边界(0.7 或略低)时断言失败,造成 flaky test;这是测试阈值问题而非推理代码缺陷。正文同时指出:方差大(0.635–0.864)与偶发低分暗示 standalone spec decoding 可能存在更深的精度问题,已在另一个 issue 单独跟踪(正文未给出链接)。

## 修复方案
`test/registered/spec/test_standalone_speculative_decoding.py` 中两个基类(`TestStandaloneSpeculativeDecodingBase` 与 `TestStandaloneV2SpeculativeDecodingBase`)同步修改:
```diff
-    accuracy_threshold = 0.7  # derived tests need to override this
+    accuracy_threshold = 0.69  # derived tests need to override this
...
-        self.assertGreater(metrics[metric_key], self.accuracy_threshold)
+        self.assertGreaterEqual(metrics[metric_key], self.accuracy_threshold)
```
即阈值从 0.7 放宽到 0.69,断言从严格大于改为大于等于,消除边界抖动。

## 影响与回归风险
- 影响窗口:正文未说明(该测试自引入起即使用 0.7 严格断言)。
- 回归面:纯测试改动,不影响任何运行时代码;只放宽 `test_standalone_speculative_decoding.py` 两个基类的精度门槛,继承这些基类并自行 override `accuracy_threshold` 的派生测试不受影响;理论上可能放过 0.69–0.70 之间的真实精度回退,正文以注释保留派生类 override 入口作为补偿。

## 关联
- [Run 23974687870](https://github.com/sgl-project/sglang/actions/runs/23974687870/job/69930222693)
- [Run 23973150935](https://github.com/sgl-project/sglang/actions/runs/23973150935/job/69925954502)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
