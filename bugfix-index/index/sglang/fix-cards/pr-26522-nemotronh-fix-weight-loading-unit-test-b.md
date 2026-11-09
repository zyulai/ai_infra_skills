# PR #26522: [NemotronH] Fix weight-loading unit test broken by Puzzle support

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#26522](https://github.com/sgl-project/sglang/pull/26522) |
| 作者 | Fridge003 |
| 合并日期 | 2026-05-28 |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#24429 改读 max_n_routed_experts 后,NemotronH 权重加载单测的假 config 缺该属性而报 AttributeError;补上即可。

## 问题现象
`test/registered/unit/models/test_nemotron_h_weight_loading.py` 在 `main` 上失败(正文附失败 run 链接),报错:
```
File "python/sglang/srt/models/nemotron_h.py", line 910, in load_weights
    num_experts=self.config.max_n_routed_experts,
AttributeError: 'types.SimpleNamespace' object has no attribute 'max_n_routed_experts'. Did you mean: 'n_routed_experts'?
```

## 根因
正文披露:PR #24429(Support `NemotronHPuzzleForCausalLM`)把 `NemotronHForCausalLM.load_weights` 改为读取 `self.config.max_n_routed_experts`(而非 `n_routed_experts`)——这是正确改动,因为 Puzzle 变体块结构异构,`max_n_routed_experts` 是返回各 MoE 块最大专家数的属性(对基础 `NemotronHConfig` 它就等于 `n_routed_experts`)。但 #24434 新增的回归测试用只定义了 `n_routed_experts` 的假 `SimpleNamespace` config,导致测试抛 `AttributeError`。缺陷纯在测试 fixture,生产代码无 bug。

## 修复方案
只改测试文件 `test/registered/unit/models/test_nemotron_h_weight_loading.py` 的 `_make_minimal_model`,给假 config 补上属性:
```diff
-        model.config = SimpleNamespace(n_routed_experts=2)
+        model.config = SimpleNamespace(n_routed_experts=2, max_n_routed_experts=2)
```
使其与真实 `NemotronHConfig` 行为一致(该属性等于 `n_routed_experts`),生产代码零改动。

## 影响与回归风险
- 影响窗口:正文未说明;按引入关系,#24429/#24434 合并之后、本 PR 之前 main 上的单测受影响。
- 回归面:仅测试 fixture;`test_nemotron_h_weight_loading.py` 恢复可跑;生产路径 `nemotron_h.py` 的 `load_weights` 不变,无行为影响。

## 关联
- https://github.com/sgl-project/sglang/pull/24429 (改读 max_n_routed_experts 的 PR)
- https://github.com/sgl-project/sglang/pull/24434 (引入该测试 fixture 的 PR)
- 失败 run:https://github.com/sgl-project/sglang/actions/runs/26541311227/job/78184088573 (正文链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
