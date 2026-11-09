# PR #14848: Fix TestGLM41VPPAccuracy test flakiness

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14848](https://github.com/sgl-project/sglang/pull/14848) |
| 作者 | byjiang1996 |
| 合并日期 | 2025-12-11 |
| labels | Multi-modal, run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
TestGLM41VPPAccuracy 的 mmmu 精度在 0.3~0.5 间波动导致 CI 难通过;为 GLM4+ 的 `<|begin_of_box|>` 答案格式加专用正则抽取,并给测试 server 加 `--reasoning-parser glm45`,本地 3 次稳定 0.50。

## 问题现象
`TestGLM41VPPAccuracy` 非常 flaky:UT 的 mmmu 精度在 0.3 到 0.5 之间波动,很难通过测试(CI 硬阈值)。正文披露两个原因:
1. GLM4+ 系列模型以特定格式返回答案 `"<|begin_of_box|>some_answer<|end_of_box|>"`,现有 SGLang mmmu 答案抽取正则引擎无法识别,需要针对 GLM4+ 模型定制正则才能正确抽出答案;
2. 需要 reasoning-parser——否则 reasoning 内容会被填进常规 content,污染答案文本。
影响面是该多模态精度测试(test/srt/test_pp_single_node.py 的 test_mmmu)在 CI 中的稳定性。

## 根因
正文披露:一是 mmmu 评测的答案抽取正则不认识 GLM4+ 的 box 标记格式,抽出的「答案」实为含标记/推理内容的整段文本,评分自然偏低且不稳定;二是测试启动的 server 未启用 reasoning parser,推理内容混入常规 content 进一步污染抽取结果。

## 修复方案
1. `python/sglang/test/run_eval.py`:mmmu 评测构造 `MMMUVLMEval` 时透传 `response_answer_regex`(用 getattr 兼容缺省);
2. `python/sglang/test/simple_eval_mmmu_vlm.py`:`MMMUVLMEval.__init__` 新增 `response_answer_regex` 参数(默认 `"(.*)"`),采样得到 response 后用 `re.search` 抽取第 1 个捕获组并 strip,作为最终答案参与评分;
3. `test/srt/test_pp_single_node.py`:server 启动参数增加 `--reasoning-parser glm45`;`run_eval` 传入 `response_answer_regex="<\|begin_of_box\|>(.*)<\|end_of_box\|>"`;断言阈值从 `metrics["score"] > 0.55` 放宽为 `> 0.45`。关键 diff 摘录:

```diff
+            match = (
+                re.search(self.response_answer_regex, response_text)
+                if response_text is not None
+                else None
+            )
+            response_text = (
+                match.group(1).strip() if match is not None else response_text
+            )
```

## 影响与回归风险
- 回归面:仅测试基建(`run_eval.py`、`simple_eval_mmmu_vlm.py`、`test_pp_single_node.py`),不影响推理运行时。`MMMUVLMEval` 默认 regex 为 `"(.*)"` 时抽取等价于原行为(整段文本),已有其他调用方不受影响;test_mmmu 阈值 0.55→0.45 与新增正则、reasoning-parser 绑定,仅作用于 GLM4.1V PP 这条测试。正文报告本地连续 3 次成功、mmmu 0.50 > 0.45(CI 硬阈值)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
