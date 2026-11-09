# PR #40889: [Fix] [NPU] qwen3_6: Set accuracy cases to greedy decoding and tune retries

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#40889](https://github.com/sgl-project/sglang/pull/40889) |
| 作者 | pllimax |
| 合并日期 | 2026-10-21 |
| labels | npu, run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

NPU 上 4 个 Qwen3.6 精度用例带非零采样温度致结果不可复现,改为贪心解码并收紧 max_retries 上限。

## 问题现象

四个 Qwen3.6 精度用例(均为 `nightly-acc-2-npu-a3`)使用非零采样温度(0.2 / 0.6 / 1.0),精度结果依赖采样、不可复现。验证时首个 run 中两个 gpqa 用例两次尝试均低于阈值(p0: 0.843 再 0.849;p1: 0.849 再 0.833),仅在全 job 重跑后通过(0.859 / 0.879);贪心解码下这两个用例观测散布为 0.833 - 0.879。另:DP attention 基础功能用例此前因 DTS2026091464350 被临时禁用,问题已解决但用例尚未恢复。

## 根因

正文披露:非零采样温度使精度用例的结果依赖采样过程、不可复现,需切换为贪心解码。同时正文说明了 `max_retries` 的语义——它是一次用例执行的 evalscope 评估次数(而非额外重试次数),一次评估达标即通过、最后一次仍失败才算失败;不设置时对 `DATASET_FLUCTUATION` 中的数据集(`gpqa_diamond`、`aime25`、`aime26`、`gsm8k`)默认 3,其余默认 1,不加约束会让最坏情况评估次数与用例时长膨胀。

## 修复方案

纯测试配置改动,三部分:

1. 四个 Qwen3.6 精度用例的 `temperature` 从 1.0 / 1.0 / 0.6 / 0.2 统一改为 0(贪心解码),涉及 `qwen3_6_27b` 两个 gpqa 用例与 `qwen3_6_35b_a3b` 两个 aime26 用例;
2. 调整 `max_retries`:Qwen3.6 四用例钉为 2(吸收单次不走运的评估,又不让最坏情况变大);Kimi-K3 32 卡 gpqa 用例(约 3600 s)钉为 1,快速失败而非默认跑三次评估;
3. 移除 `test_npu_dp_attention.py` 中 `base-b-test-4-npu-a3` 注册项上的 `disabled="Temporarily disable this test case due to DTS2026091464350"`,重新启用该用例。

关键 diff 摘录(qwen3_6_27b_1p_gpqa.py):

```diff
+    max_retries = 2
     accuracy = 0.878
-    max_retries = 1
     generation_config = {
         "max_tokens": 81920,
-        "temperature": 1.0,
+        "temperature": 0,
```

验证:四分区精度 job([Run #36510795245](https://github.com/sgl-project/sglang/actions/runs/36510795245),head `934aa80dd1`)中四个用例全部以 `temperature = 0` 通过,阈值不变(阈值为 `baseline - DATASET_FLUCTUATION / question_count`,gpqa_diamond 5/198、aime26 2/30)。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅改动 NPU 测试配置(4 个 Qwen3.6 用例 + 1 个 Kimi-K3 用例 + 1 个 DP attention 用例),不涉及模型、内核或运行时代码。作者向 reviewer 提示:gpqa 阈值(baseline 0.878)是旧温度下测得的,在贪心解码下偏紧(散布 0.833 - 0.879,围绕 169/198 阈值两侧约各一题),两个 gpqa 阈值可能仍需重新定基。

## 关联

- [Run #36510795245](https://github.com/sgl-project/sglang/actions/runs/36510795245)(验证 run)
- DTS2026091464350(内部缺陷号,正文未给出链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
