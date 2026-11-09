# PR #36459: [NPU] Fix evalscope accuracy parsing and add glm5_1 aime26 request timeout

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#36459](https://github.com/sgl-project/sglang/pull/36459) |
| 作者 | pllimax |
| 合并日期 | 2026-08-31 |
| labels | deepseek, npu, run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
evalscope 1.11+ 百分比精度未被正则匹配且未归一化,新增格式适配并除以 100;另给 aime26 测试加 7200s 请求超时。

## 问题现象
正文 Motivation 仅写「Fix some test cases bug」,并按 Modifications 分点说明:
1. evalscope 1.11+ 将 accuracy 报告为百分比形式(如 `66.67%`),现有正则无法匹配该输出格式,匹配到的数值也未对基线做归一化,导致 NPU 精度测试解析失败/结果不可比。
2. glm5_1 多节点 aime26 精度测试中,长时间运行的 evalscope 请求会超时。
3. kimi_k2_6 64k 性能测试中,`--mem-fraction-static 0.662` 使 KV pool 过小(max allowed length 61050 < 64000),而 `0.7` 又会在 MLP `down_proj` 量化 matmul 处 prefill OOM。

## 根因
正文披露:evalscope 升级到 1.11+ 后输出为表格中的百分比格式,旧正则只匹配纯小数;同时解析结果未除以 100 归一化到 0-1 区间再与基线比较。其余两项为测试参数选择问题(超时缺失、显存配比不当),正文同样按现象直接给出调整值,未披露更深层的代码缺陷。

## 修复方案
1. `python/sglang/test/ascend/e2e/test_npu_accuracy_utils.py`:新增适配 evalscope 1.11+ 表格格式的正则 ` Accuracy\s*[↑↓]?\s*│\s*[^│]*│\s*\d+\s*│\s*([\d.]+)%?\s*│`;`re.findall` 改为 `re.finditer`,以便同时拿到完整匹配(判断 `%`)与捕获组;命中含 `%` 时除以 `100.0` 归一化:
```diff
-                matches = re.findall(pattern, full_output)
+                matches = list(re.finditer(pattern, full_output))
                 if matches:
-                    final_accuracy = float(matches[-1])
+                    final_accuracy = float(matches[-1].group(1))
+                    if "%" in matches[-1].group(0):
+                        final_accuracy /= 100.0
```
2. glm5_1 aime26 测试的 `router_args` 增加 `--request-timeout-secs 7200`。
3. kimi_k2_6 64k perf 测试 `--mem-fraction-static` 由 `0.662` 改为 `0.68`(KV pool 约 77000 tokens,兼顾动态激活空间)。
4. qwen3_next aime25 测试由 `run_accuracy_multiple(n_runs=3)` 三次取平均改为单次 `run_accuracy`,并删除基类中的 `n_runs` 与 `run_accuracy_multiple`。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:改动集中在 Ascend NPU 测试基建与注册测试用例(`test_npu_accuracy_utils.py` 影响所有走 evalscope 的 NPU 精度测试;删除 `run_accuracy_multiple` 会影响仍在调用它的用例;超时与 mem-fraction 仅影响对应两个测试),不影响模型推理与内核代码路径。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
