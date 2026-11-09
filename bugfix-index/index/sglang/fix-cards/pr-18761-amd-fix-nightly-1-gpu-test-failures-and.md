# PR #18761: [AMD] Fix nightly 1-GPU test failures and bench_serving regression

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#18761](https://github.com/sgl-project/sglang/pull/18761) |
| 作者 | michaelzhang-ai |
| 合并日期 | 2026-02-15 |
| labels | amd, run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Frontend / Serving |
| bug 类型 | regression |

## 一句话结论
修复 #18566 造成的 bench_serving list/numpy 类型回归,并调整 AMD nightly 超时、Kimi 任务与套件注册。

## 问题现象
AMD 1-GPU nightly 单测失败(正文给出失败 run:[21971874189/job/63475035487](https://github.com/sgl-project/sglang/actions/runs/21971874189/job/63475035487),以及 nightly 700 与 720 的 run 21973649080、21973649043)。其中 bench_serving 回归按正文修改条目表现为:`sample_generated_shared_prefix_requests` 在 `compute_random_lens` 返回的 list 上调用需要 numpy 数组的 `.reshape()`/`.item()`。

## 根因
正文披露:#18566 把 bench_serving.py 的 `compute_random_lens` 改为返回 `List[int]`(经 `.tolist()`),但 `sample_generated_shared_prefix_requests` 随后调用 `.reshape()` 与 `.item()`,这两个操作要求 numpy 数组,从而产生回归。

## 修复方案
- `python/sglang/bench_serving.py`:`sample_generated_shared_prefix_requests` 中对 `compute_random_lens` 结果包一层 `np.array()` 再 reshape;`.item()` 取值改为 `int()`(含 `system_prompt_lens[i]`);正文声明对其他调用方无影响;
- AMD nightly workflow(`nightly-test-amd.yml`、`nightly-test-amd-rocm720.yml`):1-GPU 单测 `--timeout-per-file` 600→900s、`timeout-minutes` 60→90(MI30x/MI35x);
- 移除 Kimi-K2 任务,由 Kimi-K2.5 取代(rocm720 与 rocm700 对齐,新增 MI35x 8-GPU Kimi-K2.5 accuracy 任务);
- `run_suite.py` 注册缺失的 `nightly-amd-1-gpu` 与 `nightly-amd-1-gpu-mi35x` 套件;
- 顺带将 `test_gpt_oss_eval_amd.py` 的 accuracy_threshold 0.79→0.75。

## 影响与回归风险
- 影响窗口:#18566 合入后至本 PR 之前(正文明确)。
- 回归面:bench_serving 的 shared-prefix 采样路径、AMD nightly CI workflow 的超时与任务矩阵、套件注册;gpt-oss AMD accuracy 测试阈值放宽 0.04。

## 关联
- sgl-project/sglang#18566(引入 bench_serving 回归的 PR)
- [run 21971874189/job/63475035487](https://github.com/sgl-project/sglang/actions/runs/21971874189/job/63475035487)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
