# PR #50766: [Bugfix] serving_llama70B_tp4 benchmark was silently running at tensor_parallel_size=1

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#50766](https://github.com/vllm-project/vllm/pull/50766) |
| 作者 | wjabbour |
| 合并日期 | 2026-08-03 |
| labels | bug, performance, ready, ci/build, llama |
| 热度 | 👍 2 · 💬 1 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论
性能基准配置里 `serving_llama70B_tp4_random_128_128` 从未设置 TP 度,静默继承默认值 1 跑成了 tp1;补上 `"tensor_parallel_size": 4`。

## 问题现象
连续 perf.vllm.ai 基准中的 `serving_llama70B_tp4_random_128_128` 测试实际上一直以 TP=1 运行,而不是名字所标的 tp4。该测试的名字和固定的 8192 `max_num_batched_tokens`、`async_scheduling` 等服务端参数都是按 4 卡运行配置的,但实际 TP 度静默默认为 1,即基准结果测的并非 70B 四卡 serving 性能。发现途径:作者在制作 AMD 版基准配置(`serving-tests-rocm.json`,镜像 `serving-tests.json`)时,通过 `run-performance-benchmarks.sh` 的真实 `merge_serving_tests_stream` 合并逻辑核验该条目才暴露。

## 根因
正文披露:该测试条目的 `server_parameters` 从未设置 `tensor_parallel_size`,于是静默继承文件 `defaults` 块里的 `tensor_parallel_size: 1`。修复前用合并逻辑验证:`select(.test_name=="serving_llama70B_tp4_random_128_128") | .server_parameters.tensor_parallel_size` 输出 `1`,修复后输出 `4`。

## 修复方案
在 `.buildkite/performance-benchmarks/tests/serving-tests.json` 的 `serving_llama70B_tp4_random_128_128` 条目 `server_parameters` 中补一个字段:

```json
"server_parameters": {
  "model": "meta-llama/Llama-3.3-70B-Instruct",
  "tensor_parallel_size": 4,
  "async_scheduling": "",
  ...
}
```

## 影响与回归风险
- 影响窗口:正文未写明版本范围;该配置错误存在多久未量化,但「This test has presumably been running as tp1 ... this whole time」。
- 回归面:仅一行 JSON 配置改动,只影响该基准条目在 CI 中的服务启动参数(TP 度 1→4,意味着该测试将占用 4 张 GPU)。作者验证了 JSON 仍然合法、文件内全部 7 个测试名仍能干净合并、`pre-commit` 通过,其余条目无回归。注意 #46870 的 `check-json` 钩子只校验语法,抓不住这类语义错误。

## 关联
- #46870(正文提及:同文件但无关条目,不能防止本 bug)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
