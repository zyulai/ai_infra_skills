# PR #4542: Fix the incorrect args in benchmark_and_profiling.md

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#4542](https://github.com/sgl-project/sglang/pull/4542) |
| 作者 | tianyuzhou95 |
| 合并日期 | 2025-03-18 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
benchmark_and_profiling.md 文档中 bench_serving 示例误用 `--model-path` 参数,应为 `--model`;按文档执行会报错,本 PR 修正两处示例。

## 问题现象
正文未描述具体现象(正文为空)。(由 diff 推断)`docs/references/benchmark_and_profiling.md` 中两处「send profiling request from client」的示例命令,使用 `python -m sglang.bench_serving --backend sglang --model-path meta-llama/Llama-3.1-8B-Instruct --num-prompts ... --profile` 的写法;用户照文档发起 profiling 请求时会因参数名不存在而失败。文档同节还说明需要在 server 与 client 两侧都设置 `SGLANG_TORCH_PROFILER_DIR`,否则 trace 文件无法正确生成。

## 根因
正文未披露根因(正文为空)。(由 diff 推断)文档示例把 `sglang.launch_server` 的 `--model-path` 参数名照搬到了 `sglang.bench_serving` 命令上,而 bench_serving 客户端实际接受的参数是 `--model`,两处示例均写错,属于文档命令与 CLI 实际参数不一致的问题。

## 修复方案
只改文档 `docs/references/benchmark_and_profiling.md`,把两处示例命令中的 `--model-path` 改为 `--model`,模型名与 `--num-prompts`、`--sharegpt-output-len`、`--profile` 等其余参数保持不变。关键 diff 摘录(两处同类修改之一):

```diff
-  python -m sglang.bench_serving --backend sglang --model-path meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile
+  python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile
```

## 影响与回归风险
- 回归面:纯文档改动,不触及任何运行时代码,不影响 server/bench_serving 行为;唯一「影响」是文档读者从此看到的命令可直接复制执行。第二处示例为 `--num-prompts 2` 的小 trace 场景,同样改为 `--model`。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
