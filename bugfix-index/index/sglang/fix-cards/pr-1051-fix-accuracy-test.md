# PR #1051: Fix accuracy test

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#1051](https://github.com/sgl-project/sglang/pull/1051) |
| 作者 | merrymercy |
| 合并日期 | 2024-08-12 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
正文为空的测试修复:把 `set_ulimit()` 移入 `run_eval()`,并调整精度/吞吐测试的线程数、采样数与阈值。

## 问题现象
正文未描述具体现象(正文为空)。按标题与 diff,问题围绕 accuracy 测试本身:测试直接调用 `run_eval()` 时未经过原来的 CLI 入口,而 `set_ulimit()` 原先只在 CLI 入口的 `__main__` 分支调用(由 diff 推断);精度测试的采样与线程参数、吞吐阈值也需要调整。

## 根因
正文未披露根因(正文为空)。从 diff 描述改动所暗示的机制:`set_ulimit()` 此前位于 `python/sglang/test/run_eval.py` 的 `__main__` 解析参数之后,作为库函数被测试调用的 `run_eval(args)` 不会执行它(由 diff 推断);其余为测试参数与阈值的经验性调整,diff 未说明理由。

## 修复方案
四处改动(均由 diff 描述):
- `python/sglang/test/run_eval.py`:`set_ulimit()` 从 `__main__` 分支移到 `run_eval(args)` 函数开头,任何调用方都会执行。
- `python/sglang/test/simple_eval_humaneval.py`:清理无用 import(移除 `json`、`logging`、`multiprocessing`、`Counter/defaultdict`、`BytesIO`、`Any/Tuple`、`blobfile as bf`、`HUMAN_EVAL`)。
- `test/srt/test_eval_accuracy_large.py`:mmlu `num_examples=None→3000`、三个测试 `num_threads` 2048→1024;mmlu 阈值 `score >= 0.70` 提到 `>= 0.71`;humaneval/mgsm_en 断言加上 `{metrics}` 输出。
- `test/srt/test_serving_throughput.py`:CI 断言 A100 吞吐阈值由 1300→1400、1400→1450,注释改为 "A100 (PCIE) performance"。

```diff
 def run_eval(args):
+    set_ulimit()
+
     if "OPENAI_API_KEY" not in os.environ:
```

## 影响与回归风险
- 回归面:所有调用 `run_eval()` 的评测路径(现在都会执行 `set_ulimit()`);mmlu/humaneval/mgsm_en 精度测试与 serving throughput 测试的运行参数与通过标准;humaneval 评测脚本去掉 `blobfile` 依赖相关 import(由 diff 推断)。模型推理路径不受影响。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
