# PR #26825: Fix TokenizerManager crash on top_logprobs with tensor values

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#26825](https://github.com/sgl-project/sglang/pull/26825) |
| 作者 | kflansburg |
| 合并日期 | 2026-06-03 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Frontend / Serving |
| bug 类型 | crash |

## 一句话结论
top_logprobs 值为多元素 Tensor 时真值判断歧义崩溃并 SIGKILL prefill 进程;改 `is not None` 判断修复。

## 问题现象
客户端发送 `top_logprobs > 0` 的请求时,`detokenize_top_logprobs_tokens` 中 `token_logprobs_val[i]` 为多元素 `torch.Tensor`(而非注解的 `List[float]`)会抛出:

```
RuntimeError: Boolean value of Tensor with more than one value is ambiguous.
```

该异常从 detokenization handler 传播出去,被 `print_exception_wrapper` 捕获后调用 `kill_process_tree(os.getpid(), include_parent=True)`,以 SIGKILL 杀死整个 prefill 进程(exit code -9)。在 disaggregated PD 部署中,表现为只要客户端带 `top_logprobs > 0` 发请求,prefill 就不断重启(prefill restart storm)。

## 根因
正文披露:`detokenize_top_logprobs_tokens` 用 `if token_logprobs_val[i]:` 的真值检查来跳过空位置。当该位置的值是多元素 `torch.Tensor` 而非函数注解所写的 `List[float]` 时,Python 对张量求布尔真值会因歧义直接抛 RuntimeError。实际用于标记跳过位置的哨兵是 `None`,真值检查与注解都与实际类型不符。

## 修复方案
改 `python/sglang/srt/managers/tokenizer_manager.py` 的 `detokenize_top_logprobs_tokens`:把真值检查替换为显式的 `is not None` 测试——与标记跳过位置的实际哨兵一致,且对 list 与 tensor 两种取值都成立。另新增 CPU 单测 `test/registered/unit/managers/test_tokenizer_manager_top_logprobs_tensor.py`,用多元素 tensor 值复现崩溃,并验证 None 哨兵路径与普通 list 路径。

关键摘录:
```diff
         ret = []
         for i in range(len(token_logprobs_val)):
-            if token_logprobs_val[i]:
+            if token_logprobs_val[i] is not None:
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:改动为单行判断条件,仅影响 `TokenizerManager.detokenize_top_logprobs_tokens` 的过滤逻辑。多元素 tensor 位置现在会被正常 detokenize 而非崩溃;`None` 位置仍输出 `None`;普通 `List[float]` 路径行为不变(单测覆盖)。不再触发 kill_process_tree,Prefill 进程稳定性恢复;对模型输出与推理速度无影响。

## 关联
正文未列出关联 issue/PR。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
