# PR #15371: [DLLM] Fix dLLM regression

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15371](https://github.com/sgl-project/sglang/pull/15371) |
| 作者 | ClawSeven |
| 合并日期 | 2025-12-19 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 8 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
get_new_batch_prefill 对 chunked_req 无条件 init_next_round_input 造成 dLLM 回归;改为仅动态 chunking 开启时执行。

## 问题现象
正文未描述具体现象——PR 正文的 Motivation、Modifications 等小节均为空模板占位注释,无任何症状、复现或影响面描述。标题表明这是 dLLM(diffusion LLM)相关的一次回归修复;diff 中同步把 `test/srt/dllm/test_llada2_mini.py` 的 gsm8k 精度断言从 `> 0.6` 收紧到 `> 0.88`,可侧面反映修复前 dLLM 模型(llada2_mini)精度受损、修复后恢复到更高水平。

## 根因
正文未披露根因。由 diff 推断:`Scheduler.get_new_batch_prefill` 中,原逻辑在 `self.chunked_req is not None` 时就无条件调用 `self.chunked_req.init_next_round_input()`,再视 `enable_dynamic_chunking` 决定是否动态调整 chunk 大小;修复后仅当 `chunked_req is not None and enable_dynamic_chunking` 才进入该逻辑(含 `init_next_round_input()` 与 `predict_next_chunk_size`)。即此前在动态 chunking 关闭(如 dLLM 场景)时也会对 chunked_req 执行 `init_next_round_input()`,这一多余的状态更新被本次回退,回归随之消除——具体因果机制正文未说明,以上为改动所暗示的机制。

## 修复方案
改 `python/sglang/srt/managers/scheduler.py` 的 `get_new_batch_prefill`:将"chunked_req 存在即无条件 `init_next_round_input()`、再按需动态调 chunk size"的写法,收敛为仅在 `self.chunked_req is not None and self.enable_dynamic_chunking` 时计算 `history_len` 并调用 `predict_next_chunk_size` 调整 `chunked_prefill_size`。同时把 `test/srt/dllm/test_llada2_mini.py` 中 gsm8k 断言由 `assertGreater(metrics["accuracy"], 0.6)` 提高到 `0.88`(吞吐断言 `output_throughput > 150` 不变)。

关键摘录:
```diff
-        if self.chunked_req is not None:
-            self.chunked_req.init_next_round_input()
-            if self.enable_dynamic_chunking:
-                history_len = len(self.chunked_req.prefix_indices)
-                dynamic_size = self.predict_next_chunk_size(history_len)
-                if dynamic_size is not None:
-                    chunked_prefill_size = dynamic_size
+        if self.chunked_req is not None and self.enable_dynamic_chunking:
+            history_len = len(self.chunked_req.prefix_indices)
+            dynamic_size = self.predict_next_chunk_size(history_len)
+            if dynamic_size is not None:
+                chunked_prefill_size = dynamic_size
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:改动触及所有模型的 prefill 调度路径——动态 chunking 开启时行为等价(仍会 init_next_round_input 并动态调 size);动态 chunking 关闭时,chunked_req 不再被调用 `init_next_round_input()`,依赖该无条件调用的路径需关注。dLLM 测试阈值收紧后,精度低于 0.88 的回归将被 CI 拦截。

## 关联
正文未列出关联 issue/PR。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
