# PR #9549: [Bugfix][Misc]: fix graph capture for decoder

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#9549](https://github.com/vllm-project/vllm/pull/9549) |
| 作者 | yudian0504 |
| 合并日期 | 2024-10-21 |
| labels | ready |
| 热度 | 👍 7 · 💬 4 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
构建 GPU 输入时把 max_encoder_seq_len 误传成 max_decode_seq_len,破坏解码图捕获;改回正确实参。

## 问题现象
正文未描述具体现象,只说明该 PR 修复由 issue #8645 引入的 bug,并给出相关 issue #9448、#9417、#9409。

## 根因
正文未披露机制。从 diff 看,`vllm/worker/model_runner.py` 的 `build()` 在调用 `_get_cuda_graph_pad_size` 时,把 `max_encoder_seq_len` 误当作 `max_decode_seq_len` 实参传入(同一调用里 `max_encoder_seq_len` 又传了一次正确的 encoder 值),该错误由 #8645 引入(由 diff 推断)。

## 修复方案
改 `vllm/worker/model_runner.py` `build()` 中 `_get_cuda_graph_pad_size(...)` 的实参:`max_decode_seq_len` 位置由 `max_encoder_seq_len` 改为 `max_decode_seq_len`。

```diff
         cuda_graph_pad_size = self._get_cuda_graph_pad_size(
             num_seqs=len(seq_lens),
-            max_decode_seq_len=max_encoder_seq_len,
+            max_decode_seq_len=max_decode_seq_len,
             max_encoder_seq_len=max_encoder_seq_len)
```

## 影响与回归风险
- 影响窗口:正文未写(自 #8645 起)。
- 回归面:仅解码阶段 cuda graph pad 尺寸计算的一处实参;prefill/encoder 相关传参不变。

## 关联
- Issue #8645(引入 bug)
- Issue #9448、#9417、#9409(related issues)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
