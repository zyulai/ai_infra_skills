# PR #27863: [Fix][MTP][MM] Fix EAGLE v2 chunked-prefill next-token chain crash on multimodal models due to placeholder tokens

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27863](https://github.com/sgl-project/sglang/pull/27863) |
| 作者 | litmei |
| 合并日期 | 2026-06-15 |
| labels | run-ci, run-ci-extra |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
chunked-prefill 取下一 prompt token 时未过滤多模态占位 token,超出 vocab 触发 embedding 层崩溃,现按 vocab_size 校验。

## 问题现象
多模态模型开启 SpecV2(EAGLE 投机解码)处理多模态请求时触发报错。开启 `ASCEND_LAUNCH_BLOCKING` 后观察到崩溃发生在 embedding 层内。触发条件:chunked prefill 场景下,`origin_input_ids` 在 fill 边界处恰为多模态占位(hash)token。验证环境:kimi-k2.6-w4a8 + kimi-k2.6-eagle3,NPU 16 卡 TP/DP,`--chunked-prefill-size 32768 --enable-multimodal --speculative-algorithm EAGLE3` 等。

## 根因
正文披露:该回归由 PR #26800 引入。`_compute_chunked_req_next_prompt_token` 方法未考虑多模态场景——多模态上下文中 `input_ids` 可能包含远超 vocab_size 的占位/填充值,该值被作为下一个 prompt token 返回后送入 embedding 层,导致报错。

## 修复方案
改 `python/sglang/srt/managers/schedule_batch.py`:给 `_compute_chunked_req_next_prompt_token` 增加 `vocab_size` 参数,取 token 前先按 vocab_size 校验,越界(即多模态占位 hash token)则返回 None:

```diff
 def _compute_chunked_req_next_prompt_token(
     chunked_req: Optional[Req],
+    vocab_size: int,
 ) -> Optional[int]:
     ...
-    return int(chunked_req.origin_input_ids[fill_len])
+    if origin_ids[fill_len] < vocab_size:
+        return int(origin_ids[fill_len])
+    return None
```

调用点 `init_new` 处传入 `model_config.vocab_size`。

## 影响与回归风险
- 影响窗口:正文说明回归由 PR #26800 引入,即介于 #26800 合并至本修复之间的提交受影响。
- 回归面:`schedule_batch.py` 中 chunked_req_next_prompt_token 的计算与 SpecV2 chunked-prefill next-token 链;越界时返回 None 意味着该次不再续链,只影响含多模态占位 token 的 chunked 请求,普通文本请求(token < vocab_size)行为不变。PR 附 gsm8k(2000 题,temperature 0)与 MMMU 验证截图(CI 状态绿)。

## 关联
- [PR #26800](https://github.com/sgl-project/sglang/pull/26800)(引入回归的 PR,正文披露)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
