# PR #24230: [pd]: (Bug Fix) Incorrect out_cache_loc slicing in prepare_for_prebuilt

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24230](https://github.com/sgl-project/sglang/pull/24230) |
| 作者 | hzh0425 |
| 合并日期 | 2026-05-03 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 6 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
PD 分离 decode 侧开启 radix cache 且前缀命中时,prepare_for_prebuilt 从 0 开始切 out_cache_loc,读到 prefix 区域;改为按 pre_len 偏移切 delta 区域。

## 问题现象
PD disaggregation 且 decode 侧开启 radix cache 时,只要请求出现前缀命中(pre_len > 0),decode 侧 `prepare_for_prebuilt()` 填充的 `out_cache_loc` 就是错误的前缀位置而非本次 extend 的 delta 区域,导致 KV cache 位置错误。正文未给出具体报错日志;新增测试 `test_gsm8k_accuracy_two_passes` 用于验证 decode radix cache 不会使第二轮 GSM8K 精度相对第一轮下降超过 3%。

## 根因
正文披露:PR #19746 在 pd-decode 侧引入 radix tree 后,`_pre_alloc()` 向 `req_to_token_pool` 写入的布局为 `[prefix_locs (0..pre_len) | delta_locs (pre_len..seq_len)]`;当前缀命中(pre_len > 0)时,`prepare_for_prebuilt()` 错误地从索引 0 开始切片(`[:extend_input_len]`),读到的是 prefix 区域而不是 delta 区域。

## 修复方案
`python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py` `prepare_for_prebuilt`:对每个请求先取 `pre_len = len(req.prefix_indices)`,切片改为 `req_to_token[req.req_pool_idx][pre_len : pre_len + req.extend_input_len]`,即跳过 prefix 区域读取本次 extend 的 delta locs(pre_len 的计算位置也从循环尾部上移到切片之前)。同时在 `test/registered/distributed/test_disaggregation_decode_radix_cache.py` 新增 GSM8K 两轮精度对比测试。

```diff
+            pre_len = len(req.prefix_indices)
             chunk = self.req_to_token_pool.req_to_token[req.req_pool_idx][
-                : req.extend_input_len
+                pre_len : pre_len + req.extend_input_len
             ]
```

## 影响与回归风险
- 影响窗口:自 PR #19746 在 decode 侧引入 radix tree 起、至本修复合并前;仅影响 PD 分离且 decode 侧 radix cache 开启、发生前缀命中的请求。
- 回归面:PD disaggregation decode 侧 `prepare_for_prebuilt` 的 out_cache_loc 填充路径;无前缀命中(pre_len=0)时新切片 `[0 : extend_input_len]` 与原行为一致;radix cache 命中场景的行为由新增两轮 GSM8K 测试守护。

## 关联
- https://github.com/sgl-project/sglang/pull/19746 (引入 decode 侧 radix tree)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
