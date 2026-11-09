# PR #27004: fix(disagg): correct DSA/SWA state-page transfer mismatch in PD disaggregation

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27004](https://github.com/sgl-project/sglang/pull/27004) |
| 作者 | kflansburg |
| 合并日期 | 2026-06-03 |
| labels | (none) |
| 热度 | 👍 1 · 💬 6 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

PD 分离部署 DSA 模型时,last chunk 的状态页列表比 decode 注册多一页,传输 worker 崩溃;seq_len 改取 min(fill_ids, origin_input_ids)。

## 问题现象

PD(disaggregated)部署 DSA(sparse-attention,formerly NSA)模型——实测环境:GLM/DSA-FP8、`page_size=64`、mooncake 传输后端——prefill 传输 worker 在向 decode 发送混合「state/extra」pool 时崩溃:

```
File ".../disaggregation/mooncake/conn.py", in _send_kvcache_generic
    prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(...)
ValueError: operands could not be broadcast together with shapes (2,) (0,)
```

## 根因

正文披露两个层次的问题:

1. **根因(长度不匹配)**:最后一个 chunk 上,prefill 侧按 `seq_len = len(req.fill_ids)` 枚举 DSA/SWA 状态页列表;但发送时 `fill_ids` 已包含 prefill 期间采样的 token(`output_ids` 在 `send_kv_chunk(last_chunk=True)` 之前被 append),比 prompt 长。而 decode 侧(`DecodePreallocQueue`)按 `len(origin_input_ids)` 注册目标状态页,主 KV-pool 传输也已 clamp 到 `min(len(fill_ids), len(origin_input_ids))`。当采样 token 跨过页边界时,prefill 会多发出一个 decode 从未注册的状态页,src/dst 索引数组长度不一致。
2. **辅助函数脆弱**:`group_concurrent_contiguous` 只在 *source* 为空时提前返回,否则假设两输入等长;长度不等时要么抛出难懂的 NumPy broadcast 错误,要么(形状可广播时)静默产生错误分组。

## 修复方案

1. `python/sglang/srt/disaggregation/prefill.py`(`send_kv_chunk`):last chunk 的状态页 `seq_len` 改为 `min(len(req.fill_ids), len(req.origin_input_ids))`,与 decode 侧注册及主 pool 的 clamp 范围对齐;该值由 `_dsa_payload` 与 `_swa_payload` 共享,两条混合状态路径一并修正:

```diff
-            seq_len = len(req.fill_ids)
+            seq_len = min(len(req.fill_ids), len(req.origin_input_ids))
```

2. `python/sglang/srt/disaggregation/common/utils.py`(`group_concurrent_contiguous`):任一索引数组为空即返回 `[], []`;两者非空但长度不等时抛出带说明的 `ValueError`,把隐晦 broadcast 错误/静默错分组变成清晰行为(纵深防御)。
3. 新增 `test/registered/unit/disaggregation/test_disaggregation_wire.py` CPU 单测:覆盖连续/非连续分组、双空、空 src、空 dst 回归用例与长度不等报错。

## 影响与回归风险

- 回归面:PD 分离的 KV 传输路径——所有走 `send_kv_chunk` last chunk 的 DSA/SWA 混合状态页枚举,以及 `group_concurrent_contiguous` 的全部调用方。常见情形(采样 token 未跨页边界)下 `min(len(fill_ids), ...)` 等于原值,行为不变;仅移除多枚举的尾页。等长输入下 helper 行为不变。无性能影响(两次 size 检查、一个 `min()`)。作者建议在 DSA PD 分离部署上做端到端验证。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
