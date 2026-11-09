# PR #7952: Tiny fix mooncake log warning wrong output

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7952](https://github.com/sgl-project/sglang/pull/7952) |
| 作者 | fzyzcjy |
| 合并日期 | 2025-07-13 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
Mooncake KV 传输的告警日志在截断后才打印,两个长度值看起来相同;把 warning 移到截断之前修复。

## 问题现象
修复前,`transfer_worker` 中当 `len(chunked_dst_kv_indice)` 小于 `len(kv_chunk.prefill_kv_indices)` 时打出的 warning 日志,两边的数值看起来完全一样("w/o this change, it seems both output are identical",正文附截图)。这使告警失去诊断价值,无法看出两者长度差异。仅影响日志输出,不影响数据本身。

## 根因
正文披露:不改动的话,日志里两个输出看起来相同("it seems both output are identical"),即 warning 打印的是错误(相同)的内容;正文未给出更深入的机制分析。

机制(由 diff 推断):原代码先执行 `kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[: len(chunked_dst_kv_indice)]` 截断,再调用 `logger.warning` 打印 `len(kv_chunk.prefill_kv_indices)`;因此日志中的第二个长度已经被截断成与第一个相同,告警里永远看不出差异。

## 修复方案
改动在 `python/sglang/srt/disaggregation/mooncake/conn.py` 的 `transfer_worker`,仅调整语句顺序,把 warning 移到截断之前:

```diff
-                            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
-                                : len(chunked_dst_kv_indice)
-                            ]
                             logger.warning(
                                 f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                             )
+                            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
+                                : len(chunked_dst_kv_indice)
+                            ]
```

截断逻辑本身不变,只改变日志打印时机,使其打印截断前的真实长度。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:仅影响 Mooncake PD 分离传输路径中命中该长度不匹配分支时的日志内容;`prefill_kv_indices` 的最终截断结果与修复前一致,不影响功能正确性。风险极低。

## 关联
正文提到 cc @ShangmingCai,未列出 issue/PR 链接

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
