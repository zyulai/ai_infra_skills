# PR #1765: Fix perf regression for set_kv_buffer

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#1765](https://github.com/sgl-project/sglang/pull/1765) |
| 作者 | merrymercy |
| 合并日期 | 2024-10-23 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论

`set_kv_buffer` 每次写入都走 `torch.compile` 版 `copy_two_array` 造成性能回退;改为内联直接索引赋值,按需做 dtype/store_dtype 转换。

## 问题现象

正文为空,未描述具体现象(无复现、无数据)。从标题可知是 `set_kv_buffer` 的性能回退;diff 新增注释指认 `copy_two_array` 的编译版本在单元测试 `python3 -m unittest test_bench_serving.TestBenchServing.test_offline_throughput_non_stream_small_batch_size` 中更慢。

## 根因

正文为空,未披露根因。(由 diff 推断)原实现在每次 `set_kv_buffer` 时都调用被 `@torch.compile(dynamic=True)` 修饰的 `copy_two_array`,即使 dtype 已匹配也要经过编译函数与无条件转换,在小批量离线吞吐场景反而更慢;修复将拷贝逻辑内联进 `set_kv_buffer`,仅在 `cache_k.dtype != self.dtype` 时转换,`store_dtype != dtype` 时用 `view(self.store_dtype)` 写入。

## 修复方案

改 `python/sglang/srt/mem_cache/memory_pool.py` 的 `set_kv_buffer`,移除对 `copy_two_array` 的调用,内联为:

```diff
-        copy_two_array(
-            loc,
-            self.k_buffer[layer_id],
-            cache_k,
-            self.v_buffer[layer_id],
-            cache_v,
-            self.dtype,
-            self.store_dtype,
-        )
+        if cache_k.dtype != self.dtype:
+            cache_k = cache_k.to(self.dtype)
+            cache_v = cache_v.to(self.dtype)
+        if self.store_dtype != self.dtype:
+            self.k_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
+            self.v_buffer[layer_id][loc] = cache_v.view(self.store_dtype)
+        else:
+            self.k_buffer[layer_id][loc] = cache_k
+            self.v_buffer[layer_id][loc] = cache_v
```

`copy_two_array` 函数保留,仅在其上方加注释说明编译版本在上述单元测试中更慢。

## 影响与回归风险

- 影响窗口:正文未说明;受影响路径为 `memory_pool.py` 的 KV cache 写入(`set_kv_buffer`),是每步解码都会走的热点路径。
- 回归面:所有使用该 memory pool 的模型/后端的 KV 写入路径。内联后行为语义与原 `copy_two_array` 一致(dtype 转换 + store_dtype view),但绕过了 `torch.compile`;若原编译版本在某些场景有融合优化,该收益会丢失。fp8 等 store_dtype != dtype 的场景走 view 分支,需保证 view 语义正确。

## 关联

正文未列出(正文为空)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
