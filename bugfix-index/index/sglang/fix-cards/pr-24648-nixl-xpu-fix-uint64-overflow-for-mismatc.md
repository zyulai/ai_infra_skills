# PR #24648: [NIXL][XPU] Fix uint64 overflow for mismatched P/D TP sizes (e.g. prefill_tp=1, decode_tp=2)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24648](https://github.com/sgl-project/sglang/pull/24648) |
| 作者 | Jianhong-Zhang |
| 合并日期 | 2026-05-12 |
| labels | intel, run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

Intel XPU 上 KV cache 指针地址第 63 位置位,`send_kvcache_slice` 用 int64 数组承载致溢出,改用 np.uint64 修复。

## 问题现象

影响 P/D 分离(prefill/decode 分离)且 TP size 不匹配(如 `prefill_tp=1, decode_tp>1`)、会执行到 `send_kvcache_slice` 的场景。正文未描述具体报错或错误表现,仅给出受影响范围。

## 根因

正文未披露根因(diff 中新增代码注释给出的机制,由 diff 推断):在 Intel XPU 上,`torch.int` 表示的地址超出 `np.int64` 的范围——XPU 的内存地址第 63 位被置位(注释举例 `0xffff81ab54e01000`),而 `send_kvcache_slice` 里把 KV 数据指针、prefill/decode 索引、token 偏移都装进 `np.int64` 数组,发生无符号溢出。

## 修复方案

`python/sglang/srt/disaggregation/nixl/conn.py` 的 `send_kvcache_slice`:把参与地址计算与 NIXL 传输的数组统一改为 `np.uint64`——`kv_args.kv_data_ptrs` 与 `dst_kv_ptrs` 先转成 `np.uint64` 数组再传给 `get_mha_kv_ptrs_with_pp`;`prefill_indices`、`dst_indices`、`token_offsets` 的 dtype 从 `np.int64` 改为 `np.uint64`。

关键 diff 摘录:

```diff
+        # torch.int exceeds np.int64 range on Intel XPU (addresses have bit 63 set, e.g.
+        # 0xffff81ab54e01000). Use np.uint64 to prevent overflow on XPU.
+        kv_data_ptrs = np.array(self.kv_args.kv_data_ptrs, dtype=np.uint64)
+        dst_kv_ptrs = np.array(dst_kv_ptrs, dtype=np.uint64)
         src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
-            self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
+            self.get_mha_kv_ptrs_with_pp(kv_data_ptrs, dst_kv_ptrs)
         )
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅改动 NIXL P/D 分离传输中 `send_kvcache_slice` 一个函数的 dtype 选择;TP size 匹配的正常路径不经此函数的 mismatched 分支,非 XPU 平台上地址落在 int64 范围内,uint64 与 int64 计算结果等价(由 diff 推断)。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
