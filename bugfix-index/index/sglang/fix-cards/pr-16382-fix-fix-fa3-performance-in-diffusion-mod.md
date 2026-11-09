# PR #16382: [Fix]Fix FA3 Performance in Diffusion Model

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#16382](https://github.com/sgl-project/sglang/pull/16382) |
| 作者 | HydraQYH |
| 合并日期 | 2026-01-05 |
| labels | run-ci, diffusion |
| 热度 | 👍 7 · 💬 5 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
Diffusion 模型走 FA3 时误入 varlen 调度路径,内核退化为 VarlenDynamicPersistentTileScheduler;改为直接传整段张量、cu_seqlens 置 None,恢复 StaticPersistentTileScheduler。

## 问题现象
修复 issue [#16196](https://github.com/sgl-project/sglang/issues/16196)。正文未描述具体症状文本,但给出修复前后实际编译出的 FA3 内核签名对比:修复前主循环模板参数中出现 `(bool)1` 与 `flash::VarlenDynamicPersistentTileScheduler<...>`,且 CollectiveEpilogueFwd 模板中对应 bool 为 `(bool)1`;修复后变为 `(bool)0` 并改用 `flash::StaticPersistentTileScheduler<(bool)0>`,即修复前 Diffusion 模型实际落到 FA3 的 varlen 动态持久化调度内核。正文的 Accuracy Tests 与 Benchmarking and Profiling 章节均为空模板,未提供精度或性能数据。

## 根因
正文未明文披露根因。结合正文给出的内核签名对比与 diff 可推断:(由 diff 推断)multimodal_gen diffusion 路径的 FA3 forward 在 upstream 分支里把 q/k/v reshape 成 varlen 布局(`bsz * seqlen` 折叠)并构造 cu_seqlens 后调用 `flash_attn_varlen_func_upstream`,导致 FA3 选择 varlen 动态持久化调度器(VarlenDynamicPersistentTileScheduler)的慢路径;而 diffusion 场景各 batch 序列等长,无需 varlen 处理。

## 修复方案
改 `python/sglang/multimodal_gen/runtime/layers/attention/backends/flash_attn.py` 的 forward(upstream 分支):
- 删除对 k/v shape 的解包与 `bsz * seqlen` 的 reshape,`q_`/`k_`/`v_` 直接取 `contiguous()` 后的原始张量;
- 删除 `_get_cu_seqlens` 构造,`flash_attn_varlen_func_upstream` 的 cu_seqlens 参数改传 `None`,仍传 `seqlen, seqlen` 与 `softmax_scale=self.softmax_scale`。
这样 FA3 不再走 varlen 内核实例化,恢复 StaticPersistentTileScheduler 内核(见正文前后签名对比)。

关键 diff 摘录:
```diff
-            bsz_k, seqlen_k, nheads_k, d_k = k_shape
-            bsz_v, seqlen_v, nheads_v, d_v = v_shape
-            q_ = query.contiguous().reshape(bsz * seqlen, nheads_q, d)
-            k_ = key.contiguous().reshape(bsz * seqlen, nheads_k, d)
-            v_ = value.contiguous().reshape(bsz * seqlen, nheads_v, d)
-            cu = _get_cu_seqlens(q_.device.index, bsz, seqlen)
+            q_ = query.contiguous()
+            k_ = key.contiguous()
+            v_ = value.contiguous()
             out = flash_attn_varlen_func_upstream(
-                cu,
-                cu,
+                None,
+                None,
```

## 影响与回归风险
- 回归面:仅 `multimodal_gen` diffusion 运行时 FA3 attention 后端的 upstream 调用分支;varlen 参数改传 None、序列长度改由 `seqlen` 传入,若该分支存在 batch 内不等长序列的调用方,布局假设改变需关注;正文未提供精度/性能测试结果(相关 checklist 项未勾选)。

## 关联
- [#16196](https://github.com/sgl-project/sglang/issues/16196)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
