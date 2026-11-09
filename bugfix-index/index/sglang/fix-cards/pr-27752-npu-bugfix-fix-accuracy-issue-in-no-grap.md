# PR #27752: [NPU][Bugfix] Fix accuracy issue in no-graph with MTP

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27752](https://github.com/sgl-project/sglang/pull/27752) |
| 作者 | iridiumine |
| 合并日期 | 2026-06-12 |
| labels | npu, run-ci |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
NPU 关图+MTP 时三处无条件 `mark_forward_metadata_ready` 阻断注意力元数据初始化导致精度回归,改为仅 cuda-graph 路径标记;另修图模式 attr_name。

## 问题现象
NPU 上开启 `--disable-cuda-graph` 并使用 MTP(Multi-Token Prediction)投机解码时出现精度回归。正文未给出回归前的具体错误数值;修复后验证(ASCEND_USE_FIA=1、SGLANG_ENABLE_SPEC_V2=1,EAGLE 3 步/topk1/4 draft tokens,multi-layer eagle):MiMo-V2-Flash-w8a8 在 gsm8k 200 样本上 mean_acc 0.96。

## 根因
PR 正文披露:三个位置无条件调用 `mark_forward_metadata_ready()`,阻止了 NPU 后端在 non-cuda-graph(eager)路径上正确初始化注意力 metadata:
- `eagle_info_v2.py` 的 `prepare_for_extend_to_fill_draft_kvcache`:NPU 非图路径上 `init_forward_metadata` 会做 pre-padding,而 `prepare_mlp_sync_batch` 对 batch 做 padding 后 block_table 形状不匹配;
- `multi_layer_eagle_worker_v2.py` 的 `verify`:NPU 非图 verify 需要 forward_extend 在 padding 后初始化 metadata;
- `multi_layer_eagle_worker_v2.py` 的 `_draft_extend_for_decode`:batch 级标记阻塞了 `draft_runner_list[1+]` 初始化各自的 per-runner attention metadata。

另有一个独立问题(正文披露):`npu_graph_runner.py` 的 graph replay 硬编码 `self.attr_name[AttentionArch.MLA]`(即 `"actual_seq_lengths_kv"`),对 MiMoV2 模型不正确——其要求 `"actual_seq_kvlen"`,导致开 cuda graph 且不开 MTP 时图模式专属的精度回归(输出乱码)。

## 修复方案
- `eagle_info_v2.py` / `multi_layer_eagle_worker_v2.py`:上述三处 `mark_forward_metadata_ready()` 改为 `if not _is_npu or can_cuda_graph:` 条件执行,NPU 非图路径让 `forward_extend` 在 padding 后重新初始化 metadata;
- `npu_graph_runner.py`:replay 改用 `self._get_update_attr_name()` / `self._get_update_attr_type()` 动态选择器,按模型架构返回正确 attr_name;
- `mem_cache/allocator/swa.py`(正文未单独说明,由 diff 推断):NPU 分支的 `full_to_swa_index_mapping` 高级索引赋值改为 `torch_npu.npu_scatter_nd_update_` 原位 scatter 更新。

关键 diff 摘录:

```diff
-            forward_batch.mark_forward_metadata_ready()
+            # On NPU with --disable-cuda-graph, block_table shape won't match
+            # after prepare_mlp_sync_batch padding; defer re-init to
+            # forward_extend (post-pad) instead.
+            if not _is_npu or can_cuda_graph:
+                forward_batch.mark_forward_metadata_ready()
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围,未写。
- 回归面:NPU 上 EAGLE/multi-layer-eagle 投机解码路径(图/非图两条)、NPU graph replay 的 attr 选择、SWA 分配器 NPU decode 分配路径。非 NPU 平台三处条件恒为真(`not _is_npu`),行为不变(由 diff 推断);NPU cuda-graph 路径(`can_cuda_graph` 为真)也保持原标记行为。swa.py 的 scatter 改写语义应与原索引赋值等价(由 diff 推断)。

## 关联
- [#27091](https://github.com/sgl-project/sglang/pull/27091)(diff 注释中提及的断言相关 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
