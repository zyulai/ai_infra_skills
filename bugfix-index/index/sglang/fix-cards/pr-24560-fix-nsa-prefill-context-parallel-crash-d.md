# PR #24560: fix: NSA prefill context parallel crash (dsv4-rebase)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24560](https://github.com/sgl-project/sglang/pull/24560) |
| 作者 | yhyang201 |
| 合并日期 | 2026-05-07 |
| labels | deepseek |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
NSA prefill CP 误用 attention TP 组(CP 时收缩为 1)致 shape 不匹配崩溃;统一改用 CP 组并重建元数据。

## 问题现象
正文描述:开启 `--enable-nsa-prefill-context-parallel` 时在 `dsv4-rebase` 上崩溃。`dsv4-rebase` 通过 `attn_cp_size` 引入了 `_ATTN_CP` 进程组,但 CP 相关操作仍使用 `get_attention_tp_*`(指向 `_ATTN_TP`,CP 激活时该组收缩为 size 1),造成 metadata、compressor allgather、indexer 中的 shape 不匹配。正文附验证结果:AIME25 on B300 8xGPU TP=8 CP=8(DeepSeek-V4-Flash,480 evals):max_tokens=default 91.46%;max_tokens=400k 96.88%。

## 根因
正文披露:CP 操作使用了错误的进程组访问函数——`get_attention_tp_rank/size` 返回的是 `_ATTN_TP` 组的 rank/size,而 CP 激活时 `_ATTN_TP` 收缩为 1,与 `_ATTN_CP` 的真实 rank/size 不符,导致各处 shape 不匹配并崩溃。

## 修复方案
- `layers/attention/deepseek_v4_backend.py`:`apply_cp_reindex` 的 cp_rank/cp_size 改用 `get_attention_cp_rank/size`;`make_core_attn_metadata` 中移除 round-robin 分支下的 `apply_cp_reindex()` 调用(时机后移)。
- `layers/attention/dsv4/compressor.py`:compressor allgather 的 world size 改用 `get_attention_cp_size()`。
- `models/deepseek_v4.py`:多处 `self.cp_rank`/`self.cp_size` 改用 CP 组;round-robin split 下在模型 forward 中对 core_attn_metadata 执行 `apply_cp_reindex()` + `init_flashmla_related()`,并用 `init_forward_metadata_indexer` 重建 indexer_metadata。
- `arg_groups/deepseek_v4_hook.py`:round-robin 分支设置 `enable_dp_attention=True` 与 `attn_cp_size = tp_size // dp_size`,日志加入 attn_cp_size。

```diff
+    server_args.enable_dp_attention = True
     server_args.moe_dense_tp_size = 1
+    server_args.attn_cp_size = server_args.tp_size // server_args.dp_size
@@ -197,8 +199,8 @@ def init_compression_metadata(self):
     def apply_cp_reindex(self) -> None:
-        cp_rank = get_attention_tp_rank()
-        cp_size = get_attention_tp_size()
+        cp_rank = get_attention_cp_rank()
+        cp_size = get_attention_cp_size()
```

## 影响与回归风险
- 回归面:DeepSeek-V4(NSA)prefill CP 路径:rank/size 来源从 TP 组整体换成 CP 组;round-robin split 模式下强制开启 DP attention 并按 tp/dp 推导 attn_cp_size,改变该模式下的参数配置;CP reindex 的执行时机从 metadata 构建期移到模型 forward,且新增 indexer metadata 重建;非 CP 或非 dsv4 路径不受影响。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
