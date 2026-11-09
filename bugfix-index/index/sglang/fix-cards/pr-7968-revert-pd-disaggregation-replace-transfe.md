# PR #7968: Revert "[PD Disaggregation] replace transfer with batch transfer for better performance (#7236)"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7968](https://github.com/sgl-project/sglang/pull/7968) |
| 作者 | fzyzcjy |
| 合并日期 | 2025-07-12 |
| labels | bug, high priority |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回退 #7236:PD 分离 KV cache 传输从批量 batch_transfer_sync 恢复为逐段 transfer_sync。

## 问题现象
正文未描述具体现象,仅 cc @ShangmingCai 并整体回退 #7236;labels 标记为 bug、high priority,表明该批量传输改动引入了需要紧急回退的问题。

## 根因
正文未披露根因。(由 diff 推断)#7236 把 mooncake conn 的 `send_kvcache` 中逐块的 `transfer_sync` 聚合为 src/dst 地址与长度列表后一次 `batch_transfer_sync`;本 PR 恢复为循环内逐段 `transfer_sync`,并在状态码非 0 时立即返回错误。批量传输合入后出现问题导致回退,具体失败机制正文未说明。

## 修复方案
整体 revert #7236,`python/sglang/srt/disaggregation/mooncake/conn.py` 的 `send_kvcache` 内 `process_layer` 恢复逐块传输与错误检查:

```diff
-                src_addr_list.append(src_addr)
-                dst_addr_list.append(dst_addr)
-                length_list.append(length)
-            return self.engine.batch_transfer_sync(
-                mooncake_session_id, src_addr_list, dst_addr_list, length_list
-            )
+                status = self.engine.transfer_sync(
+                    mooncake_session_id, src_addr, dst_addr, length
+                )
+                if status != 0:
+                    return status
+            return 0
```

## 影响与回归风险
- 影响窗口:#7236 合入后至本 PR 之前(由正文回退对象推及)。
- 回归面:PD 分离(mooncake 传输引擎)的 KV cache 发送路径;#7236 声称的批量传输性能收益随之失去,大块传输吞吐可能回落;逐段传输在出错时可经状态码提前返回。

## 关联
- sgl-project/sglang#7236(被回退的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
