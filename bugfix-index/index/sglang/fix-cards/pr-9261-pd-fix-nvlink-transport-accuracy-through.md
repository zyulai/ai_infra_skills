# PR #9261: [PD] Fix nvlink transport accuracy through transferring metadata with tcp

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9261](https://github.com/sgl-project/sglang/pull/9261) |
| 作者 | ShangmingCai |
| 合并日期 | 2025-08-21 |
| labels | (none) |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
PD 分离下 Mooncake nvlink_transport 传 aux metadata 掉精度,改走 TCP(zmq)收发 metadata 并回退 batch_transfer_sync 规避。

## 问题现象
标题表明 PD disaggregation 场景下经 Mooncake nvlink transport 传输存在精度(accuracy)问题。正文 Motivation 仅有 "cc: @fzyzcjy",未描述具体现象与触发条件。被删除的旧代码注释(diff 中)曾写明:"batch_transfer_sync has a higher chance to trigger an accuracy drop for MNNVL"(由 diff 推断,原 workaround 背景)。

## 根因
正文未披露根因。

从 diff 看:Mooncake 的 nvlink_transport 传输 custom mem pool 中的 aux metadata 时结果不正确(旧代码为此临时从 `batch_transfer_sync` 回退到逐块 `transfer_sync` 并注明 "Fix me when nvlink_transport of Mooncake is bug-free");本 PR 的绕法是让 aux metadata 不再经该 transport 传输,而是通过 TCP(zmq multipart)点对点收发,并恢复使用 `batch_transfer_sync` 传 KV 数据(由 diff 推断)。

## 修复方案
改 `python/sglang/srt/disaggregation/mooncake/conn.py`:
1. 新增 `AuxDataCodec`(基于 ctypes 从裸地址序列化/反序列化字节);新增 `AUX_DATA_HEADER = b"AUX_DATA"` 消息头。
2. `send_aux` 在 `enable_custom_mem_pool` 时改走新增的 `send_aux_tcp`:把 aux buffer 序列化后经 zmq 发给 decode 端(消息含 room/buffer_index/aux_index/长度/data)。
3. decode 线程收到 `AUX_DATA` 头的消息交由新增 `_handle_aux_data` 校验长度并写回目标 aux buffer;其余消息维持原 bootstrap 状态机。
4. `_transfer_data` 删除 custom mem pool 下逐块 `transfer_sync` 的旧 workaround,统一用 `batch_transfer_sync`。

```diff
-        # TODO(shangming): Fix me when nvlink_transport of Mooncake is bug-free
-        if self.enable_custom_mem_pool:
-            # batch_transfer_sync has a higher chance to trigger an accuracy drop for MNNVL, fallback to transfer_sync temporarily
-            ...
+        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
+        return self.engine.batch_transfer_sync(
+            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
+        )
```

另改 `disaggregation/utils.py`:custom_mem_pool 下传输线程的 device 从 `"cuda"` 改为 `"cpu"`(注释注明 nvlink_transport 修复后恢复)。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:mooncake PD disaggregation 的 aux metadata 传输路径。`enable_custom_mem_pool` 时 metadata 改走 TCP(新增 zmq 消息类型,decode 线程需处理),带宽/延迟特性变化;该场景下传输线程 buffer 落在 CPU;`_transfer_data` 行为恢复为批量传输,非 custom mem pool 路径不变。正文未提供 accuracy/benchmark 数据(checklist 全未勾选)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
