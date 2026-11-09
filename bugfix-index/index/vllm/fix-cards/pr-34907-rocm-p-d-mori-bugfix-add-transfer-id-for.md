# PR #34907: [ROCm][P/D][MORI][BugFix] Add transfer_id for moriio_connector so moriio_connector to restore P/D functionality

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34907](https://github.com/vllm-project/vllm/pull/34907) |
| 作者 | rasmith |
| 合并日期 | 2026-03-16 |
| labels | bug, documentation, kv-connector, ready, rocm |
| 热度 | 👍 5 · 💬 21 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
PR #27987 给 completion id 加随机后缀后 MoRI KV connector 永久挂起;为 moriio_connector 引入 transfer_id 并在 scheduler/worker 间同步映射关系,恢复 P/D。

## 问题现象
使用 MoRIIOConnector 做 P/D(预填充/解码分离)时,KV connector 无限挂起(hang):它在等待一个永远不会到达的 completion id。起因是 PR #27987 为区分日志中的服务,给 completion id 引入了随机后缀(即引入 transaction id);其他 KV connector 都随之更新了,但该改动无法套用到 MORI KV connector 上,导致 P/D 功能失效。作者在单机与双机上用 1P1D justfile 验证功能恢复。

## 根因
正文披露:#27987 引入 completion id 随机后缀后,MORI KV connector 未同步适配,decode worker 需要收到 completion id 才能完成配对,于是永远等不到,连接器无限挂起。正文还指出难点:decode worker 要收到 completion id,且在把 transaction id 与 completion id 关联时要避免竞态。(另注:此前的修复尝试 #32937 已停更,继任的 #32630 需要改 scheduler,被认为不可取。)

## 修复方案
为 MoRI 引入独立的 `transfer_id`(`TRANSFER_PREFIX = "tx"` + uuid4,由 proxy 生成并放入 `kv_transfer_params`),并把 transfer_id↔request_id 映射在 scheduler 与 worker 间同步:
- `moriio_common.py`:`MoRIIOConstants` 新增 `TRANSFER_PREFIX`;`WriteTask`/`LayerTransferPlan`/`ReqMeta` 增加 `transfer_id` 字段;`MoRIIOConnectorMetadata` 增加 `transfer_id_to_request_id` 映射;`add_new_req` 从 `kv_transfer_params["transfer_id"]` 读取。
- `moriio_connector.py`:scheduler 侧 `update_state_after_alloc()` 中 `map_request_id()`,`request_finished()` 中 `unmap_request_id()`;`build_connector_meta()` 把映射塞进 `MoRIIOConnectorMetadata`;worker 侧 `start_load_kv()` 从 metadata 取回映射,`get_finished()` 用 transfer_id→request_id 换算完成集合;`send_notify_block` 携带 transfer_id。
- `moriio_engine.py`:`done_remote_allocate_req_dict` 改按 transfer_id 索引,完成通知与消息匹配由 `COMPLETION_PREFIX` 改为 `TRANSFER_PREFIX`。
- `moriio_toy_proxy_server.py`:为每个请求生成 `tx-<uuid4>` 并同时发给 prefill 与 decode。

关键 diff 摘录:

```diff
+        self.transfer_id_to_request_id: dict[TransferId, ReqId] = {}
...
-            if msg_str.startswith(MoRIIOConstants.COMPLETION_PREFIX):
+            if msg_str.startswith(MoRIIOConstants.TRANSFER_PREFIX):
```

## 影响与回归风险
- 影响窗口:受 PR #27987 合入之后的版本影响(正文以该 PR 为起因,未给出确切版本号)。
- 回归面:MoRI/MoRIIO connector 全路径(common/metadata、connector 的 scheduler 与 worker 两侧、engine 的写任务与消息处理、toy proxy 示例);完成通知从按 request_id 改为按 transfer_id 匹配,若对端未同步升级会产生映射缺失(worker 中已有 warning 日志覆盖该情形);其他 KV connector 不受影响。

## 关联
- [PR #27987](https://github.com/vllm-project/vllm/pull/27987)(起因)
- [PR #32937](https://github.com/vllm-project/vllm/pull/32937)(先前修复尝试,已停更)
- [PR #32630](https://github.com/vllm-project/vllm/pull/32630)(另一修复尝试,需改 scheduler)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
