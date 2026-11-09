# PR #5415: [PD] Fix dynamic port support and MLA buffer for Mooncake

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#5415](https://github.com/sgl-project/sglang/pull/5415) |
| 作者 | ShangmingCai |
| 合并日期 | 2025-04-15 |
| labels | (none) |
| 热度 | 👍 6 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
Mooncake PD 分离的动态端口从固定段扫描改为系统分配、bootstrap 路由改为按 rank 的 bootstrap_key,并修复 MLA 场景 KV buffer 遍历,支持 DeepSeek 多节点(TP>8)推理。

## 问题现象
正文仅一句描述:修复动态端口实现与 MLA buffer 问题,以支持更多场景,例如 DeepSeek 的多节点推理(TP > 8)。未描述具体报错、日志或复现步骤,也未提供测试数据。可从 diff 看到旧实现的具体形态:(由 diff 推断)prefill/decode 各自从固定基址 20000/25000 起随机步进扫描可用端口并绑定 `tcp://*`;prefill 注册 bootstrap 用 tp_rank 索引 `/kv_route`;`send_kvcache` 假设 KV 指针数为层数两倍(K、V 分开两段传输)。

## 根因
正文未披露根因,仅说明「fixes dynamic port implementation and the MLA buffer issue」。从 diff 可推断:(由 diff 推断)
- 动态端口缺陷:旧 `find_available_ports` 从固定端口段(20000/25000)随机步进扫描,多节点/多实例下易冲突或扫到非本机网卡,且绑定 `tcp://*` 广播地址;prefill/decode 通过 bootstrap 服务以 tp_rank 为 key 交换端口信息,TP>8 或多节点时 rank 语义不足以唯一定位;
- MLA buffer 缺陷:`send_kvcache` 中 `layer_num = len(kv_data_ptrs) / 2` 假定 K/V 各占一半指针并分两次传输,对 MLA(每层单一 KV buffer)布局不成立。

## 修复方案
改动 6 个文件:
1. `disaggregation/mooncake/conn.py`:删除 `find_available_ports`,`start_prefill_thread`/`start_decode_thread` 改用 `get_free_port()` 并绑定 `tcp://{get_local_ip_by_remote()}:{port}`;bootstrap 注册/查询改为 `/route` + `bootstrap_key`(格式 `{bootstrap_server_url}_{engine_rank}`),`connection_pool`、`prefill_port_table` 改按 bootstrap_key(str)索引;`_register_to_bootstrap` 移入 `MooncakeKVManager` 并支持 `dist_init_addr`;`TransferInfo` 消息字段重排(room 提到首位);`send_kvcache` 改为按 `len(kv_data_ptrs)` 遍历全部指针、每段一次 `transfer_sync`(长度 `item_len * len(prefill_index)`),不再 K/V 分半。
2. `disaggregation/base/conn.py`:`BaseKVManager.__init__` 增加 `server_args: ServerArgs` 参数;`KVArgs` 增加 `gpu_id` 字段。
3. `disaggregation/decode.py` / `prefill.py`:`_init_kv_manager` 填 `kv_args.gpu_id = self.scheduler.gpu_id`,构造 KV manager 时传入 `self.scheduler.server_args`;`PrefillConstructor` 接收并保存 `scheduler`。
4. `managers/scheduler.py`:`init_disaggregation` 构造 prefill 侧时传入 `scheduler=self`。
5. `utils.py`:新增 `get_free_port()`(bind 端口 0,IPv4 失败回退 IPv6)与 `get_local_ip_by_remote()`(UDP connect 8.8.8.8 / IPv6 回退)。

关键 diff 摘录:
```diff
-        self.rank_port = find_available_ports(20000, 1)[0]
-        self.server_socket.bind("tcp://*:" + str(self.rank_port))
+        self.rank_port = get_free_port()
+        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")
```

## 影响与回归风险
- 回归面:PD 分离 Mooncake 后端全路径——端口分配与 socket 绑定地址变化;ZMQ 消息字段顺序改变(prefill/decode 两侧同步改,旧版本混用会不兼容);bootstrap HTTP 路由 `/kv_route`→`/route` 及 payload 字段变化(bootstrap server 需同版本);`BaseKVManager` 子类与 `PrefillConstructor` 构造签名变化,自定义/第三方 transfer backend 需适配;`send_kvcache` 传输分段方式变化影响传输语义。正文未提供精度或基准测试数据。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
