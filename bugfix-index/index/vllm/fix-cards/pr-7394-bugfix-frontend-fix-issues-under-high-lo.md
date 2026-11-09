# PR #7394: [Bugfix][Frontend] Fix Issues Under High Load With `zeromq` Frontend

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#7394](https://github.com/vllm-project/vllm/pull/7394) |
| 作者 | robertgshaw2-redhat |
| 合并日期 | 2024-08-21 |
| labels | ready |
| 热度 | 👍 5 · 💬 11 |
| 分类 | Frontend / Serving |
| bug 类型 | general |

## 一句话结论
mp 前端每请求开一条 zmq 连接,约 1000 并发即触顶报 Too many open files;改为 inproc 代理 + 单条 ipc 连接,并禁用 HWM 防丢消息。

## 问题现象
v0.5.4 的多进程(mp)前端在重负载下失败:用 `vllm serve meta-llama/Meta-Llama-3-8B-Instruct` 起服务,`benchmark_serving.py` 以 `--num-prompts 10000 --request-rate 1000` 压测,系统内约 1000 个活跃请求时 server 直接死掉(dies)。影响面:mp 前端模式下并发请求数被硬性封顶。

## 根因
正文披露:mp 前端的 `zeromq` RPC 用 `ipc` 连接,每个新请求都在 RPC Server 侧建立一条连接,N 个活跃请求就是 N 条连接;而 `zmq.MAX_SOCKETS` 默认 1023,超过即报 `Too many open files`,构成活跃请求硬上限。即使调高该值,底层 unix socket 数仍受系统 `ulimit` 限制,UX 很差。启用代理设计后正文又披露了第二个根因:高负载下 zeromq 因 high-water mark(默认 1000)丢消息——DEALER 到达 HWM 后静默、ROUTER 到达 HWM 后丢消息;由于 RPC 双端都归 vLLM 控制、任一侧死亡即需整体重启,且 mp 设计下的 CPU OOM 在非 mp 下同样会发生,故直接禁用 HWM。

## 修复方案
在 `vllm/entrypoints/openai/rpc/client.py` 的 `AsyncEngineRPCClient` 中引入代理:后台 asyncio task 跑 `run_proxy`,在 `from_api_server`(ROUTER,`inproc://`)与 `to_rpc_server`(DEALER,`ipc`)之间按 identity 转发消息,全系统只有 1 条 unix socket;每个 API server 线程经 `to_proxy_socket()` 用 DEALER 连 inproc 地址。同时:`MAX_SOCKETS` 设为 `SOCKET_LIMIT` 且低于 `VLLM_RPC_SOCKET_LIMIT_CUTOFF=2000` 时直接报错;算出 `limit_concurrency = socket_limit // 2 - 2` 传给 uvicorn(过载返回 503);所有 socket `set_hwm(0)`(HWM 无穷)。`server.py` 侧由 ROUTER bind 改为 DEALER connect,`get_*_config` 合并为 `get_config` 并把异常回传客户端;`async_llm_engine.py` 的输出队列改为 `_is_raisable` 判定并支持取消;健康检查由 `CHECK_HEALTH`/`HEALTHY` 串改为 `IS_SERVER_HEALTHY`/`SUCCESS` 并可复用现有 socket。新增 `test_load.py`、`test_accuracy.py`(lm-eval gsm8k 500 并发)。作者称该架构 E2E 快约 5%(少大量建 unix socket 的 syscall);附 QPS=10 基准:main 51.4 / pr 49.0 / `--disable-frontend-multiprocessing` 72.3。

## 影响与回归风险
- 影响窗口:正文提到问题存在于 v0.5.4 的 mp 前端;未给出修复前版本范围。
- 回归面:mp 前端整条 RPC 链路(client/server 消息格式改 multipart + identity 路由、健康检查语义变更、engine 侧 `limit_concurrency` 新接口、输出队列异常/取消处理重写);HWM 置 0 意味着慢消费下靠整体重启兜底而非限流;uvicorn 会在 mp 模式下启用 `limit_concurrency`,过载请求得 503。

## 关联
- #7309
- #7290

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
