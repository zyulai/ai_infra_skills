# PR #19805: Fix /health regression from early prebound socket listen

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19805](https://github.com/sgl-project/sglang/pull/19805) |
| 作者 | mmangkad |
| 合并日期 | 2026-03-04 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#17754 预绑定 socket 过早 listen,/health 在服务就绪前被接收而挂起;改为 bind-only、uvicorn 前才 listen。

## 问题现象
正文给出对比日志。修复前:uvicorn 启动日志后约 7 秒出现 `TimeoutError: timed out ← Health check hangs`、`Runner failed with exception: ReadTimeout`(`requests.exceptions.ReadTimeout: HTTPConnectionPool...`),即模型/子进程初始化期间 /health 探针的 TCP 连接被接受但无人应答,健康检查挂起超时。修复后:/health 立即返回 `503 Service Unavailable`、重试可用,服务就绪(`The server is fired up and ready to roll!`)后返回 200。

## 根因
正文披露:#17754 为端口冲突 fail-fast 加入了 HTTP socket 预绑定,但同时把 socket 在模型/子进程初始化阶段就放入 `listen()` 状态,使 uvicorn/FastAPI 尚未就绪时 /health 的 TCP 连接即可被接收,造成健康探针挂起/超时。

## 修复方案
- `python/sglang/utils.py` 的 `_prebind_listening_socket()`:只 `bind()` 预留端口,不再提前 `listen()`;删除 `SO_REUSEPORT` 以保持独占预留的冲突检测语义;
- `python/sglang/srt/entrypoints/http_server.py`:在 `uvicorn.run(...)` 之前才执行 `reserved_socket.listen(128)`,listen 推迟到应用可服务时;finally 中为 `multi_tokenizer_args_shm` 与 `_global_state` 增加防御性判空清理,避免早期失败时清理报错。

```diff
+        # Delay listen() until uvicorn startup to avoid accepting probe traffic
+        # while model/subprocess initialization is still in progress.
+        reserved_socket.listen(128)
```

## 影响与回归风险
- 影响窗口:#17754 合入后至本 PR 之前(正文明确该 PR 引入回归)。
- 回归面:launch_server 的端口预留与 HTTP 启动序列;预绑定 socket 不再提前 listen,初始化期间对同端口的连接行为随之改变;multi-tokenizer 共享内存与全局状态的清理路径增加判空;SO_REUSEPORT 移除后多进程同端口绑定语义收紧。

## 关联
- sgl-project/sglang#17754(引入回归的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
