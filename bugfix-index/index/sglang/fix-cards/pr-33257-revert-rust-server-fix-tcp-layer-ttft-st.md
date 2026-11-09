# PR #33257: Revert "[rust-server] fix TCP-layer TTFT stalls"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33257](https://github.com/sgl-project/sglang/pull/33257) |
| 作者 | sherlockwu |
| 合并日期 | 2026-08-02 |
| labels | run-ci, run-ci-extra |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论

整体回滚 #33026:rust-server 恢复 `std::net::TcpListener` 绑定方式,撤销 TcpSocket + SO_REUSEADDR + 16MB 收包缓冲 + TCP_NODELAY 一套改动。

## 问题现象

正文未描述具体现象。正文仅一句 "Reverts sgl-project/sglang#33026",未填写复现条件、影响面或根因说明(CI Base/Extra 均为通过状态)。

## 根因

正文未披露根因。(由 diff 推断)被回滚的 #33026(标题为 "fix TCP-layer TTFT stalls")曾为缓解 Nagle/delayed-ACK 导致的约 13 ms TTFT 惩罚,把 `serve()` 的入参从 `std::net::TcpListener` 换成预先同步绑定的 `tokio::net::TcpSocket`,并加 `set_reuseaddr(true)`、16MB `set_recv_buffer_size`、`tap_io` 设置 `TCP_NODELAY`;本次将其全部撤销,恢复原有的 std listener + `from_std` 方式。

## 修复方案

按 revert 提交恢复两个文件:

`rust/sglang-server/src/api_server.rs`:`serve()` 入参恢复为 `std::net::TcpListener`,用 `tokio::net::TcpListener::from_std(listener)` 接入 tokio reactor;删除 `tap_io` 设置 `TCP_NODELAY` 的整段代码(含 `ListenerExt` 引用)。

`rust/sglang-server/src/runtime.rs` 的 `start()`:

```diff
-    let socket = match addr {
-        std::net::SocketAddr::V4(_) => tokio::net::TcpSocket::new_v4(),
-        std::net::SocketAddr::V6(_) => tokio::net::TcpSocket::new_v6(),
-    }
-    .map_err(|e| format!("socket for {addr} failed: {e}"))?;
-    socket.set_reuseaddr(true)...
-    if let Err(e) = socket.set_recv_buffer_size(16 * 1024 * 1024) {...}
-    socket.bind(addr)...
+    let listener = std::net::TcpListener::bind(cfg.rust_server_args.http_addr)
+        .map_err(...)?;
+    listener.set_nonblocking(true).map_err(...)?;
```

调用 `api_server::serve` 处相应把 `socket` 改回传 `listener`。

## 影响与回归风险

- 影响窗口:含 #33026 改动的版本受影响(依据正文 revert 声明;正文未给出具体版本号)。
- 回归面:改动集中在 rust server 的端口绑定与监听路径(`api_server.rs`、`runtime.rs`)。恢复后不再设置 `TCP_NODELAY`/`SO_REUSEADDR`/收包缓冲,keep-alive 连接可能重新暴露 #33026 想解决的 Nagle 相关 TTFT 停顿;IPv6/绑定语义回到 std listener 行为。

## 关联

- [#33026](https://github.com/sgl-project/sglang/pull/33026)(正文声明被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
