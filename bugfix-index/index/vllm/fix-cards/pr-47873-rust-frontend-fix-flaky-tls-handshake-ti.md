# PR #47873: [Rust Frontend] Fix flaky `tls_handshake_timeout_drops_silent_client` test

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47873](https://github.com/vllm-project/vllm/pull/47873) |
| 作者 | tahsintunan |
| 合并日期 | 2026-07-15 |
| labels | ready, rust |
| 热度 | 👍 2 · 💬 1 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论
Rust TLS 测试手动 advance 暂停时钟并与 socket read 竞态导致 flaky;去掉手动 `advance`/`yield_now`,让 paused clock 自动推进裸 `read().await`,并断言关闭恰好发生在握手 deadline(修后 20/20 稳定)。

## 问题现象
`tls_handshake_timeout_drops_silent_client` 测试不稳定(flaky)。复现方式:执行 `for i in $(seq 20); do cargo nextest run -p vllm-server tls_handshake_timeout_drops_silent_client; done`,修复前并非每次都通过。问题只涉及测试本身,正文未描述生产环境的具体故障。

## 根因
PR 正文披露:测试手动推进被暂停的 tokio 时钟(manual `advance`),然后让握手关闭与 socket 读操作相互竞争(race),时序不确定导致测试偶发失败。即测试对调度时序的假设不稳固,而非被测的 TLS 握手超时逻辑有错。

## 修复方案
修改 `rust/src/server/src/tls_tests.rs` 的 `tls_handshake_timeout_drops_silent_client`:

1. 删除手动的 `tokio::task::yield_now().await` 与 `tokio::time::advance(TLS_HANDSHAKE_TIMEOUT + 1ms).await`,改为记录 `start = tokio::time::Instant::now()`;
2. 直接 `let read = tcp.read(&mut buf).await;`(tokio paused clock 会自动推进裸 read),断言从 `matches!(read, Ok(Ok(0)) | Ok(Err(_)))` 改为 `matches!(read, Ok(0) | Err(_))`;
3. 新增断言 `start.elapsed() >= tls::TLS_HANDSHAKE_TIMEOUT`,确认关闭发生在握手 deadline 而非更早;
4. 新增 `rust/.config/nextest.toml`,配置 `slow-timeout = { period = "60s", terminate-after = 2 }`,让挂死的测试在每测 120s 预算内被杀掉、直接失败而非整体卡死。

关键 diff 摘录:

```diff
-    tokio::task::yield_now().await;
-    tokio::time::advance(tls::TLS_HANDSHAKE_TIMEOUT + Duration::from_millis(1)).await;
-    tokio::task::yield_now().await;
+    let start = tokio::time::Instant::now();
     let mut buf = [0u8; 1];
-    let read = tokio::time::timeout(Duration::from_secs(1), tcp.read(&mut buf)).await;
+    let read = tcp.read(&mut buf).await;
```

## 影响与回归风险
- 影响窗口:正文未说明受影响的版本/提交范围。
- 回归面:全部改动都在测试与测试基建(`tls_tests.rs` 一个测试函数、新增 nextest 配置),不影响任何生产代码路径。测试语义略有加强:除「静默客户端会被关闭」外,还断言关闭时刻不早于 `TLS_HANDSHAKE_TIMEOUT`,若未来实现提前关闭会被捕获;nextest 的 slow-timeout 只作用于超时预算,不影响通过用例的执行。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
