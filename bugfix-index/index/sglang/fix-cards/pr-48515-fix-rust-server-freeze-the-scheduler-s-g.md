# PR #48515: [Fix] rust-server: Freeze the scheduler's garbage collector after the startup warmup

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#48515](https://github.com/sgl-project/sglang/pull/48515) |
| 作者 | metamergebot |
| 合并日期 | 2026-11-07 |
| labels | (none) |
| 热度 | 👍 0 · 💬 9 |
| 分类 | Overlap / Scheduler |
| bug 类型 | correctness |

## 一句话结论
Rust 前端从未执行 Python 模式的启动后 `/freeze_gc`,`--gc-threshold` 被静默忽略,每次全量 GC 都遍历整个启动堆;warmup 结束后内部发送 FreezeGCReq 修复。

## 问题现象
Python 前端模式下,launcher 在 startup warmup 结束后(或设置 `--skip-server-warmup` 时立即)POST `/freeze_gc`:tokenizer manager 将其转为携带 `--gc-threshold` 的 `FreezeGCReq`,每个 scheduler rank 应用阈值并调用 `gc.freeze()`,使启动期分配的对象不再进入后续全量回收。Rust 前端从未执行这一步:在 Rust 模式下 `--gc-threshold` 被静默忽略,调度器里每一次 full collection 都要遍历整个启动堆。正文未给出具体量化影响数据,影响定位为 GC 行为偏离 Python 模式(启动对象未冻结)。

## 根因
PR 正文披露:`--gc-threshold` 只经 `FreezeGCReq` 到达 scheduler,而 Rust 前端没有任何路径在启动后发送该请求,导致标志被静默忽略、启动堆不被 `gc.freeze()` 冻结。由 diff 推断:Rust `startup_warmup` 成功后直接返回,没有对应 Python launcher 的 post-warmup `/freeze_gc` 调用;且 `FreezeGCReq` 无应答,不能走 Rust 既有的 await 控制路径。

## 修复方案
Rust 侧补齐 Python 模式的 freeze GC 语义:
1. Rust `ServerArgs` 新增 `gc_threshold` 字段,`config.py` `_build_server_args` 从 `--gc-threshold` 填充。
2. `app.rs` 的 `startup_warmup`:warmup 成功、跳过 warmup、以及 disaggregation 模式下 warmup 失败但服务器保持存活三种情形,都调用新的 `CoreHandle::freeze_gc`;失败或超时的 warmup 不发送(与 Python 一致),失败即上报并返回。
3. 新增 `TmEvent::OneWay` 单向事件:`FreezeGCReq` 无应答,不能走 await 控制路径(会挂死);intake 将其推到 abort 已在用的保留 lane(`push_priority`),满载请求环也无法丢弃,并回报是否入队,失败记为 error。
4. `Scheduler.handle_freeze_gc`:仅在没有内嵌 Rust server 时才把请求转发到 detokenizer socket——Rust 模式下 detokenizer 跑在 scheduler 进程内,已被冻结,且没有进程读该 socket:

```python
if self.rust_server is None:
    self.ipc_channels.send_to_detokenizer.send_output(recv_req, recv_req)
```

5. 传播语义对齐 Python:无本地控制广播时由 DP0 listener 广播到 TP 组、其余 listener 副本丢弃;`--enable-dp-attention-local-control-broadcast` 时各 DP 组冻结自己的 rank;PP 阶段经正常转发接收。不为 Rust server 新增 `/freeze_gc` HTTP 路由(Python 模式的唯一生产调用方是 launcher 自身的 post-warmup 请求,Rust 现在内部完成)。
测试:`app.rs` 四情形冻结时序测试、`to_scheduler_tests.rs` 满环不丢消息测试、`test_rust_server_extension.py` detokenizer socket 跳过测试,以及多个 runtime 测试先取走启动 `FreezeGCReq`。

## 影响与回归风险
- 影响窗口:正文未说明;适用于 Rust 前端(`sglang-server`)部署。
- 回归面:Rust 启动流程新增一次 post-warmup freeze 调用,失败仅记 error 不阻断启动;`Scheduler.handle_freeze_gc` 的 detokenizer 转发新增 `rust_server is None` 条件,Python 嵌入场景行为不变;保留 lane 复用给 one-way 控制消息,abort 仍在同一 lane,饱和语义由测试覆盖。本 PR 位于 Rust-server 栈(#48455→…→#48815)中间层,基于 #48512 分支,#48492 合入后其测试需先取走启动 FreezeGCReq;CI 栏为整栈顶部 #48815 绿后原子合并整条链。GC 冻结与 Python 模式效果一致(无模型执行变化,Accuracy/Speed 测试均 N/A)。

## 关联
- Stack:#48455 → #48457 → #48512 → #48515 → #48518 → #48523 → #48572 → #48753 → #48797 → #48798 → #48818 → #48815(正文 CI note 列出)
- 关联:#48512(本 PR stacked 之上)、#48492(合并 main 引入的测试 `drain_end_release_is_latched`)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
