# PR #34219: Revert "feat(cache-sim-tee): concurrent ingest sender (fix peak tee-drop / oracle undercount)"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34219](https://github.com/sgl-project/sglang/pull/34219) |
| 作者 | NLRX-WJC |
| 合并日期 | 2026-08-10 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回滚 #34088:移除 cache-sim ingest tee 的并发发送与 `cache_sim_send_concurrency` 配置,恢复单任务串行 sender。

## 问题现象
正文未描述具体现象,仅一句话说明回滚 #34088(feat(cache-sim-tee): concurrent ingest sender (fix peak tee-drop / oracle undercount))。本 PR 的 CI States 显示 Latest PR Test (Base) 与 (Extra) 均为失败(x)。

## 根因
正文未披露根因。从 diff 看,#34088 曾为实验性 sgl-router 的 cache-sim ingest tee 引入并发发送(机制描述由 diff 推断):

- 新增 `cache_sim_send_concurrency` CLI 参数与 `RADIXARK_CACHE_SIM_SEND_CONCURRENCY` 环境变量(默认 8)及 `default_cache_sim_send_concurrency()`;
- `run_sender` 从逐条 `await` 的串行 POST 改为信号量限流的并发任务扇出,拆出 `post_one`;
- 新增 `sender_posts_concurrently_not_serially` 回归测试,并给出动机:串行 sender 吞吐上限约 `1/POST-latency`(≈110 req/s),峰值流量会撑爆有界 tee channel 造成丢记录(`result="dropped"`)、oracle 相对引擎低估。

本回滚将上述全部移除,恢复串行 sender。被回退动机(峰值 tee-drop/oracle 低估)是否重新成立,正文未评价。

## 修复方案
experimental/sgl-router 下:

- `config/cli.rs`、`config/types.rs`:删除 `cache_sim_send_concurrency` 参数、env 绑定与默认值函数
- `server/app_context.rs`:`CacheSimTee::spawn` 恢复三参调用 `(url, metrics, max_captures)`
- `server/cache_sim_tee.rs`:`run_sender` 恢复循环内逐条构造 body 并 `send().await` 的串行实现,删除 `post_one` 与并发回归测试;相关测试恢复三参构造

## 影响与回归风险
- 影响窗口:#34088 合并至本回滚之间。
- 回归面:实验性 sgl-router 的 cache-sim 观测路径——tee 发送吞吐回到串行上限(≈110 req/s 量级),峰值流量下 tee channel 溢丢与 oracle 低估可能重新出现(按 #34088 自述动机,由 diff 推断);CLI 参数与 env 的消失对已有部署是配置面破坏性变更。

## 关联
- [#34088](https://github.com/sgl-project/sglang/pull/34088)(被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
