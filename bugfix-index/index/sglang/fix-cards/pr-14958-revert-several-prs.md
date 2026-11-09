# PR #14958: Revert several PRs

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14958](https://github.com/sgl-project/sglang/pull/14958) |
| 作者 | zhyncs |
| 合并日期 | 2025-12-12 |
| labels | bug, documentation, high priority, run-ci |
| 热度 | 👍 0 · 💬 24 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
提交 `70758d45` 破坏了 B200 CI,整体回退该提交:移除 CI 重试/改时配置与 remote instance weight loader 的 transfer_engine 后端。

## 问题现象
正文未描述具体现象,仅说明被回退的 commit `70758d457eb4594e74b1076e503022bb0975c379` "has broken b200 cis"——即 B200 CI 因该提交而损坏。labels 带 `bug` 与 `high priority`。

## 根因
正文未披露根因,只给出结论:该 commit 破坏了 B200 CI,故回退。(由 diff 推断)被回退的提交做了两类改动:一是给 CI 引入重试机制(`SGLANG_CI_ENABLE_RETRY` 环境变量、各 suite 的 `--enable-retry` 参数)并同步下调了各 job 的 `timeout-minutes`;二是为 remote instance weight loader 新增 `transfer_engine` 后端(基于 mooncake `TransferEngine` 的 RDMA 权重传输,新增 `remote_instance_weight_loader_backend` 参数、`register_memory_region_v2` 内存注册、`/get_remote_instance_transfer_engine_info` HTTP 端点等),本次回退将两者一并撤销。

## 修复方案
整体 revert commit `70758d457eb4594e74b1076e503022bb0975c379`,按 diff 覆盖:
1. `.github/workflows/pr-test.yml`:删除 `SGLANG_CI_ENABLE_RETRY: true` 与所有 suite 的 `--enable-retry`;各 job 超时回调(如 1-GPU 30→40 分钟、4-GPU 20→30 分钟、B200 30→40 分钟等,恢复回退前的数值)。
2. `server_args.py` / `load_config.py`:删除 `remote_instance_weight_loader_backend`、`remote_instance_weight_loader_support_transfer_engine` 两个参数及相关 CLI 参数与校验逻辑。
3. `engine.py` / `http_server.py` / `scheduler.py` / `tp_worker.py` / `model_runner.py` / `loader.py` / `remote_instance_weight_loader_utils.py`:删除 TransferEngine 初始化、内存注册、per-rank 权重信息上报、`load_model_from_remote_instance_by_transfer_engine` 及相关辅助函数;remote instance 加载恢复为仅 NCCL 路径,`model_runner` 的 `load_model` 恢复用 `get_model`。

注:pack 的 diff 在 40000 字符处截断(文件粒度),截断部分的改动无法从本素材确认。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:CI 侧,重试机制与缩短的超时被撤销,失败用例不再自动重跑;功能侧,`load_format=REMOTE_INSTANCE` 的 transfer_engine 后端(mooncake RDMA 传输)整体不可用,仅保留 NCCL 传输路径,依赖该后端或 `--remote-instance-weight-loader-backend transfer_engine` 的部署会失去支持。`_handle_remote_instance_weight_loader_support_transfer_engine` 中 mooncake 探测逻辑恢复到回退前形态。

## 关联
正文未列出(仅提及被回退的 commit hash `70758d457eb4594e74b1076e503022bb0975c379`)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
