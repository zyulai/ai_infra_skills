# PR #53023: [CI] Fix MultiConnector accuracy test lifecycle

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#53023](https://github.com/vllm-project/vllm/pull/53023) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-08-21 |
| labels | kv-connector |
| 热度 | 👍 2 · 💬 4 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
AMD CI MultiConnector 精度测试布局间存在进程生命周期竞争(proxy 端口被占、405 被当就绪);改为 harness 持有进程组、探测 /health 并等待显存释放。

## 问题现象
AMD CI build #12250 暴露的潜在生命周期竞争:sequential 的 normal 与 cross-layer 两个测试布局之间,第一个布局在 API server 尚在关闭时把 proxy 留在 8192 端口,下一个布局复用了陈旧进程,替换的 proxy 无法绑定端口;就绪检查用 `curl -s` 打 POST-only endpoint,HTTP 405 响应也被当作成功。

## 根因
正文披露的机制:测试脚本对启动的服务与 proxy 没有归属跟踪——服务用裸 `eval "$BASE_CMD" &` 启动、清理用 `pkill -f "vllm serve"` 加固定 sleep,无法保证上一布局的进程已退出、端口已释放;就绪探测 `curl -s localhost:${port}/v1/completions` 既不带失败处理(`-f`)又选了只接受 POST 的端点,405 同样返回成功;布局间仅 `sleep 3`,不等 ROCm GPU 显存回落。

## 修复方案
`tests/v1/kv_connector/nixl_integration/run_multi_connector_accuracy_test.sh` 重写生命周期管理:
- `start_background_process` 用 `setsid` 启动服务/proxy 并登记到 `PROCESS_GROUPS`;
- `wait_for_server` 先用 `kill -0` 确认进程存活(提前退出即报错返回退出码),再用 `curl -fs` 探测 `/health`(服务)与 `/healthcheck`(proxy),带 1200 秒 deadline;
- `cleanup_instances` 向领主与进程组发 SIGTERM,最多等 30 秒,超时 `kill -KILL` 并 `wait` 收尸;
- 新增 `wait_for_gpu_memory_release`(ROCm 上调用 `tests.utils.wait_for_memory_to_settle`)替代固定 sleep;
- `trap cleanup_on_exit EXIT` 与 SIGINT/SIGTERM 处理取代旧的 `kill $(jobs -pr)`。

## 影响与回归风险
- 回归面:仅测试脚本,不影响产品代码;CI 行为变化:进程被强制归属与清理、就绪判定更严格(405/5xx 不再算就绪)、布局启动前等待 GPU 显存释放;若被测服务健康端点不可用,测试会更早失败而不是误判就绪。

## 关联
正文未列出 issue/PR 链接;问题来源为 AMD CI build #12250(Buildkite 链接)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
