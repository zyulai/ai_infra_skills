# RFC #35639: [RFC]: `vllm bench eval` for Unified Accuracy + Performance Evaluation

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#35639](https://github.com/vllm-project/vllm/issues/35639) |
| 作者 | AndreasKaratzas |
| 创建日期 | 2026-03-01 |
| 状态 | ⚪ closed(关闭于 2026-07-09) |
| labels | RFC, rocm, stale |
| 热度 | 👍 1 · 💬 19 |
| 分类 | Other |

## 一句话结论
提议新增 `vllm bench eval` 子命令,在同一次服务会话中同时完成 lm_eval 精度评测与 Prometheus 性能采集,一次运行产出一条自描述 JSONL 记录。

## 提案动机
CI 与硬件验证环境已经在跑 lm_eval 套件:起一个 `vllm serve`、发数千请求、再关掉,消耗大量 GPU 小时,但只拿到精度数字,没有任何推理性能数据——没有 TTFT、TPOT、吞吐,也没有缓存利用率。由此带来三个问题:性能回归要等用户发版后报障才被发现;硬件特有问题(如 MI300X 上的 ROCm kernel 回归、8 卡节点上的 NCCL 变更)在 CI 精度面板上不可见;精度与性能分属两套系统、没有共同出处,无法关联「这次提交精度涨了 0.3 分但 TTFT p95 回归 40%」这类结论。RFC 的定位是给本来就要付费的评测任务加装观测,而不是再开一个新任务。

## 提出的方案
- 新增 `vllm bench eval` 子命令,高层流程:collect-env → 托管子进程启动 `vllm serve` → 后台线程以 2s 间隔轮询 `/metrics` → 并发运行 lm_eval → 停止采集 → 追加 JSONL 记录 → 关闭服务。
- 性能指标直接读 vLLM 现有 Prometheus `/metrics` 端点,不需要新增埋点;采集器直接解析直方图 bucket,因此能给出 p50/p95/p99,而不是仅由 `_sum/_count` 推出的均值;指标覆盖 TTFT、TPOT、端到端延迟、token/请求吞吐、GPU 缓存与队列峰值占用。
- 每次运行追加一条 JSONL 记录,含四段:`metadata`(run ID、模型、任务、时间戳)、`accuracy`(lm_eval 各任务结果)、`performance`(延迟分位数、吞吐、峰值资源占用)、`environment`(完整 `vllm collect-env` 输出),记录自描述、可独立解读。
- `vllm collect-env` 在服务启动前调用,避免对 GPU 初始化计时造成干扰。
- 提供 `--existing-server` 标志,跳过服务生命周期管理、直接连接已运行实例,适配多步 CI 流水线与本地开发。
- 支持多任务与张量并行(如 `--tasks hellaswag,gsm8k,mmlu --tp 8`),结果跨运行累积,可按 `metadata.run_id` 或 `timestamp` 过滤做回归检测。

## 优化点与收益
正文未给出精度或吞吐的量化对比数字。给出的开销数字为:指标采集与 lm_eval 并发执行、不占额外 GPU 时间;相对普通 lm_eval 运行的总开销约 ~10s(服务启动 + `collect-env`)。收益主要是工程与流程层面:性能回归能在 CI 中被及时发现,精度与性能共享同一 provenance,可用 pandas 按模型分组绘制 `ttft_p95_s` 等指标随时间的变化做趋势/回归检测。

## 现状与落地
已关闭(2026-07-09)。正文中的实现进度表显示:CLI 子命令注册已完成,指标采集器(Prometheus 抓取 + 分位数)与服务管理器/runner/CLI 进行中,CI 集成、lm_eval 可选依赖声明与测试均未开始;关闭时的最终落地情况需查关联 PR。开放问题:JSONL 为便携格式,后续可加 `--upload-to` 上传到 S3/GCS 或 SQLite 后端以支持趋势查询。

## 关联
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [vLLM 文档页](https://docs.vllm.ai/en/latest/)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
