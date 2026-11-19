# PR #63294: [EC Connector] Add ECTieringConnector filesystem tiering for encoder cache

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Akine-Ko |
| 合并 | 2026-11-19 |
| 分类 | KV Cache / Memory（tier: major） |
| Label | documentation, performance, ready, ci/build, cpu, scheduler |
| Diff | pack 标注 +None/-None, None files；pack 末尾说明「17 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/63294 |

## 一句话结论

为原生 encoder-cache offloading 增加文件系统分层（CPU 为主层、文件系统为次层），在 22.416 GiB encoder-cache 工作集超出 8 GiB CPU 缓存的 H800 回放负载下，replay mean TTFT 降 34.41%、请求吞吐 +25.84%。

## 动机

既有 ECCPUConnector 只支持 GPU→CPU 的 encoder 输出 offload；当 encoder-cache 工作集超过 CPU 缓存容量时，只能重新计算（recompute）。本 PR 把 CPU 内存作为主层，将 encoder 输出异步持久化到文件系统存储，工作集超限后可从文件系统提升（promote）回 CPU 复用，且服务重启后仍可复用兼容文件，避免代价高昂的 vision encoder 重算。

## 优化点

- **异步 write-through 与 promotion**：encoder 输出先走既有 ECCPU GPU→CPU 路径；CPU entry 就绪后被 pin 住，完整分配块异步写入 `.ec` 文件（临时文件 + 发布），完成后释放写 pin。
- **请求级 prefetch**：请求到达（on_new_request / 入队）时发起 CPU 缓存检查与文件系统 prefetch，shared prefetch pins 保护仍被引用的 CPU entry；调度准入时 GPU-cache hit 不等低层、就绪 CPU entry 走 ECCPU CPU→GPU load、到达时 prefetch 未能启动则在准入时重试准备，读提交前 CPU 分配失败允许 cache miss 并重新计算。
- **Lookup 与持久化**：稳定 namespace 用模型身份、revision、dtype、encoder-output 维度与文件格式版本标识兼容文件；启动恢复重建索引与容量账目；LRU 驱逐管理文件系统容量；兼容实例可发现并复用同目录文件（容量账目与 active-read 保护为实例本地）。
- **Backpressure**：复用原生 KV offloading 的 `EMABackpressureDetector`（默认 high_water_s 0.010 / low_water_s 0.002，seconds/MiB），按文件系统压力自适应写准入；`n_read_threads` 默认 4、`n_write_threads` 默认 2、fsync 默认 false。
- **基类扩展**：`ECConnectorBase` 新增 `on_new_request` 与 `bind_encoder_cache_query` 两个调度侧钩子；工厂注册 `ECTieringConnector`。

## 关键改动

```text
vllm/distributed/ec_transfer/ec_connector/tiering/（新包）:
  connector.py: class ECTieringConnector(ECCPUConnector)  # CPU-primary connector with optional asynchronous secondary tiers
    - ensure_cache_available(): 用 _encoder_cache_query 收集本地缓存 hash 后转交 ECTieringScheduler
  config.py: ECTieringConfig / ECFileSystemTierConfig / ECFileSystemBackpressureConfig
    （要求 ec_role='ec_both'，最多一个 secondary tier，type 必须 "fs"，root_dir 必须绝对路径）
  executor.py: ECFileIOExecutor —— DualQueueThreadPool 上每任务一个完整 .ec 文件传输，
    ECFileIOCompletion 带 submit/queue/transfer 时间, shutdown 先 drain 再拒新任务, 提交失败回滚不跑 IO
  filesystem.py: ECFileSystemTierManager（容量受限 LRU 索引）、ECTemporaryFileOwner（flock 实例租约 + 孤儿临时文件清理）、
    ECFileEntry / ECReadResult / ECWriteResult
  layout.py / lookup.py / scheduler.py / io.py 等（pack 截断未展示）
vllm/distributed/ec_transfer/ec_connector/base.py:
  + def on_new_request(self, request): ...
  + def bind_encoder_cache_query(self, query): ...
vllm/distributed/ec_transfer/ec_connector/factory.py:
  + ECConnectorFactory.register_connector("ECTieringConnector",
  +     "vllm.distributed.ec_transfer.ec_connector.tiering.connector", "ECTieringConnector")
tests/v1/ec_connector/unit/tiering/test_executor.py（新, 209 行）: 读写删 round-trip、IO 失败单次上报、
  shutdown 语义、提交错误回滚、completion 计时取 executor 而非 pool wrapper
```

## 性能声明

- **基准配置**：Qwen3-VL-8B-Instruct（BF16，TP=1）；NVIDIA H800；8 GiB CPU + 128 GiB 文件系统缓存，encoder-cache 工作集 22.416 GiB；max-model-len=8192、GPU memory utilization=0.8；8 进程客户端、总并发 48；负载为 32 张独立 warmup 图像 + 1,024 个 fill 请求 + 3 轮同图回放，每请求 100 output tokens；prefix caching 关闭、文件系统 backpressure 关闭、fsync 关闭。
- **Combined replay 结果**（每配置 3,072 个 replay 请求；百分位由合并请求样本计算，吞吐按合并请求窗口时长）：mean TTFT 1102.80 / 1080.68 / 723.30 ms（no offloading / CPU-only / Tiering；Tiering vs no offloading −34.41%）；p50 TTFT −35.41%（1102.94 → 712.34 ms）；p95 TTFT −31.61%（1657.96 → 1133.85 ms）；mean TPOT −9.83%（14.13 → 12.74 ms/token）；mean end-to-end latency −20.67%（2501.31 → 1984.40 ms）；request throughput +25.84%（18.84 → 23.71 req/s）。
- **Cache-path 验证**：三配置共 12,384 个请求（含 warmup）全部成功；每次 tiering replay 记录 1,024 次成功文件系统读、1,024 次 CPU-load 选择、0 次重算选择、0 次文件系统读写失败。

## 验证与回归风险

单元测试 `tests/v1/ec_connector/unit/tiering/test_executor.py`（`pytestmark = pytest.mark.cpu_test`）覆盖：write→read→delete 往返每个 job 恰好一次完成回调、IO 失败单次上报（含 FileNotFoundError 错误串）、shutdown 先 drain 后拒绝新提交、提交异常回滚且不执行 IO（read/write/delete × 提交前/后失败四种）、completion 计时使用 executor 自身时间而非 pool wrapper（防止 999.0 之类包装值穿透）。回归风险：`base.py` 新增两个可覆写钩子改变所有 EC connector 的调度侧接口面；`ECTieringConnector` 继承 ECCPUConnector，写路径新增 pin/unpin 与异步落盘，`.ec` 文件格式由 `EC_FILE_FORMAT_VERSION` 约束；多实例共享同一 root_dir 时容量账目与 active-read 保护仅实例本地，跨实例一致性依赖文件锁（ECTemporaryFileOwner 的 flock 与孤儿清理）；backpressure 默认开启而 benchmark 关闭，生产默认行为下的性能未在正文给出；要求 Model Runner V2（`VLLM_USE_V2_MODEL_RUNNER=1`）与 `ec_role='ec_both'`。

## 关联

- 复用 `vllm.v1.kv_offload.tiering.fs.thread_pool.DualQueueThreadPool` 与 `EMABackpressureDetector`（原生 KV offloading 分层基础设施）。
- 相邻配置：ECCPUConnector、ECMooncakeConnector（factory 中并列注册）。
- pack 未提供其他关联 PR/issue 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
