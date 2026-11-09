# PR #25233: [Core] Encoder separation for Encode-Prefill-Decode Disaggregation

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fake0fan |
| 合并 | 2025-11-12 |
| 分类 | Overlap / Scheduler (tier: major) |
| Label | documentation, performance, rocm, structured-output, frontend, ready, ci/build, v1, qwen, deepseek, kv-connector |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾标注 "31 file(s) dropped from diff for size"(见 PR 完整 diff),diff 仅保留 1 个测试文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/25233 |

## 一句话结论

新增 EC connector 抽象,把多模态 vision encoder 拆到独立进程实现 E/P/D 分离;PR 正文未给出定量性能数据。

## 动机

PR 正文定义 "disaggregated encoder":把多模态 LLM 的 vision-encoder 阶段放到与 prefill / decoder 阶段独立的 vLLM 实例中运行,带来三点实际收益——独立细粒度扩缩容、更低 TTFT、encoder output 跨进程复用与缓存。该设计基于 #21740 初版的反馈重做,并 CLOSE issue #20799。正文 Next-Step Plan 把 "Comprehensive Performance Evaluation" 列为后续工作,故本 PR 未提供定量性能结果。

## 优化点

V1 中通过显式的 EC(encoder cache)connector 抽象 `ECConnectorBase` 实现,它划分 scheduler 侧与 worker 侧职责并提供生命周期钩子。Scheduler 角色:`check_caches_exist(request)` 探测每个多模态数据的 EC 是否存在、`update_state_after_alloc` 记录加载意图、`build_connector_meta` 生成每步供 worker 消费的元数据并重置内部状态、`request_finished`、`update_connector_output`。Worker 角色:`bind_connector_metadata` / `clear_connector_metadata`、`start_load_caches` 在 gather 前把 EC 载入本地 encoder cache、`save_caches`、`wait_for_save`、`get_finished`。Scheduler 在 `ec_transfer_config` 存在时经 `ECConnectorFactory.create_connector(..., role=SCHEDULER)` 初始化 connector,追踪 encoder-token budget 并在缓存已存在时跳过 EC 工作,在 `SchedulerOutput` 中下发 `ec_connector_metadata`。GPUModelRunner 的 producer 侧在 `execute_model` 中当 `is_producer` 时进入 `maybe_get_ec_connector_output` 上下文,encode 后按 `mm_hash` 调 `maybe_save_ec_to_connector` 持久化;consumer 侧在 `_gather_mm_embeddings` 前调 `start_load_caches` 从外部存储填充 `encoder_cache[mm_hash]`。参考实现 `ECSharedStorageConnector` 以共享目录按 mm_hash 存 `encoder_cache.safetensors`;PD 侧 KV 传输沿用 `NixlConnector`。相关代码位于 `vllm/distributed/ec_transfer`。

## 关键改动

```text
vllm/v1/core/sched/scheduler.py: 初始化 EC connector、encoder-token budget、分配后 update_state_after_alloc、下发 ec_connector_metadata 并回读完成状态
vllm/v1/worker/gpu_model_runner.py: producer(encode 后按 mm_hash 保存 EC、wait_for_save/get_finished)与 consumer(start_load_caches 后再 gather)两侧逻辑
vllm/distributed/ec_transfer/*: ECConnectorBase 接口、ECConnectorFactory、默认注册的 ECSharedStorageConnector
vllm/config/*: ec_transfer_config(ec_connector / ec_role / ec_connector_extra_config)配置
tests/v1/core/test_scheduler.py: diff 中唯一保留的文件,参数化 use_ec_connector / ec_role 并断言 decode 阶段 ec_connector_metadata.mm_datas 为空
```

关键代码(diff 保留的 EC connector 参数化用例):

```python
@pytest.mark.parametrize(
    "use_ec_connector, ec_role", [(False, None), (True, "ec_consumer")]
)
def test_kv_connector_unable_to_allocate(use_ec_connector, ec_role):
```

注:pack 的 diff 因超长仅保留该测试文件(31 个文件被丢弃),上述 scheduler / worker / connector 文件作用来自 PR 正文描述。

## 性能声明

PR 正文未给出定量性能数据。正文仅以动机形式列出三项收益(独立细粒度 scaling、更低 TTFT、跨进程复用与缓存 encoder outputs),并把 "Comprehensive Performance Evaluation" 列入 Next-Step Plan;未给出 benchmark 数字,也未注明模型、硬件、精度或 workload。

## 验证与回归风险

PR 正文列出 ready-to-run 示例脚本(`examples/online_serving/disaggregated_encoder/shared_storage_connector/` 下 `disagg_1e1pd_example.sh`、`disagg_1e1p1d_example.sh` 等)与最小 ECTransfer CLI 配置(producer / consumer 各一条 `vllm serve ... --ec-transfer-config`,未设 `shared_storage_path` 时默认 `/tmp`)。diff 保留的 `tests/v1/core/test_scheduler.py` 增加 EC connector 参数化用例。风险面:改动 scheduler 与 GPUModelRunner 的共享路径,涉及多模态、KV connector、结构化输出、ROCm、deepseek / qwen 等标签相关路径;正文称 EC connector 为显式新抽象、PD 部分复用已有 `NixlConnector`,未给出上述路径的回归测试详情;后续计划含拓宽 connector 类型、多硬件平台支持与全面性能评估。

## 关联

PR 正文明确:基于 #21740(初版 disaggregated encoder)的反馈重做;并 CLOSE issue #20799。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
