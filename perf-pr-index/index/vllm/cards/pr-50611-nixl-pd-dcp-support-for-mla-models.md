# PR #50611: [Nixl][PD] DCP support for MLA models

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @NickLucche |
| 合并 | 2026-08-29 |
| 分类 | Overlap / Scheduler (tier: major) |
| Label | ready, kv-connector, nvidia, mrv2 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明 "13 file(s) dropped from diff" |
| 链接 | https://github.com/vllm-project/vllm/pull/50611 |

## 一句话结论

让 NIXL P/D 解耦支持 MLA 的 DCP 分片 KV，7 种 TP/DCP 组合 gsm8k strict match 均 PASS。

## 动机

本 PR 是 #38433 的替代方案（与 @pisceskkk 迭代设计），作者认为这版在代码结构与 workflow 修改上更接近 core 文件应有的样子——把 DCP 逻辑注入现有抽象而非在旁边另建一套。NIXL P/D disaggregation 原本假设 MLA 的 KV cache 在 TP 间完全复制，远端 rank 只需读一次；Decode Context Parallel (DCP) 按 block 在 rank 间分片 KV，打破了该假设，需要新的远端 rank 路由。

## 优化点

关键简化假设：每侧 `dcp_size` 要么为 1（完全复制，保持既有行为），要么为 `tp_size`（完全分片，每 rank 持有不相交的 1/dcp_size slice），绝不为部分分片（1 < dcp_size < tp_size）。这使 DCP 分片与旧的"多个 TP replica 选一个 canonical"逻辑不必在同一侧组合，路由可归约为三种扁平情形（both replicated / one-replicated-one-sharded / both-sharded）而非通用部分重叠求解。两条支撑假设：两侧 DCP size 总能互相整除（实际部署用 power-of-two/highly-composite，普通 modulo 即可，无需 gcd/lcm）；挂 KV connector 时 DCP 以 block 粒度分片（`cp_kv_cache_interleave_size` 固定为 `block_size`），作者为此更新了几个 attention backends 以取得更新后的值。示意图给出 Producer dcp_size=4（KV 分 4 片）到 Decoder dcp_size=2（每 rank 读 2 个远端 rank：D0 读 {P0,P2}、D1 读 {P1,P3}，按 `remote_rank % local_dcp_size == local_dcp_rank`）。

## 关键改动

```text
vllm/config/kv_transfer.py: 新增 has_connector()，判断是否直接配置或在 MultiConnector 中配置某 connector
vllm/config/vllm.py: 接入 DCP / cp_kv_cache_interleave_size 相关配置
vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py、pull_scheduler.py、metadata.py: DCP 分片下的远端 rank 路由
vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py、pull_worker.py、connector.py: 读写分片 KV 的 worker/connector 路径
tests/v1/worker/test_gpu_model_runner.py 与 tests/v1/worker/test_gpu_model_runner_v2.py: late interleave adjustment 后 input batch 重初始化与 cp_interleave 快照测试
```

关键代码（来自 `vllm/config/kv_transfer.py` diff）：

```python
    def has_connector(self, connector_name: str) -> bool:
        """Whether ``connector_name`` is configured, directly or in MultiConnector."""
        if self.kv_connector == connector_name:
            return True
        return self.kv_connector == "MultiConnector" and any(
            child.get("kv_connector") == connector_name
            for child in self.kv_connector_extra_config.get("connectors", [])
        )
```

## 性能声明

PR 正文未给出吞吐/延迟/加速比类数据，Sweep 给出的是 gsm8k eval 精度。条件：`gsm8k eval sweep` at `block_size=64`（128 需 #51031 的 fix）on `deepseek-ai/DeepSeek-V2-Lite-Chat`；7 种配置全部 `PASS`：P TP4/DCP1 → D TP4/DCP1 `0.6535253980288097`；P TP2/DCP1 → D TP4/DCP1 `0.6489764973464746`；P TP4/DCP4 → D TP4/DCP4 `0.6588324488248674`；P TP4/DCP1 → D TP4/DCP4 `0.6648976497346475`；P TP4/DCP4 → D TP4/DCP1 `0.6595905989385898`；P TP2/DCP2 → D TP4/DCP4 `0.6573161485974223`；P TP2/DCP1 → D TP4/DCP4 `0.6542835481425322`。PR 未注明硬件、精度或并发等更多测试条件。

## 验证与回归风险

正文给出上述 gsm8k sweep（7 种 P/D 的 TP/DCP 组合全部 PASS）与 `block_size=64` 的限制。diff 中的新测试覆盖：late interleave adjustment 后 `InputBatch` 重新初始化（`test_input_batch_reinitialized_after_late_interleave_adjustment`）、V2 runner 快照 `cp_interleave`（`test_v2_runner_snapshots_late_interleave_adjustment`）、`test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table` 增补 `parallel_config`。风险面：NIXL P/D 在 MLA + DCP 下的远端 rank 路由、`cp_kv_cache_interleave_size` 被固定为 `block_size` 后各 attention backend 读取更新值、部分分片（1 < dcp_size < tp_size）不在支持范围、PCP/其他 KV connector 组合；正文未声明 GPU 全量回归套件。

## 关联

#38433（本 PR 作为其替代方案，正文明确）、#51031（`block_size=128` 所需 fix）。均为正文明确出现的编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
