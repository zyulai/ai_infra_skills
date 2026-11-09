# PR #46370: [Feature] Initial support for fault tolerant ep using scale-down

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fangyuchu |
| 合并 | 2026-10-30 |
| 分类 | Other(tier: major) |
| Label | documentation, frontend, speculative-decoding, ready, ci/build, v1, mrv2, dflash |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数);diff 有截断(20 个文件未展示) |
| 链接 | https://github.com/vllm-project/vllm/pull/46370 |

## 一句话结论

为 DP+EP MoE 的容错框架新增 scale-down 恢复:永久移除死 rank 后不重启即可继续推理,降低故障后的不可用时间。

## 动机

现有 `--enable-fault-tolerance` 框架只支持 `retry`,前提是所有 rank 都存活。当某个 DP rank 永久死亡(硬件故障、OOM kill)时,该 rank 的 expert slot 不可达,EPLB placement 表仍引用不存在的 slot,`retry` 无法恢复。此前唯一出路是整机重启,DP+EP MoE 大规模部署的恢复代价很高。

## 优化点

核心是 "slots model":EP 拓扑冻结不变,死 rank 变成空 slot。恢复流程为:在 all2all kernel 层用 `update_mask()` 掩掉死的 EP rank(`deepep_low_latency` 与 `nixl_ep` 两个后端);利用 EPLB 冗余 slot 把孤儿逻辑专家确定性重分配到存活 rank(稳定迭代顺序、bit-identical、无跨 rank 通信);只从磁盘选择性重载被重分配的专家权重;在存活 slot 上重建 Gloo DP group 后恢复推理。由于拓扑不变,恢复路径无需 model re-init、graph recompile 与 CUDA graph 重捕获。恢复由外部 orchestrator 通过 `POST /v1/fault_tolerance/apply` 显式驱动,引擎侧无自愈;幸存者通过 `all_reduce(MIN)` 投票,全票成功才提交。约束:仅外部 LB 模式、MoE DP+EP、需要支持超时的 all2all 后端、scale-down 需 EPLB `num_redundant_experts > 0`。

## 关键改动

```text
vllm/distributed/device_communicators/all2all.py: 为 DeepEP-LL 与 NixlEP manager 新增 update_mask(),在 kernel 层掩掉死 EP rank
vllm/distributed/parallel_state.py: GroupCoordinator 新增 dead_dp_ranks,DP allreduce 后本地填充死列中性值
vllm/model_executor/model_loader/base_loader.py: 新增 get_all_weights(),支撑按 checkpoint 名过滤的部分重载
vllm/entrypoints/serve/fault_tolerance/api_router.py: /v1/fault_tolerance/apply|status 端点,支持 scale_down 指令
vllm/config/fault_tolerance.py: ft_tp_barrier_required() 判定是否启用 per-step TP Gloo barrier
```

## 性能声明

无吞吐/延迟类 benchmark。PR 正文的手工验证给出恢复耗时:故障注入后恢复流程(mask 回放 + 专家重分配 + 权重重载 + group 重建)在派发后约 1s 完成;端到端延迟由故障检测主导——TP 兄弟 rank 经 gloo connection-reset 立即感知,其余幸存引擎经 nixl_ep kernel timeout 约 30s 感知。已知热路径代价:FT 开启且 TP>1 时,每 step 增加一次 loopback Gloo barrier 往返;首次 scale-down 提交后引擎不再 idle-pause,持续跑 dummy batch(额外功耗)。

## 验证与回归风险

新增 E2E 测试 `tests/v1/fault_tolerance/test_fault_tolerance_e2e.py`(gated on `has_nixl_ep()`,覆盖 retry 与 DP=4 scale-down);`tests/model_executor/model_loader/test_reload.py` 新增 deferred reload 用例;`tests/v1/engine/test_admission_control.py`、`test_engine_core.py` 相应适配。风险点:FT 开启时 DP 同步被强制切到 Gloo CPU group(`disable_nccl_for_dp_synchronization=True`);启动期创建的 gloo group 在 steady state 前不设超时(`effective_cpu_distributed_timeout`);EPLB async worker 异常处理路径改变(FT 下不再上抛);layerwise reload 引入 `defer_processing` 语义,影响非 FT 的 reload 路径;all-None 投票在死列存在时读作 0 的已知 gap。

## 关联

同区域:EPLB 相关优化(如 pr-28369、pr-29499)与 DP allreduce 路径;本 PR 是 FT 框架从 retry-only 到 scale-down 的后续扩展,PR 正文未列出具体前序/后续 PR 编号。
