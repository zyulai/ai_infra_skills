# PR #5414: [Speculative Decoding] Support draft model on different tensor-parallel size than target model

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wooyeonlee0 |
| 合并 | 2024-06-25 |
| 分类 | Speculative Decoding（tier: major） |
| Label | (none) |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/5414 |

## 一句话结论

让 spec decode 的 draft model 用比 target 更小的 TP（暂限 draft_tp=1），降低小 draft 模型通信开销。

## 动机

此前 multi-step draft workers 需与 target workers 使用相同的 TP degree。本 PR 支持 draft workers 用比 target 更小的 tp degree；`SmallerTpProposerWorker` 的 docstring 说明目标为 "This reduces the communication overhead of small draft models."。PR 明确限定目前只支持 `draft_tp=1`，将由后续 PR 扩展，并声明 Resolves #4632。

## 优化点

新增 `SmallerTpProposerWorker`（继承 `ProposerWorkerBase`）包装 `MultiStepWorker`：`maybe_wrap_worker` 在 `draft_tensor_parallel_size != target_tensor_parallel_size` 时按 `draft_ranks = list(range(draft_tensor_parallel_size))` 选出参与 draft 生成的 rank 并包装。docstring 说明按 `is_dummy` 区分参与 / 不参与 draft 生成的 worker，参与方在 draft forward 期间通过 `patch_tensor_parallel_group` 临时 patch vLLM 的 TP group。在 `parallel_state.py` 新增 `init_world_group` / `init_model_parallel_group` 接口以构造较小 TP 的 `GroupCoordinator`。`vllm/config.py`、`engine/arg_utils.py`、`benchmarks/benchmark_latency.py` 新增 `speculative_draft_tensor_parallel_size`（`-spec-draft-tp`）配置项，`spec_decode_worker.py` 接入包装逻辑。

## 关键改动

```text
vllm/spec_decode/smaller_tp_proposer_worker.py（新增 149 行）: SmallerTpProposerWorker 包装 MultiStepWorker，按 draft_ranks 选 rank、临时 patch TP group。
vllm/distributed/parallel_state.py: 新增 init_world_group / init_model_parallel_group，构造较小 TP 的 GroupCoordinator。
vllm/config.py、vllm/engine/arg_utils.py、benchmarks/benchmark_latency.py: 新增 speculative_draft_tensor_parallel_size 配置与 CLI 参数。
vllm/spec_decode/spec_decode_worker.py: 接入 maybe_wrap_worker 包装逻辑。
tests/spec_decode/e2e/test_integration_dist_tp2.py（新增）、test_integration_dist.py→test_integration_dist_tp4.py（rename）: TP2/TP4 集成与 greedy equality 测试。
```

```text
    @classmethod
    def maybe_wrap_worker(cls, worker, draft_tensor_parallel_size: int,
                          target_tensor_parallel_size: int):
        """Wrap the worker in a SmallerTpProposerWorker if necessary.
        """
        if draft_tensor_parallel_size == target_tensor_parallel_size:
            return worker

        # gpu ranks that will generate draft tokens together
        draft_ranks = list(range(draft_tensor_parallel_size))

        logger.info("Wrapping {%s} in {%s}", type(worker), cls)
        return cls(worker, draft_ranks)
```

## 性能声明

PR 正文未给出定量性能数据（无 benchmark 数字），PR 未注明测试条件。正文仅定性说明支持 draft workers 用比 target 更小的 tp degree。

## 验证与回归风险

验证：新增 `test_integration_dist_tp2.py`（`test_target_model_tp_gt_1`、`test_draft_model_tp_lt_target_model_tp2`，`tensor_parallel_size=2`、`enforce_eager=True`、`use_v2_block_manager=True`、`use_async=True`、`speculative_draft_tensor_parallel_size=1`，`run_greedy_equality_correctness_test`），把 `test_integration_dist.py` rename 为 `test_integration_dist_tp4.py`，并更新 `.buildkite/test-pipeline.yaml` 加入 TP2/TP4 用例。风险：PR 明确限定目前只支持 `draft_tp=1`，更大 draft TP 留待后续 PR；测试中 `is_hip()` 时 skip（"hip is not well-supported yet"）；涉及 `parallel_state.py` 的 TP group 临时 patch，可能影响分布式 group 状态与 NCCL 清理。

## 关联

Resolves #4632；正文称将由 "following PR" 扩展（未给编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
