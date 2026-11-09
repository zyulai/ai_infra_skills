# PR #11381: perf: optimize qwen-vl with symm mem allreduce

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-10-10 |
| 分类 | Parallelism / Communication(major) |
| Label | run-ci |
| Diff | diffstat 缺失(素材未提供统计);diff 实际涉及 6 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/11381 |

## 一句话结论

VL 模型 in-place all-reduce 接入 torch symm mem,Qwen2.5-VL E2E 输入吞吐 +2%,单次 AR 446.9→209.1 µs。

## 动机

VL 模型推理中 All-Reduce 占比高(profiling 图示);Qwen2.5-VL-72B 使用 in-place all-reduce,默认走 NCCL AR,效率低。torch symmetric memory 提供更低的 AR 延迟,但 `parallel_state._all_reduce_in_place` 未接入 symm_mem communicator,且 symm mem 的尺寸门槛表对 TP4(32 MiB)偏保守,大 hidden 的 VL 模型上不去。

## 优化点

主干是把 in-place AR 路由到 symm mem:`_all_reduce_in_place` 优先级改为 pynccl → symm_mem_comm → torch.distributed 兜底;同时放宽 `SYMM_MEM_ALL_REDUCE_MAX_SIZES`(SM90: TP4 32→64 MiB、TP6/8 64→128 MiB;SM100: TP4 32→64 MiB),让 VL 常见的大 hidden 张量能命中 symm mem 路径(服务端需 `--enable-torch-symm-mem`)。diff 中还包含几项配套/搭车改动:qwen2.py 引入 `LayerCommunicator` 与 `should_allreduce_fusion`(MLP `down_proj` 支持 `skip_all_reduce`,为 AR 融合预留)、`rotary_embedding.py` 新增 `mrope_interleaved` 支持(`apply_interleaved_rope`,THTHW... 交错频率布局)、`schedule_batch.py` 的 `seq_lens_sum` 支持 numpy list 求和。PR 正文 Modifications 章节为空,未逐项说明。

## 关键改动

```text
python/sglang/srt/distributed/parallel_state.py: in-place AR 接入 symm_mem_comm
python/sglang/srt/distributed/device_communicators/all_reduce_utils.py: 放宽 symm mem 尺寸门槛
python/sglang/srt/models/qwen2.py: LayerCommunicator 接入 + MLP skip_all_reduce(AR 融合钩子)
python/sglang/srt/layers/rotary_embedding.py: mrope_interleaved 支持
python/sglang/srt/managers/schedule_batch.py: seq_lens_sum 兼容非 tensor
```

```python
def _all_reduce_in_place(self, input_: torch.Tensor) -> None:
    pynccl_comm = self.pynccl_comm
    symm_mem_comm = self.symm_mem_comm
    if pynccl_comm is not None and not pynccl_comm.disabled:
        pynccl_comm.all_reduce(input_)
    elif symm_mem_comm is not None and not symm_mem_comm.disabled:
        symm_mem_comm.all_reduce(input_)
    else:
        torch.distributed.all_reduce(input_, group=self.device_group)
```

## 性能声明

单次 AR 延迟:446.9 µs → 209.1 µs(条件 PR 未注明,配图给出)。

E2E bench_serving,`--dataset-name random-image`(360p),`--random-output-len 1`,768 prompts,max-concurrency 16,TP4,`--enable-torch-symm-mem`(示例命令为 Qwen2.5-VL-7B-Instruct;GPU 型号 PR 未注明):

| 指标 | Main | PR |
| --- | ---: | ---: |
| Input token throughput (tok/s) | 6085.89 | 6171.31 |
| Benchmark duration (s) | 69.40 | 68.52 |
| Mean TTFT (ms) | 1268.69 | 1220.77 |

作者结论:Qwen2.5-VL 提升 2% E2E 输入吞吐;Qwen3-VL 提升 3.9%。表中实测样本约 +1.4%。

## 验证与回归风险

PR 正文 Accuracy Tests / Benchmarking / Modifications 章节均为空,未提供准确性验证。回归风险:in-place AR 优先级变化影响所有走 `_all_reduce_in_place` 的模型(不止 VL),symm mem 尺寸门槛放宽使更多张量走该路径;qwen2.py 的 LayerCommunicator 改写引入 `should_allreduce_fusion` 分支与 scatter modes,改变 Qwen2 系列的通信编排;`mrope_interleaved` 默认 False,仅 rope_scaling 声明的模型受影响。

## 关联

无(正文未提及其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
