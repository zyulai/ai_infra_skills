# PR #50327: [ModelRunnerV2] Fix scalar Mamba state update with int32 mappings

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#50327](https://github.com/vllm-project/vllm/pull/50327) |
| 作者 | shenoyvvarun |
| 合并日期 | 2026-08-03 |
| labels | ready, v1, mrv2 |
| 热度 | 👍 6 · 💬 2 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
ModelRunnerV2 下 PP 混合 Mamba 模型非末 rank 用 int32 的 idx_mapping 调 index_fill_ 报 dtype 错致 worker 崩溃;新增专用 Triton scalar-fill 内核修复。

## 问题现象
pipeline-parallel 的 hybrid Mamba 模型在 Model Runner V2 下,当非最后一个 pipeline rank 处理请求时 worker 直接崩溃。日志(照抄关键帧):

```text
(Worker_PP0_TP0 pid=1777) ERROR [multiproc_executor.py:1018] WorkerProc hit an exception.
...
  File ".../v1/worker/gpu/model_states/mamba_hybrid.py", line 314, in postprocess_state
    self.num_accepted_tokens_gpu.index_fill_(
IndexError: index_fill_(): Expected dtype int64 for index.
```

调用栈为 `sample_tokens` → `model_state.postprocess_state(input_batch.idx_mapping, 0)`。作者检索了 GitHub issues 与开放 PR,未找到相同报错的已有报告或修复。

## 根因
正文披露:`InputBatch.idx_mapping` 是有意设计的 `int32` CUDA tensor,而 `MambaHybridModelState.postprocess_state` 的标量分支(num_sampled 为 int 时)把它直接传给了 `Tensor.index_fill_`,该 API 只接受 `int64` 索引,于是抛出 `IndexError`。

## 修复方案
改动 `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`:为标量分支新增专用 Triton 内核 `_fill_num_accepted_kernel`,直接消费 int32 mapping、跳过 -1 哨兵项,并避免每步分配转换后的 index tensor;同时把 `num_reqs` 提前计算并对空 batch 提前返回。新增 CUDA 回归测试 `tests/v1/worker/test_mamba_hybrid_model_state.py`,覆盖 pipeline-prefill 值(`num_sampled=0`,归一为 1)与普通采样值两个参数化用例。内核关键摘录:

```python
@triton.jit
def _fill_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_accepted_ptr,  # [max_num_reqs]
    num_sampled,
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    tl.store(num_accepted_ptr + req_state_idx, num_sampled)
```

调用侧由 `self.num_accepted_tokens_gpu.index_fill_(0, idx_mapping, max(num_sampled, 1))` 换成 `_fill_num_accepted_kernel[(num_reqs,)](...)`。

## 影响与回归风险
- 影响窗口:正文未给出受影响的版本/提交范围。
- 回归面:仅 `mamba_hybrid.py` 的 `postprocess_state` 标量(int)分支及其空 batch 早退、对齐分支的 `num_reqs` 复用;向量分支(`_scatter_num_accepted_kernel`)逻辑等价重排。空 batch 时不再进入对齐调用(`run_fused_postprocess_align`),对正常带请求路径无影响。测试在 4 个 8xH100 RDMA 节点(PP=4, TP=8)上用 MoonshotAI/Kimi-K3 完成端到端验证,多路 chat 请求与 11/11 冒烟检查通过。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
