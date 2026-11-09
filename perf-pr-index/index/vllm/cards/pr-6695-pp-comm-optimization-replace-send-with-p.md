# PR #6695: PP comm optimization: replace send with partial send + allgather

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @aurickq（代 @zhisbug 提交） |
| 合并 | 2024-08-01 |
| 分类 | Model-specific Fast Path（tier: major） |
| Label | ready |
| Diff | diffstat 缺失（素材记录为 +None/-None）；diff 含 2 个文件（parallel_state.py、worker_base.py，约 +55/-6 行） |
| 链接 | https://github.com/vllm-project/vllm/pull/6695 |

## 一句话结论

流水线并行的阶段间通信从整张量 send 改为"每 TP rank 只 send 自己的 1/N 分片 + 接收端 allgather"，Llama 3.1 405B 上吞吐提升 5%。

## 动机

PP 相邻 rank 间传递 `IntermediateTensors` 时，原实现把完整中间张量从发送 rank 的 TP 组中某一个 rank 发出。当 PP 组与 TP 组秩重叠（同一张 GPU 既属于 TP 又属于 PP）时，发送侧只有一个 rank 持有/发送全量数据，其余 TP rank 侧的数据没有并行利用，通信成为大模型（Llama 3.1 405B 量级）的瓶颈之一。

## 优化点

给 `send_tensor_dict`/`recv_tensor_dict` 增加 `all_gather_group` 参数，机制为 send-allgather：(1) 发送端把每个可整除的 tensor reshape 成 `(all_gather_size, -1)` 后只发送自己 rank 对应的分片（`tensor.numel() % all_gather_size == 0` 才生效，否则退回整张量 send）；(2) 接收端先 recv 该分片，再用 `all_gather_group.all_gather(tensor, dim=0)` 沿第 0 维聚合并 reshape 回原形状。调用点在 `WorkerBase.execute_model`：非首 PP rank 用 `get_pp_group().recv_tensor_dict(all_gather_group=get_tp_group())` 接收、非尾 rank 用同样方式发送——即以 TP 组作为 allgather 组，让 TP 的每个 rank 各自参与收发自己那 1/TP 份。CPU tensor 的 metadata 路径与空 tensor 跳过逻辑保持不变。

## 关键改动

```text
vllm/distributed/parallel_state.py: send/recv_tensor_dict 增加 all_gather_group 参数，发送侧只发分片、接收侧 allgather 还原
vllm/worker/worker_base.py: execute_model 的 PP 中间张量收发传入 all_gather_group=get_tp_group()
```

发送侧分片：

```python
# send-allgather: send only a slice, then do allgather.
if (all_gather_group is not None
        and tensor.numel() % all_gather_size == 0):
    tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]
```

## 性能声明

正文仅一句：Llama 3.1 405B 上观察到 **5% 吞吐提升**（"we observed 5% improvement in throughput for llama 3.1 405b"）。未注明硬件、TP/PP 配置、workload 或测量脚本，PR 未注明测试条件的其余部分。

## 验证与回归风险

素材未列出新增测试。风险面：(1) 只有 `tensor.numel()` 能被 all_gather_group.world_size 整除的张量走新路径，形状不可整除的 tensor（如某些标量/元数据）静默退回整张量 send，两种路径在同一个 dict 中混用，正确性依赖 recv 侧用相同的 `use_all_gather` 判断（recv 侧按 metadata 中记录的 shape 先 reshape 再 allgather，两侧判断一致才成立）；(2) allgather 的通信量与 send 的集合通信语义不同（多对多 vs 点对点），在 TP 组与 PP 组秩不重叠或 NVLink 拓扑不同的机器上收益可能为负，PR 未给出适用条件分析；(3) `orig_shape` 捕获于 recv 分支，非 allgather 路径不受影响。

## 关联

同区域为分布式通信优化（PP/TP comm）；素材未列出关联 PR 编号。分类 Model-specific Fast Path 下的其他大模型通信优化：素材未给出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
