# PR #66375: [Perf][Bugfix][RL] Fix sparse NCCL update OOM and improve performance

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#66375](https://github.com/vllm-project/vllm/pull/66375) |
| 作者 | ShuoleiWang |
| 合并日期 | 2026-11-14 |
| labels | bug, documentation, ready |
| 热度 | 👍 0 · 💬 11 |
| 分类 | Parallelism / Communication |
| bug 类型 | resource |

## 一句话结论
sparse NCCL 更新不分批,patch 全部滞留 GPU 导致 OOM、非连续张量发错字节;改为按 512 MiB 分批收发并逐批 apply。

## 问题现象
main 上 `sparse_nccl` 三个问题:(1) 内存——worker 在最后一个 patch 到齐前把每个 patch 都留在 GPU 上,无论 TP size 每个被改元素占 6 字节;默认配置下可扩展 KV cache 只剩约 3 GiB 空闲,Qwen3-30B-A3B TP4 一次性 `send_weights` 在 1.25% 元素被改时失败,且失败发生在部分权重已写入之后;(2) 时间——每个张量两次分配、两次 broadcast(该模型一次更新 37,734 次),更新末尾 `empty_cache` 释放数千个缓存块,3% 改动时这些占 4.3 s 更新中的 2.2 s;(3) 非连续 patch 张量按原样 broadcast,发送错误字节。验证(4×H20 TP4 + 第五张 H20 上的 trainer,Triton MoE,main `cb59326ae` wheel):1.25% 时 main `OutOfMemoryError` while staging,本 PR 一路通过到 50%(85.3 GiB),峰值内存稳定在 95.6 GiB 中的 94.4-95.6;批处理把单引擎更新时间在 0.1%/1%/3% 改动下分别降 21%/33%/50%(约 2.1 s),broadcast 次数从 37,734 降到 2-22,接收从 0.54/0.70/1.28 s 降到 0.05-0.06 s,`empty_cache` 降到 10 ms 以下;HTTP 负载下 pause 从 5.04/5.55/7.24 s 降到 4.41/4.53/4.87 s,3% 时额外 GPU 内存从 6.65 降到 2.02 GiB;Qwen3.5-35B-A3B 上 main 在 3% 接收时 OOM 且 trainer 卡死在 NCCL,本 PR 通过 20% 和 30%。`pytest tests/distributed/test_weight_transfer.py -k sparse`:23 passed, 1 skipped;三个 batching 用例在 main 上失败,新代码的 16 个手写 mutant 全部被新测试抓住。

## 根因
正文披露三点:worker 收完全部 patch 才 apply,patch 内存上界是整个更新而非批次;每 tensor 两次分配 + 两次 broadcast,末尾留下大量待 `empty_cache` 的缓存块;非连续 patch 张量直接上 wire 导致字节错位。

## 修复方案
`vllm/distributed/weight_transfer/sparse_nccl_engine.py`:新增 `_WIRE_BATCH_BYTES = 512 << 20`;`SparseNCCLWeightTransferUpdateInfo` 增加必填字段 `wire_batch_bytes` 与 `batches()` 方法——两端基于同一份 update 元数据在相同 patch 处切批,超过上限的 patch 独自成批:
```python
def batches(self) -> Iterator[slice]:
    start = 0
    batch_bytes = 0
    for i, (dtype_name, num_updates) in enumerate(
        zip(self.dtype_names, self.num_updates_list, strict=True)
    ):
        patch_bytes = num_updates * (
            torch.int32.itemsize + getattr(torch, dtype_name).itemsize
        )
        if i > start and batch_bytes + patch_bytes > self.wire_batch_bytes:
            yield slice(start, i)
            start = i
            batch_bytes = 0
```
trainer `_broadcast_chunk` 按 batch 先 `torch.cat` 全部 indices 一次 broadcast,再按 dtype 首次出现顺序每 dtype 一次 broadcast values,发送前 concat 保证 wire 恒为连续字节;worker `receive_weights` 每批收进一个 indices buffer 与每 dtype 一个 values buffer,`split` 成 per-patch view 交给 `load_checkpoint_weight_patches`,一批 apply 完才收下一批;某批失败则继续收完剩余 broadcast(避免 trainer 挂死),`traceback.clear_frames` 后在末尾统一 raise。测试与文档同步更新。

## 影响与回归风险
- 影响窗口:正文未给出统一窗口(问题存在于 main 的 sparse_nccl backend)。
- 回归面:wire format 变化,trainer 与 worker 必须同版本——`wire_batch_bytes` 为必填,不匹配的 worker 会拒绝更新而不是收错布局,但 trainer 可能已阻塞在 broadcast;验证改为 per batch,后面的非法批次在前面的批次已落盘后才失败,worker 保持 incomplete(文档已说明 partial failure 后需重启或 reseed);trainer 端多持一个批次(≤512 MiB 或单独的大 patch);limits:仅两种模型、TP4、H20、Triton MoE 验证,index 位置为合成数据,cross-node transport 未测量。

## 关联
正文未列出带编号的 issue/PR 链接(仅泛指一个以更窄改动修非连续 patch 的 open PR,未给编号)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
