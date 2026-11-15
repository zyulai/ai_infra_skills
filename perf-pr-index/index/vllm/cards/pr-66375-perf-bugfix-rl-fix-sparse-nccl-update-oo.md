# PR #66375: [Perf][Bugfix][RL] Fix sparse NCCL update OOM and improve performance

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ShuoleiWang |
| 合并 | 2026-11-14 |
| 分类 | Parallelism / Communication(tier: major) |
| Label | bug, documentation, ready |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/66375 |

## 一句话结论

`sparse_nccl` 权重更新改为按最多 512 MiB 分批发送:Qwen3-30B-A3B TP4 默认配置下 main 在 1.25% 元素变更即 OOM,本 PR 通过到最高测试的 50%;单引擎更新时间在 0.1%/1%/3% 变更密度下降 21%/33%/50%,broadcasts 从 37,734 次/更新降到 2–22 次,并顺带修复 non-contiguous patch 张量被原样广播导致发错字节的问题。

## 动机

main 上 `sparse_nccl` 的三个问题:

1. **内存**:worker 把每个 patch 都留在 GPU 上直到最后一个到达,无论 TP size 每个变更元素占 6 bytes;默认配置下 extensible KV cache 只剩约 3 GiB 空闲,Qwen3-30B-A3B TP4 一次性 `send_weights` 在 1.25% 元素变更时失败(部分权重已写入)。
2. **时间**:每个 tensor 两次 allocation、两次 broadcast(该模型一次更新 37,734 次),更新末尾的 `empty_cache` 释放数千个 cached block;3% 变更时这些占 4.3 s 更新中的 2.2 s。
3. **Non-contiguous patch 张量**被原样广播,发出错误字节。

## 优化点

- **分批 wire**:发送端与 worker 端用同一份 update metadata 的 `batches()` 在相同 patch 处切批(每批最多 `_WIRE_BATCH_BYTES = 512 MiB`,超限的单个 patch 独占一批);update info 新增 `wire_batch_bytes` 字段,版本不匹配的 worker 会拒绝更新而不是收错布局。
- **内存有界**:worker 收完一批立即 apply,再收下一批,patch 内存由 batch 大小而非 update 大小决定。
- **合并 broadcast**:每批是其全部 indices 的一次 broadcast 加按 dtype 各一次 values broadcast;worker 收进单个 buffer 再切成 per-patch views——每次更新从"每 tensor 两次"降到几次 allocation/broadcast,留给 `empty_cache` 的 cached block 也极少。
- **Non-contiguous 修复**:trainer 发送前把每批 concatenate,线上始终是连续字节。
- **失败语义**:某批 validate/load 失败时 worker 继续接收(让 trainer 的 broadcast 全部配对完成)并在结尾统一 raise,NCCL group 保持可用;validation 变为 per-batch,后面的批失败时前面的批已落地。

## 关键改动

```text
vllm/distributed/weight_transfer/sparse_nccl_engine.py: _WIRE_BATCH_BYTES=512MiB、UpdateInfo.batches() 切批、receive_weights 按批接收+apply+失败续收、trainer 侧按批 concat broadcast
tests/distributed/test_weight_transfer.py: SparseNCCLWeightTransferUpdateInfo 增加 wire_batch_bytes;新增 wire batching 与 failed-batch drain 测试
docs/training/weight_transfer/nccl.md: 文档更新分批语义
```

```python
+    def batches(self) -> Iterator[slice]:
+        """Split the patches into consecutive batches of at most `wire_batch_bytes`.
+
+        A patch larger than the limit is a batch by itself.
+        """
+        start = 0
+        batch_bytes = 0
+        for i, (dtype_name, num_updates) in enumerate(
+            zip(self.dtype_names, self.num_updates_list, strict=True)
+        ):
+            patch_bytes = num_updates * (
+                torch.int32.itemsize + getattr(torch, dtype_name).itemsize
+            )
+            if i > start and batch_bytes + patch_bytes > self.wire_batch_bytes:
+                yield slice(start, i)
+                start = i
+                batch_bytes = 0
+            batch_bytes += patch_bytes
+        yield slice(start, len(self.names))
```

## 性能声明

(4×H20 TP4 + 第 5 块 H20 上的 trainer,Triton MoE,main `cb59326ae`,torch 2.14.0+cu130;patches 位于等距排序位置)

- **内存上限**:main 在 1% 变更通过、1.25%(2.13 GiB)在 staging 时 `OutOfMemoryError`;本 PR 通过到 50%(85.3 GiB,最高测试密度),峰值内存稳定在 94.4–95.6 / 95.6 GiB。
- **单引擎更新时间**(`send_weights` 时间,6 轮中位,两臂逐轮交替):0.1% 变更 2.63 s → 2.08 s(−21%);1% 3.14 s → 2.11 s(−33%);3% 4.29 s → 2.14 s(−50%)。其中接收 0.54/0.70/1.28 s → 0.05–0.06 s,末尾 `empty_cache` 0.04/0.29/0.88 s → <10 ms。
- **Broadcast 次数**:每更新 37,734 → 2–22。
- **HTTP under load**(32 streaming clients,每臂 2 server,8 轮):pause 5.04/5.55/7.24 s → 4.41/4.53/4.87 s;3% 变更时额外 GPU 内存 6.65 → 2.02 GiB。
- **Qwen3.5-35B-A3B**(1,811 tensors,packed experts):默认配置下 main 在 3% 变更时接收 OOM 且 trainer 卡死在 NCCL;本 PR 通过 20% 与 30%;单引擎更新时间少 4%/8%/12%(该模型 tensor 少)。
- **正确性**:`/weight_checker` 对全部 1,740 个权重的 digest 与 greedy per-token logprobs 与 main 一致,零 client 错误;真实 NCCL 下混合 dtype、奇数 count、同名双 dtype、空 patch、non-contiguous tensor、五种 batch size 均正确落地。

## 验证与回归风险

- `VLLM_ALLOW_INSECURE_SERIALIZATION=1 pytest tests/distributed/test_weight_transfer.py -k sparse`:23 passed,1 skipped(no GPU);nightly Distributed Features step 跑该文件。
- 三个 batching case 在 main 上失败(main 收完全部才 apply);drain case 在 main 上也通过。新代码的 16 个手写 mutant 全部被新测试抓出。
- 风险:wire 格式变更,trainer 与 worker 必须同版本(`wire_batch_bytes` 必填,不匹配则拒绝);per-batch validation 意味着后面的批失败时前面的批已落地(worker 保持 incomplete,文档本就要求部分失败后 restart/reseed);trainer 需额外持有最多一批(≤512 MiB 或一个更大的独立 patch)。局限:仅两个模型、TP4、H20 + Triton MoE,index 位置为合成数据,未测跨节点 transport;CI 无法复现内存上限场景。

## 关联

- 无直接关联 PR;正文提及一个修复 non-contiguous patches 的 open PR(范围更窄),本 PR 覆盖同一情况;Qwen3.5-35B-A3B 验证所用的一行 GDN conv1d NaN 修复也以 open PR 形式存在。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
