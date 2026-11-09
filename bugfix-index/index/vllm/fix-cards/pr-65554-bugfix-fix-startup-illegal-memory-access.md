# PR #65554: [Bugfix] Fix startup illegal memory access from unaligned extensible KV cache segments

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#65554](https://github.com/vllm-project/vllm/pull/65554) |
| 作者 | xsxszab |
| 合并日期 | 2026-11-06 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 13 |
| 分类 | KV Cache / Memory |
| bug 类型 | crash |

## 一句话结论

extensible KV cache 只提交 null block,FlashMLA sparse prefill 自调优时 TMA gather 遇无效索引越界读取约 100-130 KiB 触发启动崩溃;按布局分别做 granule 对齐/双 block 提交修复。

## 问题现象

使用 extensible KV cache(#56492)时,DeepSeek-V3.2 在 8x B300(EP+TP2+DP2+PP2,eager,关闭 prefix caching,`max_model_len=8192` 等)启动即崩溃:main 分支 3/3 次启动出现 illegal memory access。触发条件是 FlashInfer autotuning 在 warmup 扩展缓存之前就运行 FlashMLA——此时只提交了 null block(`model.layers.17` 为 72 KiB null block,距 view 起点仅映射 80,128 字节,小于 kernel 读取的约 128 KiB)。

## 根因

正文披露:分配器此前只提交 block 0,却暴露每层完整的保留 KV view。FlashMLA 的 SM100 sparse prefill kernel 在 TMA gather 包含无效(padding)索引时会读过所选行之外的数据——全部索引无效时约读 view 起点 100 KiB,部分无效时最多约 130 KiB。只有 null block 提交时,这些读会到达层保留 KV view 内未映射的内存。kernel 测量在 B300 与 GB300(Bnick Hill)上相互印证:全部索引有效时未检测到越界读。

## 修复方案

把 #65225 引入的段对齐扩展到无 RDMA connector 的缓存,并按布局分别处理:

- block-compact/layer-outermost:每层段对齐到内存 granule,提交 null block 即映射整个首个 granule(复用现有 RDMA padding 路径,只增长地址空间不增物理内存);
- block-outermost:各层共享一个段且 view 起点不同,分配时改为提交 2 个 block,保证每个 view 起点之后至少半 granule(测试设备上 1 MiB)被映射,远高于 FlashMLA 读取的约 130 KiB。

关键 diff(`vllm/v1/worker/extensible_kv_cache.py`):

```diff
-        self.commit(1)
+        self.commit(2 if self._is_block_outermost else 1)
```

`pad_kv_cache_segments` 增加 `registers_memory` 参数,非 block-compact 布局(如 `LHBNC`)不再强行断言而是原样返回;`gpu_worker.py` 改为无条件调用 `pad_kv_cache_segments`(对齐不再以注册内存为前提)。 neither 布局如 `LHBNC` 保持现状,强制 `LHBNC` + sparse MLA 仍可能崩溃。main 上仅应用测试改动时 6 个测试失败,完整修复后 B300 上 70 个测试全过;端到端 GSM8K(250 题)0.884(与禁用 extensible cache 的 0.880 相当)。

## 影响与回归风险

- 影响窗口:extensible KV cache(#56492)启用后的版本;正文未给出更早引入版本。
- 回归面:block-outermost 布局初始物理映射会增大(提交两个 block);层外布局 padding 增加保留地址空间(不加物理内存);逻辑 KV 容量与 block stride 保持不变。`LHBNC` 等非两类布局未修复,配 sparse MLA 仍可能 fault。测试需 SM100,在 H100 CI runner 上跳过,曾在 B300/GB300 上人工运行。

## 关联

- #56492(extensible KV cache,复现背景)
- #65225(RDMA 段对齐,本 PR 扩展其机制)
- Nick Hill 的后续提交(block-outermost 双 block,见正文)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
