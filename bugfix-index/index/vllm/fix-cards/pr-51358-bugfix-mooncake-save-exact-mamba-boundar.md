# PR #51358: [Bugfix][Mooncake] Save exact Mamba boundary states

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51358](https://github.com/vllm-project/vllm/pull/51358) |
| 作者 | ivanium |
| 合并日期 | 2026-08-29 |
| labels | bug, ready, kv-connector, scheduler, kv-cache-manager |
| 热度 | 👍 5 · 💬 12 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
Mooncake 对 Mamba align 表按位置解析块,会把坏状态存进合法前缀哈希;改为 core 精确交接边界块并钉住到全部 rank 完成。

## 问题现象
正确性 bug:在合法前缀哈希下可能持久化 null、过期/已重分配、或仍活跃的投机 Mamba 状态;后续外部前缀命中会恢复被污染的循环状态,产生乱码或失控输出。生产症状由 DSpark/EAGLE lookup 路径命中受影响的内部键暴露(无该路径时,后续合法 CoW 边界键可能胜出而掩盖破坏)。修复前基线为乱码/失控输出、llm_judge 约 0.76/0.78;修复后 0/200 乱码、llm_judge 约 0.77/0.76、外部前缀命中率约 91–98%。

## 根因
正文披露:Mooncake 正常保存路径从 worker 侧块 ID 快照按位置解析各组源块,这对 append-only 的 attention 块表成立,但 Mamba `align` 表稀疏且可变——被取代的状态位置被 null 块替换、物理块被释放;投机 scratch 块被搬到不同逻辑位置;而连接器的块 ID 镜像只增不改,看不到这些原地修改。于是按位置的 Mamba 保存可能把 `NULL_BLOCK_ID` 当物理源,或读到已释放/已重分配/仍投机的块,却用正确的前缀哈希作键;`batch_is_exist` 去重还会保住坏值、阻止之后的正确保存覆盖。

## 修复方案
- Core 为已提交的 Mamba align 边界状态(含子块 partial tail 的 CoW 目标)发出精确 `(group_id, block_id, boundary_tokens)` 交接,连接器不再按位置重建源 ID;
- Mooncake `store_mask` 对 Mamba 组恒为 False(精确交接成为唯一 Mamba 写入者),普通路径对所有组跳过 null/缺失源 ID 作纵深防御;
- 调度器回收前用 `BlockPool.touch()` 认领块,单调 handoff ID 一一对应引用释放与在途上限;worker 上报跨 rank 完成水位,全部终止后才经 `BlockPool.free_blocks()` 释放;
- 恢复的 prefill 以跟踪到的 prefill/save 边界为准(非原始 prompt 长度);投机 scratch 重定位改为 fail-closed,需 `is_block_writable()` 确认非 null、独占(ref_cnt==1)、未哈希。

```diff
+    def is_block_writable(self, block: KVCacheBlock) -> bool:
+        """Return whether a block can be mutated by its sole owner."""
+        return not block.is_null and block.ref_cnt == 1 and block.block_hash is None
```

> 注:pack 中 diff 在 40000 字符处截断,仅覆盖 coordinator/data/scheduler/worker、offloading/scheduler、block_pool、kv_cache_manager 等文件的一部分,截断部分未编造。

## 影响与回归风险
- 影响窗口:正文未写。
- 回归面:Mooncake Store 连接器保存/加载与调度器引用生命周期;仅支持 Mamba `align` 模式(配置校验拒绝非 align/`all` 布局);`partial_tail_offloads` 接口更名为 `boundary_state_offloads`;投机块重定位条件收紧;外部持久化仍为 best-effort(超在途上限或存储失败即丢弃)。

## 关联
- #50359(外部 lookup 边界校验)
- #46896、#46906(Mamba all 模式支持与保留策略)
- #51362(本 PR 含其防御性 null/缺失检查并进一步修复精确源选择等)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
