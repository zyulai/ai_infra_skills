# PR #1534: Fix hanging in the scheduler caused by long prompts

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#1534](https://github.com/vllm-project/vllm/pull/1534) |
| 作者 | chenxu2048 |
| 合并日期 | 2023-11-21 |
| labels | (none) |
| 热度 | 👍 4 · 💬 2 |
| 分类 | Overlap / Scheduler |
| bug 类型 | hang |

## 一句话结论
超长 prompt 的 seq_group 永远留在 waiting 队首卡死调度器,现识别为不可分配并加入 ignored_seq_groups 跳过。

## 问题现象
调度器可能因超长 prompt 卡死:调度器在发现无法完成分配时停止从 waiting 列表加入 seq_group;若某个 seq_group 小于调度限制(max_num_batched_tokens)但又大于全部 GPU block 数,它既无法分配、又永远不会被移出 waiting 列表,从而阻塞其后所有请求。影响面为提交超过 GPU KV cache 容量的长 prompt 的场景,后续所有请求都被队首的不可分配请求堵住。

## 根因
正文披露:`BlockSpaceManager.can_allocate` 原本只返回 bool(当前空闲块够/不够),调度器对「分配不了」一律 break 等待;但不存在区分「暂时不够(LATER)」与「永远装不下(NEVER)」的逻辑——当 `num_required_blocks` 超过 GPU 总块数(而非仅当前空闲数)时,空闲块永远不会变够,seq_group 永远留在 waiting 队首,形成死等。

## 修复方案
- `vllm/core/block_manager.py`:`can_allocate` 返回类型由 `bool` 改为新增的三值枚举 `AllocStatus`(OK / LATER / NEVER);当 `num_total_gpu_blocks - num_required_blocks < watermark_blocks`(即总容量都装不下)时返回 NEVER,空闲块暂时不足返回 LATER:
```python
if (self.num_total_gpu_blocks - num_required_blocks <
        self.watermark_blocks):
    return AllocStatus.NEVER
if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
    return AllocStatus.OK
else:
    return AllocStatus.LATER
```
- `vllm/core/scheduler.py` `_schedule`:LATER 时保持原有 break 等待;NEVER 时打 warning("Input prompt (...) is too long and exceeds the capacity of block_manager"),把该 seq_group 全部 seq 置为 `FINISHED_IGNORED`,加入 ignored_seq_groups 并 `self.waiting.pop(0)` 移出队列后 continue 处理后续请求。

## 影响与回归风险
- 影响窗口:正文未说明(2023-11 合并,属 v0 时代调度器)
- 回归面:`can_allocate` 返回类型变化影响所有调用方(scheduler 调度与 swap 路径需同步适配,本 diff 中 scheduler 已适配);被判 NEVER 的请求不再排队等待而是直接被 FINISHED_IGNORED 丢弃,长 prompt 用户会收到被忽略的结果而非挂起;NEVER 判定依赖 `num_total_gpu_blocks` 与 watermark,sliding window 分支(取 min)的语义需保持一致。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
