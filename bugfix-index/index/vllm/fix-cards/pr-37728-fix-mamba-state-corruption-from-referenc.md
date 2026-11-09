# PR #37728: Fix Mamba state corruption from referencing stale block table entries (#37728) (#37728)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37728](https://github.com/vllm-project/vllm/pull/37728) |
| 作者 | minosfuture |
| 合并日期 | 2026-03-24 |
| labels | ready, v1, nvidia, meta-exported, fb-exported |
| 热度 | 👍 2 · 💬 2 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
DP dummy_run 用 seq len 0 的 padding 请求间接引用已完成请求的陈旧 Mamba block,线性注意力模型输出全零 token id;请求结束即清理 block table 行修复。

## 问题现象
线性注意力(Mamba/GDN)模型出现 zero-token-id 响应(模型输出 token id 全为 0),即状态损坏导致的错误生成。触发条件:DP(data parallel)+ full CUDA graph 场景下,一个 rank 先完成 batch 而其他 rank 仍在运行,触发 `dummy_run(1)` 生成 `[1,0,0,0...]` 的 seq_len 以对齐其他 rank 的 num_decode——seq len 0 的 padding 请求间接映射到陈旧的 mamba block。这解释了为什么 TP 或 piecewise cuda graph 不触发(该 padding 只在 DP full cuda graph 需要)。验证方式:DP2 + prefix caching 压力测试,确认不再出现 NaN/zero-token-id。

## 根因
正文披露根因:使用了陈旧(stale)的 mamba block。链条为:(1) DP dummy_run 的 seq len 0 padding 请求会间接映射到 gdn_attn.py 中的陈旧 mamba block;(2) padding 到 CUDA graph captured size 本身没问题,因为 `num_reqs` 是真实请求数,gpu_model_runner.py 会正确 mask 掉其余 padded 请求的 block id;(3) 但由于 attention/utils.py 中 `num_decodes = num_reqs`,gdn_attn.py 里「mask 掉 num_decodes 之后的 block id」的逻辑成为 no-op——陈旧 block id 本来就落在 num_decodes 范围内,不会被 mask。根源在于:请求结束后其 block table 行未被清理,残留的 block id 会被后续 padding 行引用。

## 修复方案
「请求结束时清理 block table」+ dummy_run 同步:
- `vllm/v1/worker/block_table.py`:新增 `clear_row(row_idx)`,把该行已有的 block id 清零并将 `num_blocks_per_row` 置 0;`BlockTable`(多 KV 组)版本逐个转发。
  ```python
  def clear_row(self, row_idx: int) -> None:
      num_blocks = self.num_blocks_per_row[row_idx]
      if num_blocks > 0:
          self.block_table.np[row_idx, :num_blocks] = 0
      self.num_blocks_per_row[row_idx] = 0
  ```
- `vllm/v1/worker/gpu_input_batch.py::remove_request`:删除请求时调用 `self.block_table.clear_row(req_index)`。
- `vllm/v1/worker/gpu_model_runner.py::_dummy_run`:在构建 attention metadata 前补 `self.input_batch.block_table.commit_block_table(num_reqs_padded)`,把被清零的行 CPU→GPU 同步,使 attention metadata builder 看到清理后的 block table(否则清零不生效,陈旧 block id 仍会破坏 Mamba 状态)。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:所有使用 `BlockTable` 的模型的请求移除路径:`remove_request` 现在会额外清零该请求槽位的 block table 行(CPU 侧);dummy_run(FULL CUDA graph 路径)新增一次 `commit_block_table` 的 CPU→GPU 拷贝,带轻微开销;对非 Mamba/线性注意力模型,被引用的 block id 语义同样变为「已结束请求的行不再残留旧值」,属于收紧而非放宽。

## 关联
正文未列出(正文仅含 Differential Revision: D97354760,无 GitHub issue/PR 链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
