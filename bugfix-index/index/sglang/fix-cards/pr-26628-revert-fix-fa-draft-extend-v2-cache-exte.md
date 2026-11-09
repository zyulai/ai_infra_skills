# PR #26628: Revert "Fix FA DRAFT_EXTEND_V2 cache extent"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#26628](https://github.com/sgl-project/sglang/pull/26628) |
| 作者 | ch-wan |
| 合并日期 | 2026-05-29 |
| labels | documentation |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
整体回滚 #26512 对 FA DRAFT_EXTEND_V2 cache extent 的修复,并重新把 FA3/FA4 相关 CUDA graph 测试标记为已知失败。

## 问题现象
正文未描述具体现象,仅声明 "Reverts sgl-project/sglang#26512";正文附带的 CI 状态显示本 PR 的 Latest PR Test (Base) 与 (Extra) 两个运行均为失败(:x:)。

## 根因
正文未披露回滚 #26512 的原因(Motivation 留空)。
(由 diff 推断)回滚恢复了 #26512 之前的逻辑:`init_forward_metadata` 与 `init_forward_metadata_replay_cuda_graph` 中 DRAFT_EXTEND_V2 的 `cache_seqlens_int32`/`max_seq_len_k`/`cu_seqlens_k` 不再按 "prefix + extend" 计算有效 KV 范围,而是直接用 `seq_lens`(即 prefix 长度)。同时被恢复的 KNOWN_FAILURES.md 条目写明该现状下的缺陷:replay 路径 `cache_seqlens_int32 = seq_lens`(仅 prefix),而 DRAFT_EXTEND_V2 的有效 KV 范围应为 prefix + extend,内核只读到 prefix 部分 KV,导致约 82% 的注意力取值错误;FA4 继承同一路径,问题相同。即 #26512 的修复方式被撤下,底层缺陷重新成为已知失败。

## 修复方案
原样回滚 #26512,共 4 个文件:
- `python/sglang/srt/layers/attention/flashattention_backend.py`:`init_forward_metadata` 恢复为
  ```diff
  -            metadata.cache_seqlens_int32 = effective_cache_seqlens.to(torch.int32)
  -            metadata.max_seq_len_k = effective_max_seq_len_k
  +            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
  +            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
  ```
  并删除 draft_extend_v2 分支的 effective extent 计算;`init_forward_metadata_replay_cuda_graph` 同样恢复为 `cache_seqlens_int32.copy_(seq_lens)` 等原始逻辑。
- `test/registered/attention/unittest/KNOWN_FAILURES.md`、`dense/README.md`:重新登记 FA3/FA4 的 DRAFT_EXTEND_V2 CUDA-graph replay 已知失败及其根因描述。
- `dense/test_fa3.py`、`dense/test_fa4.py`:为 `test_runner_mode_eagle_draft_extend_v2_cuda_graph_cases` 重新加回 `self.skipTest(...)`,注明需 `cache_seqlens = prefix + extend` 修复并指向 KNOWN_FAILURES.md §C.3。

## 影响与回归风险
- 影响窗口:正文未说明;按 revert 对象,#26512 合并之后、本 PR 之前的提交受影响。
- 回归面:FA3/FA4 后端在 DRAFT_EXTEND_V2 模式(EAGLE draft extend)下的 eager 与 CUDA graph replay 元数据路径全部回到 prefix-only 语义;两个对应单测恢复为跳过状态;若下游曾依赖 #26512 的 prefix+extend 行为,回滚后 KV 读取范围再次不正确(即已知失败文档所述 ~82% 错值问题复现),正确修复需另行重做。

## 关联
- https://github.com/sgl-project/sglang/pull/26512 (被 revert 的 PR)
- CI 运行:https://github.com/sgl-project/sglang/actions/runs/26612849939 、https://github.com/sgl-project/sglang/actions/runs/26612849848 (正文 CI 状态链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
