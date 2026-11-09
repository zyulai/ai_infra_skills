# PR #10310: Revert "Fix flashinfer version in sgl-kernel (#10135)"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10310](https://github.com/sgl-project/sglang/pull/10310) |
| 作者 | zhyncs |
| 合并日期 | 2025-09-11 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
回退 #10135 对 flashinfer 版本与 CUDA graph KV 索引缓冲的改动,修复 test_release_memory_occupation.py。

## 问题现象
正文未描述具体现象,动机一栏仅写「fix `test_release_memory_occupation.py`」(@merrymercy),即 #10135 合入后该测试出问题,本 PR 通过整体回退来修复。

## 根因
正文未披露根因,只说明回退 commit 76a2c86b88e4949ccb8d6b845d3d14959df10794(即 #10135)以修复 test_release_memory_occupation.py。(由 diff 推断)被回退的改动有两处:一是 sgl-kernel CMakeLists 拉取的 flashinfer GIT_TAG 从 018b5518… 换成 1a85c439…;二是 flashinfer_backend 的 `init_cuda_graph_state` 中 CUDA graph KV 索引缓冲形状从 `(speculative_num_steps, max_bs * max_context_len)` 改为 `(speculative_num_steps, max_bs * topk * max_context_len)`(投机 decode 路径按 topk 扩容)。本 PR 将两者恢复原样,flashinfer 版本变更与缓冲扩容疑似与该测试失败相关。

## 修复方案
整体 revert #10135:
- `python/sglang/srt/layers/attention/flashinfer_backend.py`:`init_cuda_graph_state` 的 `cuda_graph_kv_indices` 形状恢复为 `(self.speculative_num_steps, max_bs * self.max_context_len)`,并删除 fast_decode_plan 附近添加的 TODO 注释块;
- `sgl-kernel/CMakeLists.txt`:repo-flashinfer 的 GIT_TAG 恢复为 018b551825c8e5579206e6eb9d3229fa679202b3。

```diff
         self.cuda_graph_kv_indices = torch.zeros(
-            (self.speculative_num_steps, max_bs * self.topk * self.max_context_len),
+            (self.speculative_num_steps, max_bs * self.max_context_len),
             dtype=torch.int32,
             device="cuda",
         )
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:flashinfer 后端的 CUDA graph 状态初始化(spec/decode 路径的 kv indices 缓冲大小)与 sgl-kernel 构建时拉取的 flashinfer 版本;#10135 原本要解决的问题可能随之重现。

## 关联
- sgl-project/sglang#10135(被回退的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
