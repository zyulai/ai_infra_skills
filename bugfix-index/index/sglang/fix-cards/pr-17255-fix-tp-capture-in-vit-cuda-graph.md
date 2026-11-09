# PR #17255: fix tp capture in vit cuda graph

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#17255](https://github.com/sgl-project/sglang/pull/17255) |
| 作者 | narutolhy |
| 合并日期 | 2026-03-27 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 27 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | general |

## 一句话结论
ViT CUDA Graph 捕获未纳入 TP capture-aware 通信器的捕获生命周期;按 `tp_group.ca_comm.capture()` 包裹 `torch.cuda.graph` 修复。

## 问题现象
在 ViT CUDA Graph runner 上开启 Tensor Parallelism 时,CUDA Graph 捕获可能隐式涉及 TP 通信路径(如 all-reduce 或 capture-aware 通信器)。由于现有实现未让通信器的捕获生命周期与 `torch.cuda.graph(...)` 对齐,可能导致捕获行为不正确、运行时报错,或在 TP 配置下 replay 时挂起(hangs)。

## 根因
正文披露:TP 场景下 ViT CUDA Graph 捕获会隐式走 TP 通信,但实现没有把(capture-aware)通信器的 capture 生命周期与 `torch.cuda.graph(...)` 对齐,从而导致捕获不正确、运行时错误或 replay 挂起。

## 修复方案
改动在 `python/sglang/srt/multimodal/vit_cuda_graph_runner.py` 的 `_create_graph`:

1. 通过 `get_tp_group()` 取当前 TP group,读取其 `ca_comm`(capture-aware 通信器)。
2. 若 `ca_comm` 存在,用 `ca_comm.capture()` 作为外层上下文与 `torch.cuda.graph(graph)` 嵌套包裹捕获;否则用 `nullcontext()` 保持原行为、零开销回退:

```diff
-        with torch.cuda.graph(graph):
+        tp_group = get_tp_group()
+        ca_comm = tp_group.ca_comm
+        capture_ctx = ca_comm.capture() if ca_comm is not None else nullcontext()
+
+        with capture_ctx, torch.cuda.graph(graph):
             y = None
```

正文说明该改动严格限定在 CUDA Graph 创建路径,不影响非 TP 或非图执行;有 capture-aware 通信器时用其 `capture()`,没有时保持原行为。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:仅影响 ViT CUDA Graph 捕获路径(`_create_graph`);无 capture-aware 通信器的环境(非 TP 或无 ca_comm)走 `nullcontext`,行为不变。TP + ViT CUDA Graph 场景的捕获顺序变为 ca_comm.capture() 在外层,若某通信器的 `capture()` 语义与预期不符需另查。精度覆盖见 `test_vlm_vit_cuda_graph.py`。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
