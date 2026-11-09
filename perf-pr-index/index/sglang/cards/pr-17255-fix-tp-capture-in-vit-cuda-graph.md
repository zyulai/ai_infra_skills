# PR #17255: fix tp capture in vit cuda graph

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @narutolhy |
| 合并 | 2026-03-27 |
| 分类 | CUDA Graph / Compiler |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/17255 |

## 一句话结论

修复 ViT CUDA Graph 在 TP 下未包裹 capture-aware communicator 导致的 capture 错误与 replay hang。

## 动机

ViT CUDA Graph runner 开启 Tensor Parallelism 时,capture 过程可能隐式涉及 TP 通信路径(如 all-reduce 或 capture-aware communicator)。但现有实现没有把 communicator 的 capture 生命周期与 `torch.cuda.graph(...)` 对齐,TP 设置下会出现错误的 capture 行为、运行时错误或 replay 时的 hang,使 ViT 图执行在 TP 部署中不可用或不稳定。

## 优化点

在 `_create_graph` 中先取 `get_tp_group()` 的 `ca_comm`:若 TP 组提供 capture-aware communicator,则用 `ca_comm.capture()` 作为上下文与 `torch.cuda.graph(graph)` 一同包裹 capture 体,让通信器进入 capture 模式、把 TP 通信正确录进图;若无 communicator 则用 `nullcontext()` 零开销回退,保持原行为。改动严格限定在 CUDA Graph 创建路径,不影响非 TP 或非 graph 执行。

## 关键改动

```text
python/sglang/srt/multimodal/vit_cuda_graph_runner.py: _create_graph 包裹 ca_comm.capture()
```

```python
tp_group = get_tp_group()
ca_comm = tp_group.ca_comm
capture_ctx = ca_comm.capture() if ca_comm is not None else nullcontext()

with capture_ctx, torch.cuda.graph(graph):
    y = None
    deepstack_outs: List[torch.Tensor] = []
    deepstack_capture_idx = 0
```

## 性能声明

PR 未提供 benchmark 数字;属正确性/稳定性修复,使 TP 下 ViT CUDA Graph 可以正常 capture 与 replay(即让已有的图加速在该场景下可用),而非直接的新加速。

## 验证与回归风险

PR 正文称由 `test_vlm_vit_cuda_graph.py` 覆盖;checklist 中"添加单测/更新文档/提供精度与速度基准"三项未勾选。风险面:改动仅一个文件的单处 capture 上下文;无 ca_comm 的配置走 nullcontext 与原行为一致,有 ca_comm 的 TP 配置下通信器 capture 生命周期改变,replay 语义依赖 communicator 实现的 capture() 正确性。

## 关联

同区域为 VLM/ViT CUDA Graph 系列工作(如同批 #14422 的 VLM ViT Piecewise CUDA Graph)。前序/后续 PR 素材未提供。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
