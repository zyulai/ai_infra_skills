# PR #45487: [Bugfix][DCP] Fix illegal memory access in DCP a2a decode under full CUDA graphs

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45487](https://github.com/vllm-project/vllm/pull/45487) |
| 作者 | majunze2001 |
| 合并日期 | 2026-06-14 |
| labels | bug, nvidia, ready, v1 |
| 热度 | 👍 2 · 💬 4 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | crash |

## 一句话结论
DCP a2a 的 send/recv buffer 取自可增长的共享 WorkspaceManager,full CUDA graph 捕获的地址在捕获后 warmup 中被 realloc 释放,首个请求 replay 即 IMA;改为 `torch.empty` 分配进 graph 私有内存池。

## 问题现象
`--dcp-comm-backend a2a` 且 `cudagraph_mode=FULL_AND_PIECEWISE` 时,DCP4 跑 `nvidia/Kimi-K2.5-NVFP4` 的第一个 `/v1/completions` 请求会使所有 worker 崩溃。复现命令(正文 Symptom):`vllm serve nvidia/Kimi-K2.5-NVFP4 -tp 4 -ep --decode-context-parallel-size 4 --kv-cache-dtype fp8 --dcp-comm-backend a2a --all2all-backend flashinfer_nvlink_one_sided --enable-ep-weight-filter --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'`。报错先是延迟出现在异步输出拷贝线程(`WorkerAsyncOutputCopy ... copy_event.synchronize()`,`torch.AcceleratorError: CUDA error: an illegal memory access was encountered`),`CUDA_LAUNCH_BLOCKING=1` 下定位到真实位置:full-graph replay(`cudagraph_manager.run_fullgraph` → `graph.replay()`)。

## 根因
正文用插桩(instrumentation)完整披露:a2a send/recv buffer 来自 `WorkspaceManager.get_simultaneous(...)`,是指向同一个可增长 buffer 的视图。FULL decode graph 在捕获时把该 buffer 地址烘焙进去;捕获时 workspace 大小只是最大捕获批次的高水位(`max_cudagraph_capture_size = min(max_num_seqs*2, 512)`,max_num_seqs=1024 时捕获上限 512)。捕获后的 warmup 以 `max_num_seqs`(1024)跑 eager decode,该 a2a 调用(唯一会在这里 realloc 的调用方,MoE/EP all-to-all 与 flashinfer_nvlink_one_sided 均不会)向 workspace 要更大 buffer,`_ensure_workspace_size` empty_cache 后在新地址重新分配,释放了所有已捕获 graph 仍指向的旧块——这一切发生在 server ready 之前。插桩时间线(rank 0):51 个 FULL graphs(B=1..512)全部捕获在 `0xffb704400000`;warmup eager B=1024 触发 `RESIZE 0xffb704400000 -> 0xffd424000000`(64.25→128.50 MB)并释放旧块;首个 B=1 请求 replay 仍解引用已释放的 `0xffb704400000` → IMA。

## 修复方案
`vllm/v1/attention/ops/dcp_alltoall.py` 的 `_dcp_a2a_send_recv_buffers` 删除 WorkspaceManager 分支,总是返回 `torch.empty` buffer,使其驻留在 CUDA graph 私有内存池中、在 graph 生命周期内地址稳定(与 `_dcp_a2a_unpack_combine` 及 AG+RS combine 路径已有的做法一致)。

```diff
-    if is_workspace_manager_initialized():
-        send_buffer, recv_buffer = current_workspace_manager().get_simultaneous(
-            (shape, dtype),
-            (shape, dtype),
-        )
-        return send_buffer, recv_buffer
-
+    # Don't use the shared WorkspaceManager here. ...
     return (
         torch.empty(shape, device=device, dtype=dtype),
         torch.empty(shape, device=device, dtype=dtype),
```

## 影响与回归风险
- 影响窗口:正文未提及具体版本/提交范围,不写。
- 回归面:仅改变 send/recv scratch 的分配来源,不影响 packing/exchange/combine 语义(`tests/distributed/test_dcp_a2a.py` 覆盖 `dcp_a2a_lse_reduce`);内存影响按正文:每步 decode 实际工作集约一个 send+recv pair(B=1 时约 128 KB,由 caching allocator 跨层复用),graph pool 为最大捕获批次预留(B=512 约 64 MB/GPU),净上限最坏约 +64 MB/GPU。验证:修复后 a2a + full cudagraphs 无 IMA(`--load-format dummy` 验证);对照组 ag_rs 后端同参数本就不崩。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
