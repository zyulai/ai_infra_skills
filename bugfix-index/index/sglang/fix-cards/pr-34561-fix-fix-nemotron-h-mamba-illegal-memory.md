# PR #34561: [Fix] Fix Nemotron-H Mamba illegal memory access under DP attention with CUDA graph

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34561](https://github.com/sgl-project/sglang/pull/34561) |
| 作者 | elvischenv |
| 合并日期 | 2026-08-19 |
| labels | blackwell, bug, run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | crash |

## 一句话结论
Nemotron-H Mamba 在 DP attention 分支未接入 CUDA graph split-op dispatch,replay 时非法内存访问;补齐分派修复。

## 问题现象
Nemotron-H 开启 DP attention 并配合 CUDA-graph runner backend(breakable CUDA graph / torch.compile piecewise)时,在 graph replay 阶段崩溃,报 `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`。堆栈经 `prefill_cuda_graph_runner._execute_body_capture` → `nemotron_h.py forward` → `breakable_cuda_graph_backend.replay` → `torch/cuda/graphs.py replay`。作者给出了完整服务端复现命令(Nemotron-3-Ultra-550B-A55B-NVFP4,`--cuda-graph-backend-prefill breakable`、`--tp 4 --ep 4 --dp 4 --enable-dp-attention`、trtllm_mha attention backend、flashinfer mamba backend 等),修复前必崩;修复后 gsm8k 5-shot exact_match 达 0.9431/0.9424。

## 根因
正文未单独写根因分析段落,但修改说明披露了机制:`NemotronHMambaDecoderLayer.forward` 的 `is_dp_attention_enabled()` 分支此前直接调用 `self._forward_mamba`(eager 路径),没有像非 DP 路径那样把 mixer 接入 breakable/piecewise CUDA graph 已有的 split-op dispatch,导致该层在 CUDA graph 捕获/重放流程下行为不一致,graph replay 时触发非法内存访问。

## 修复方案
改 `python/sglang/srt/models/nemotron_h.py` 的 `NemotronHMambaDecoderLayer.forward`,在 `is_dp_attention_enabled()` 分支按运行环境三分派:
- `is_in_breakable_cuda_graph()` → `breakable_nemotron_mamba2_with_output(hidden_states, output, self.layer_id, False)`;
- `is_in_tc_piecewise_cuda_graph()` → `nemotron_mamba2_with_output(hidden_states, output, self.layer_id, False)`;
- 否则保持 `self._forward_mamba(hidden_states, forward_batch)`(eager 路径不变)。

DP 路径上 `fuse_mlp_allreduce` 一律传 `False`,与该路径现有行为一致(allreduce 由 layer communicator 处理)。另将 `test_nvidia_nemotron_3_super_nvfp4.py` 中移除 `--cuda-graph-backend-prefill disabled`,使该测试重新走 breakable prefill graph。

关键摘录:
```diff
-            output = self._forward_mamba(hidden_states, forward_batch)
+            if is_in_breakable_cuda_graph():
+                output = torch.empty_like(hidden_states)
+                breakable_nemotron_mamba2_with_output(
+                    hidden_states, output, self.layer_id, False
+                )
+            elif is_in_tc_piecewise_cuda_graph():
+                output = torch.empty_like(hidden_states)
+                nemotron_mamba2_with_output(hidden_states, output, self.layer_id, False)
+            else:
+                output = self._forward_mamba(hidden_states, forward_batch)
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:改动集中在 Nemotron-H Mamba 解码层的 DP attention 分支。eager(非 CUDA graph)行为不变,对现有非 graph 运行是 no-op;breakable 与 piecewise 两条 CUDA graph 路径新增 split-op 调用,`fuse_mlp_allreduce=False` 与原 DP 行为一致;测试文件改动使该场景进入 CI 覆盖。

## 关联
正文未列出关联 issue/PR。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
