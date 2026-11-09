# PR #24281: [torch.compile] CUDAGraph Inductor partition integration

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @BoyuanFeng |
| 合并 | 2025-09-20 |
| 分类 | CUDA Graph / Compiler \| tier: major |
| Label | documentation, performance, new-model, rocm, frontend, ready, torch.compile, v1, multi-modality, llama, qwen, deepseek |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/24281 |

## 一句话结论

接入 Inductor graph partition，Meta-Llama-3.1-8B / B200 下 TTFT 快 0.4%、TPOT 慢 2%。

## 动机

PR 依赖 pytorch/pytorch#162207（已 landed，PyTorch 2.9 可用）。正文通过测试对比说明现状：Piecewise 编译在 fx graph 层切分时产生 5 个 `num_piecewise_graphs_seen` 与 3 次 `num_backend_compilations`，而 Inductor graph partition 下为 1 与 1，两者都断言 `num_cudagraph_captured=6`。

## 优化点

新增 `use_inductor_graph_partition` 配置，命令 `vllm bench latency -O.cudagraph_mode=PIECEWISE -O.use_inductor_graph_partition=true`；与 `level=CompilationLevel.PIECEWISE`、`cudagraph_mode=CUDAGraphMode.PIECEWISE` 协同工作。命令通过 `-O` 把编译配置传入 `vllm bench latency`；`use_inductor_graph_partition` 作为新的编译选项与既有的 `cudagraph_mode`、`level` 组合，`test_custom_compile_config` 专门校验三者可同时生效。正文另给出 Start Time 与 kernel trace 截图，用于对比两种分区方式下内核排布与启动开销，结论仍是 TTFT 略快、TPOT 略慢。正文的 trace 说明该模式下 `attention+FP8Quant` 融合生效：原本位于 cudagraph 区域的第二个 `vllm::scaled_fp8_quant_kernel_strided` 被移入 `fmhaSm100Kernel`，因此 `scaled_fp8_quant_kernel` 紧随 attention 内核之前执行。

## 关键改动

```text
vllm/config/compilation.py: 新增 use_inductor_graph_partition 编译配置项
vllm/compilation/backends.py: 编译后端接入 Inductor graph partition
vllm/compilation/decorators.py: 编译装饰器适配新分区路径
vllm/attention/layer.py: attention+FP8Quant 融合支持
vllm/v1/cudagraph_dispatcher.py: CUDAGraph 分派适配
```

参数化测试片段（摘自 tests/compile/piecewise/test_simple.py 的 diff）：

```diff
+def _run_simple_model(
+    splitting_ops,
+    use_inductor_graph_partition,
+    use_inductor,
+    expected_num_piecewise_graphs_seen,
+    expected_num_piecewise_capturable_graphs_seen,
```

## 性能声明

Model: `meta-llama/Meta-Llama-3.1-8B`；Hardware: B200；对比 "With vllm Piecewise CUDAGraph backend" 与 "With inductor graph partition" 两张基准图，正文结论 "TTFT is 0.4% faster, TPOT is 2% slower"（pack 的 "perf claims detected in body" 为 "0.4% faster"）。精度与 workload 正文未注明，故 PR 未注明精度与 workload 的测试条件。

## 验证与回归风险

Test1 `test_simple_inductor_graph_partition` 检查 `use_inductor_graph_partition=True` 时 1 `num_piecewise_graphs_seen` / 1 `num_backend_compilations`，对比 `test_simple_piecewise_compile` 的 5 / 3（fx graph 层切分），两者都断言 `num_cudagraph_captured=6`；Test2 `test_custom_compile_config` 检查与 `CompilationLevel.PIECEWISE`、`CUDAGraphMode.PIECEWISE` 协同；Test3 `test_attention_quant_pattern` 检查 `attention+FP8Quant` 融合。回归风险：正文给出 TPOT 慢 2%；依赖 PyTorch 2.9 中的 pytorch#162207；其余受影响路径 PR 未注明。三个测试都落在 `tests/compile/` 下，分别覆盖分区计数、编译配置组合与注意力量化融合；正文另给出两组基准截图与一组启动耗时截图，但没有以文字列出延迟数字，只以百分比给出 TTFT 与 TPOT 的变化，且未注明精度与请求负载，因此无法从素材判断该收益在其他模型上的表现。

## 关联

正文明确依赖 pytorch/pytorch#162207（landed，PyTorch 2.9 中可用）；同分类其他 vLLM PR 无明确线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
