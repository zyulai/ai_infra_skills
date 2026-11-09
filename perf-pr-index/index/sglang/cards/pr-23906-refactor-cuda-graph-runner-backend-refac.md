# PR #23906: [Refactor] Cuda Graph Runner/Backend Refactor

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Oasis-Git |
| 合并 | 2026-06-10 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | documentation, high priority, lora, deepseek, speculative-decoding, blackwell, npu, run-ci, piecewise-cuda-graph, bypass-fastfail, run-ci-extra |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);140 files dropped |
| 链接 | https://github.com/sgl-project/sglang/pull/23906 |

## 一句话结论

把 CUDA Graph 重构为"phase runner × 可插拔 backend"两层结构:新图策略只需加 backend 子类,不再各写一份 runner。

## 动机

关联 issue #23004。此前每种 CUDA Graph 策略(full / breakable / piecewise)都要求一套自己的 runner 实现,策略间大量重复且扩展困难;重构目标是把策略机制下沉到可插拔 backend,phase runner 保持固定。

## 优化点

两层结构:(1) Phase runner 层,`PrefillCudaGraphRunner` / `DecodeCudaGraphRunner` 均继承 `BaseCudaGraphRunner`,每 phase 恰好一个 runner,对外只暴露 `can_run` / `capture` / `capture_one_shape` / `replay_prepare` / `replay` 五个调用点,策略相关行为全部委托给 `self.backend`。(2) CG backend 层,`BaseCudaGraphBackend` 及三个子类:`FullCudaGraphBackend`(整模型 `torch.cuda.CUDAGraph`)、`BreakableCudaGraphBackend`(在 `eager_on_graph` 断点处拼接多段图)、`TcPiecewiseCudaGraphBackend`(`torch.compile` reduce-overhead + FX 级 piecewise 切分,由 torch.compile 接管执行);按 phase 经 `resolve_{prefill,decode}_backend(model_runner)` 选择,配置来自 `cuda_graph_config[phase].backend`。配套重构:torch.compile 内部 phase 标记从 `piecewise_context_manager.py` 拆出为 `compile_phase.py` 与 `tc_piecewise_cuda_graph`,全局布尔标志统一为 `is_in_tc_piecewise_cuda_graph`,同步替换 custom_all_reduce(v1/v2)、pymscclpp、parallel_state 中的判断点。NPU 侧新增 `NPUCudaGraphBackend`(`torch.npu.NPUGraph` 全图捕获,`replay_with_input_update` 用 `NPUGraph.update` 后台线程重绑 seq_lens),`NPUGraphRunner` 降级为 `DecodeCudaGraphRunner` 薄子类。遗留 CLI 参数通过 deprecation 兼容层翻译(`cuda_graph_max_bs` → `cuda_graph_max_bs_decode`,`--disable-piecewise-cuda-graph` → `cuda_graph_backend_prefill="disabled"`)。

## 关键改动

```text
python/sglang/srt/compilation/piecewise_context_manager.py: 删除,状态拆分至 compile_phase 与 tc_piecewise
python/sglang/srt/compilation/compile_phase.py: 新文件,torch.compile warmup/capture-stream phase 标记
python/sglang/srt/compilation/torch_compile_decoration.py: 新文件,decode-Full 路径 torch.compile decoration
python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py: 新文件,Ascend NPU 全图 backend
python/sglang/srt/model_executor/runner_backend_utils/tc_piecewise_cuda_graph.py: tc_piecewise 全局状态新家(被分布式/通信模块引用)
```

## 性能声明

PR 正文未提供 benchmark 数字(Accuracy Tests / Speed Tests 章节留空),PR 未注明测试条件。重构定位为架构性改动,收益在扩展性而非单点性能。

## 验证与回归风险

PR 页面 CI 状态显示 Base/Extra 测试仍在排队(hourglass)。风险:涉及约 140+ 文件的大范围重构,触碰 custom all-reduce、pymscclpp、parallel_state、NPU graph runner、kv_canary 等广泛路径;`Full`/`Breakable`/`TcPiecewise` 三种 backend 及各硬件后端(NPU/MLX stub)行为等价性依赖既有测试兜底;遗留 flag 兼容层有 deprecation warning,未来版本将移除。

## 关联

关联 issue #23004(正文);同区域(卡片库中):#17613(piecewise CUDA graph 重构前作)、#10062(piecewise CUDA graph 引入)、#22128(piecewise + 投机解码)、#38943(DCP breakable prefill CUDA graph)、#37810(ROCm breakable CUDA graph)。
