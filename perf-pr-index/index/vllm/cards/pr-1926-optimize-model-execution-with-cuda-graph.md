# PR #1926: Optimize model execution with CUDA graph

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @WoosukKwon |
| 合并 | 2023-12-17 |
| 分类 | CUDA Graph / Compiler（tier: major） |
| Label | (none) |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾注明 "18 file(s) dropped from diff for size (see PR for full diff)" |
| 链接 | https://github.com/vllm-project/vllm/pull/1926 |

## 一句话结论

用 CUDA graph 优化模型执行的 CPU 开销，正文称对小模型与 tensor parallelism 尤其有效；PR 未给出定量性能数据。

## 动机

正文仅写：本 PR 使用 CUDA graph 优化模型执行中的 CPU 开销，对小模型以及使用 tensor parallelism 时尤其有效。正文未展开说明改动前的具体瓶颈或测量方式；另致谢 @scv119 与 @Yard1 处理 NCCL 相关问题。

## 优化点

PR 正文未展开机制细节。从 diff 可见：在 `vllm/config.py` 的 ModelConfig 新增 `enforce_eager`（True 时禁用 CUDA graph、总以 eager 执行；False 时 CUDA graph 与 eager 混合执行）与 `max_context_len_to_capture`（超过该 context length 的序列回落到 eager），并新增 `_verify_cuda_graph()` 把该值钳到 `max_model_len`；`vllm/engine/arg_utils.py` 暴露 `--enforce-eager`、`--max-context-len-to-capture`（默认 8192）并透传；`vllm/engine/llm_engine.py` 在 `_init_cache` 之后调用 `warm_up_model`，注释写明 warm up 包含在 `enforce_eager` 为 False 时把模型 capture 进 CUDA graph；Ray 初始化传 `cupy_port=get_open_port()`，`ray_utils.py` 改用共享的 `get_open_port`；另对 `attention.py`、`input_metadata.py` 与多个 `models/*.py` 做适配（pack 因体积丢弃 18 个文件，未展示具体内容）。

## 关键改动

```text
vllm/config.py: 新增 enforce_eager / max_context_len_to_capture 字段与 _verify_cuda_graph 校验
vllm/engine/arg_utils.py: 新增 --enforce-eager / --max-context-len-to-capture CLI 参数并透传
vllm/engine/llm_engine.py: _init_cache 后调用 warm_up_model（含 CUDA graph capture），Ray 路径传 cupy_port
vllm/engine/ray_utils.py / vllm/entrypoints/llm.py: get_open_port 等适配
```

```diff
+        enforce_eager: Whether to enforce eager execution. If True, we will
+            disable CUDA graph and always execute the model in eager mode.
+            If False, we will use CUDA graph and eager execution in hybrid.
+        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
+            When a sequence has context length larger than this, we fall back
+            to eager mode.
```

```diff
+        # Warm up the model. This includes capturing the model into CUDA graph
+        # if enforce_eager is False.
+        self._run_workers("warm_up_model")
```

## 性能声明

PR 正文未给出定量性能数据（perf claims 段标注 "(none detected by regex)"）。正文仅定性说明："This PR uses CUDA graph to optimize the CPU overheads in model execution. This is particularly effective for small models and when using tensor parallelism."；PR 未注明测试条件。

## 验证与回归风险

PR 正文未注明引入/依赖的测试；diff 可见部分未包含测试文件改动（pack 丢弃 18 个文件，未能确认其中是否含测试）。可能影响的路径：从 diff 看涉及配置/CLI、engine warm up、Ray 端口分配、attention 与多个模型执行文件；正文致谢 @scv119、@Yard1 处理 NCCL 相关问题，提示 tensor parallelism / NCCL 路径曾需适配。PR 未注明具体回归风险。

## 关联

无（PR 正文未给出其他 PR 编号或前序/后续线索；@scv119、@Yard1 为致谢的个人，非 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
