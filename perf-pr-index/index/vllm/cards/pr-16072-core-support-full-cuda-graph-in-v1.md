# PR #16072: [Core] Support full cuda graph in v1

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @chanh |
| 合并 | 2025-05-08 |
| 分类 | CUDA Graph / Compiler \| tier: major |
| Label | documentation, ready, ci/build, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/16072 |

## 一句话结论

V1 支持整模型单 CUDA graph，Qwen2.5-1.5B 中位 TPOT 5.75→5.33 ms（-7%）。

## 动机

此前 v1 使用 piecewise CUDA graph，每层多次 kernel launch、kernel 间空隙较多（profiling 下决定一个 token 约 13ms）。PR 支持对整模型 forward 捕获单个 CUDA graph，需要创建 persistent buffers 使 attention 可 graphable；正文致谢 @tlrmchlsmth 的原始实现。

## 优化点

新增 `full_cuda_graph` 编译选项与 `tests/compile/piecewise/test_full_cudagraph.py`，并在 `docs/source/design/v1/torch_compile.md` 补充设计说明。`gpu_model_runner.py` 的 `_dummy_run` 增加 `skip_attn` 参数：捕获整图时构建 `CommonAttentionMetadata` 与 `attn_metadata` 并传入 `set_forward_context`，而非传 `None`；`capture_model` 据 `full_cuda_graph` 决定 skip_attn。`vllm/v1/attention/backends/flash_attn.py` 提供 persistent buffers 使 attention 可被单图捕获。具体做法是 `_dummy_run` 在 `skip_attn=False` 时用 `self.query_start_loc[:num_reqs+1]` 与 `self.seq_lens[:num_reqs]` 构造 `CommonAttentionMetadata`，再由 `attn_metadata_builder.build` 生成 `attn_metadata` 并替换原本传给 `set_forward_context` 的 `None`，从而让整张图包含 attention。限制：仅 V1 + FA3（FA2 因 GQA 优化不可 graphable）、不支持 Cascade Attention，Llama4/local attention 仍在调查。

## 关键改动

```text
vllm/v1/worker/gpu_model_runner.py: _dummy_run 增加 skip_attn、构建 attn_metadata 并传入 set_forward_context；capture_model 据 full_cuda_graph 决定
vllm/v1/attention/backends/flash_attn.py: 提供 persistent buffers 使 attention 可 graphable
vllm/config.py: 新增 full_cuda_graph 编译选项
tests/compile/piecewise/test_full_cudagraph.py、docs/source/design/v1/torch_compile.md: 测试与文档
```

```python
+            skip_attn = not self.vllm_config.compilation_config.full_cuda_graph
             for num_tokens in reversed(self.cudagraph_batch_sizes):
                 for _ in range(self.vllm_config.compilation_config.
                                cudagraph_num_of_warmups):
-                    self._dummy_run(num_tokens)
-                self._dummy_run(num_tokens)
+                    self._dummy_run(num_tokens, skip_attn=skip_attn)
+                self._dummy_run(num_tokens, skip_attn=skip_attn)
```

## 性能声明

条件：`VLLM_FLASH_ATTN_VERSION=3 VLLM_USE_V1=1 vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-prefix-caching --dtype float16 --disable-log-requests -O3`（或 -O4）；`vllm bench serve --request-rate 1 --num-prompts 100 --random-input-len 1000 --random-output-len 100 --ignore-eos`。PR 未注明 GPU 型号。Serving Benchmark Result：Before Median TPOT 5.75 ms（Mean 5.75 / P99 6.00）、Median TTFT 28.89 ms；After Median TPOT 5.33 ms（Mean 5.31 / P99 5.56）、Median TTFT 30.47 ms；Output token throughput 96.95→96.99 tok/s、Total 1066.46→1066.92 tok/s。正文总结：对小模型（如 Qwen 2.5 1.5B）中位 TPOT 降低 7%；profiling 下决定一个 token 由 13ms（piecewise、多 launch）降到 6ms（单 launch）。

## 验证与回归风险

正文未给出 pytest 通过数，但给出 Before/After 两张 profiling 截图，分别显示 piecewise 下每层多次 launch 之间的空隙与整图单次 launch 几乎无间隙的执行；diff 中含 `tests/compile/piecewise/test_full_cudagraph.py`。回归风险：限制仅 V1 + FA3（FA2 因 GQA 优化不可 graphable）、不支持 Cascade Attention，Llama4/local attention 仍在 WIP；Median TTFT 由 28.89 升到 30.47 ms。

## 关联

无（正文仅致谢 @tlrmchlsmth 的原始实现，未给出 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
