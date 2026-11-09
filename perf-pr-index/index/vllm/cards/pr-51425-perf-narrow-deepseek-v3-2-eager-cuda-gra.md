# PR #51425: [Perf] Narrow DeepSeek V3.2 eager CUDA graph region

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @WoosukKwon |
| 合并 | 2026-08-07 |
| 分类 | CUDA Graph / Compiler（tier: major） |
| Label | ready, deepseek, nvidia |
| Diff | diffstat 缺失（素材记录为 +None/-None）；diff 含 2 个文件（attention.py、amd/rocm.py，约 +80/-40 行） |
| 链接 | https://github.com/vllm-project/vllm/pull/51425 |

## 一句话结论

把 DeepSeek V3.2 portable attention 的 eager 边界从整个 `_fused_attention` 收窄到 sparse indexer + `forward_mqa` 一段，单请求 kernel trace 减少 333 次直启 launch，TTFT 约减半。

## 动机

portable DeepSeek-V3.2 attention 路径此前用 `eager_break_during_capture` 装饰整个 fused attention 方法，导致 query 准备、fused norm/RoPE、W_UK BMM 等本可进 graph 的工作在每一层都变成 graph 外直启。真正需要 eager 边界的只有 sparse indexer 与后端 `forward_mqa` 调用。

## 优化点

三步收窄 eager 区域：(1) 将 `_fused_attention` 中 graph-safe 的部分（forward context/metadata 解析、indexer Q/K 投影与量化、query 准备）内联进 `forward`；(2) `eager_break_during_capture` 装饰器移到新建的 `_sparse_indexer_and_attn`，只覆盖 `sparse_attn_indexer` 与 `impl.forward_mqa`；(3) UV projection（`attn_out` view/transpose 后的 UV BMM）暂留 eager 区——是为了直接消费后端持有的 attention output 避免 extra copy，代码注释注明等 `forward_mqa` 支持 `out` 参数后可移回 captured 区。ROCm 子类保留原有宽 eager 路径（为其新增独立 `forward` override）。

## 关键改动

```text
vllm/models/deepseek_v32/attention.py: eager_break_during_capture 从 _fused_attention 移到新 _sparse_indexer_and_attn；graph-safe 部分内联进 forward
vllm/models/deepseek_v32/amd/rocm.py: ROCm 子类新增 forward override，保留原宽 eager 行为
```

边界调整后（UV 段留 eager 的注释）：

```python
# NOTE(woosuk): While the below does not need to be in the eager region,
# we put it here to avoid copying the attention output. Move this back to the
# captured region once forward_mqa supports `out` argument.
x = attn_out.view(
    num_actual, self.num_local_heads, self.kv_lora_rank
).transpose(0, 1)
```

## 性能声明

kernel trace（单请求）：333 fewer direct launches，保留 79 个 CUDA graph launch；移入 graph 的 launch 为 78 个 fused norm/RoPE kernel、78 个 W_UK BMM、99 个 query/indexer GEMM、78 个 fused-Q kernel。TTFT 基准（GLM-5.2-NVFP4 dummy weights、Rust frontend、TP4/EP4、FP8 KV cache、prefix caching 关、concurrency 1，256 个顺序请求、8-token prompt + 1 output token、8 次 warmup）：main 均值 57.42 ms / 中位 56.99 ms / p99 60.89 ms、17.39 req/s → 本卡 28.72 ms / 28.65 ms / 30.76 ms、34.74 req/s（吞吐约 2 倍）。另有明确 PIECEWISE-only 对照：复制 attention output 并捕获 UV BMM 的变体比 no-copy 形式再快 **7.43%**（26.51 ms vs 28.64 ms mean TTFT），但在默认 `FULL_AND_PIECEWISE` 的 FULL capture 阶段 OOM，故本卡保留 no-copy 形式。

## 验证与回归风险

`pytest tests/v1/cudagraph/test_breakable_cudagraph.py -q` 14 passed；两个改动文件 pre-commit 全过。作者注明不改变 attention 计算与 checkpoint 加载，无需模型质量评估。风险面：(1) eager 边界依赖 `forward_mqa` 不支持 `out` 参数这一现状，后端接口演进时需回移；(2) metadata 解析逻辑（dict/list/对象三种形态）在两处重复出现，`_sparse_indexer_and_attn` 内重新取 forward context，可能与捕获期行为耦合；(3) ROCm 路径通过子类 override 保持原行为，NVIDIA portable 路径与 ROCm 路径从此分叉，后续改动需两边同步。

## 关联

相关但不同路径：#50005（NVIDIA DCP correctness）、#47355（indexer side-stream overlap）、#47335（ROCm fused indexer-Q）。作者做了 non-duplication 检索，未见收窄 portable DeepSeek-V3.2 eager-break 边界的其他 PR。后续方向：`forward_mqa` 支持 `out` 参数后把 UV BMM 移回捕获区（可再拿 7.43%）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
