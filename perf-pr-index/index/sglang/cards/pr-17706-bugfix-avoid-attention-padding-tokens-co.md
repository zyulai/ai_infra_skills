# PR #17706: [bugfix] avoid attention padding tokens computation in pcg

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Chen-0210 |
| 合并 | 2026-04-14 |
| 分类 | Attention / Kernel(classify.py 给出,tier: major) |
| Label | documentation, quant, amd, dependencies, Multi-modal, deepseek, hicache, sgl-kernel, blackwell, npu, run-ci, deterministic, piecewise-cuda-graph, diffusion, model-gateway |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 显示 8 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/17706 |

## 一句话结论

修复 PCG(piecewise CUDA graph)下 padding token 进入 attention 计算导致 nan/输出损坏的 bug,使 FlashInfer backend 可以重新安全地开 PCG。

## 动机

PCG 开启时,attention metadata 以 real_num_tokens 初始化,但输入张量仍含 padding token;FlashInfer 等 attention backend 无法正确处理这种不一致,会出现未定义行为,包括 nan 值、输出损坏(重复的 `!!!!!`)以及异常长的输出长度。此前的修复(PR #21452)在 FlashInfer backend 内为 padding token 追加 dummy request 来满足 shape check,但这条路径脆弱——两个 scheduler 测试甚至因此专门加了 `--disable-piecewise-cuda-graph`。本 PR 换一种思路:让 attention 永远只见 real token。

## 优化点

- 回滚 PR #21452:删除 FlashInfer backend 的 dummy-request 机制(padding token 的 dummy page 预留、qo/kv_indptr 末尾追加、`bs_eff` 逻辑)以及 ForwardContext 携带 `num_tokens` 的通道,`kv_indices` 分配恢复为 `paged_kernel_lens_sum + 256`。
- 核心修复:在 `unified_attention_with_output` 与 `unified_linear_attention_with_output` 中,调用 backend 前把 `query/key/value`(及 `q_rope/k_rope`、`mixed_qkv/a/b`)按 `forward_batch.num_token_non_padded_cpu` 切片,只窄化 `forward_batch.out_cache_loc`(及其 SWA 对应字段)而不替换 ForwardBatch 对象,调用后恢复原值,使模型/backend 状态仍写回同一 batch;输出写回改为 `output[:real_num_tokens].copy_(ret)`。
- PCG runner:capture/replay 构造 ForwardBatch 时补上 `num_token_non_padded_cpu`,replay 不再需要预计算 static_num_tokens 传给 context。
- 两个 scheduler 测试移除 `--disable-piecewise-cuda-graph`,PCG 与 FlashInfer 组合恢复默认可测。

## 关键改动

```text
python/sglang/srt/layers/radix_attention.py: attention 前后按 real_num_tokens 切片/恢复
python/sglang/srt/layers/radix_linear_attention.py: linear attention 同样处理
python/sglang/srt/layers/attention/flashinfer_backend.py: 删除 #21452 的 dummy request 机制
python/sglang/srt/compilation/piecewise_context_manager.py: 移除 ForwardContext.num_tokens
python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py: 补 num_token_non_padded_cpu
```

```python
real_num_tokens = forward_batch.num_token_non_padded_cpu
query = query[:real_num_tokens]
key = key[:real_num_tokens]
value = value[:real_num_tokens]
...
forward_batch.out_cache_loc = original_out_cache_loc[:real_num_tokens]
ret = forward_batch.attn_backend.forward(query, key, value, ...)
forward_batch.out_cache_loc = original_out_cache_loc  # 调用后恢复
output[:real_num_tokens].view(ret.shape).copy_(ret)
```

## 性能声明

PR 正文未提供 benchmark 数字(Benchmarking 章节为空,注明 "Tested in CI")。定性收益:修复后 PCG 无需再为 FlashInfer 兼容性关掉(dummy request 元数据构建开销也随之移除),但 PR 未量化。

## 验证与回归风险

精度在 CI 验证;两个 registered scheduler 测试(`test_priority_scheduling.py`、`test_scheduler_control.py`)移除 PCG 禁用 flag,改为在 PCG 开启下直接跑,构成回归保护。风险:`num_token_non_padded_cpu` 成为 attention 统一入口依赖的字段,所有走 `unified_attention_with_output`/`unified_linear_attention_with_output` 的 backend(PCG 路径)共享此行为;SWA pool 的 `set_swa_loc` 恢复逻辑依赖 backend 实现该接口;删除 dummy-request 机制后,若其他 backend 依赖 ForwardContext.num_tokens 的旧行为会受影响。

## 关联

回滚对象:PR #21452(FlashInfer PCG dummy request 方案)。同区域:PCG/inductor 编译线(PR #21977 combo_kernels)、FlashInfer backend 相关 PR(PR #27689 MLA spec-decode plan 等)。PR 正文未列出其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
