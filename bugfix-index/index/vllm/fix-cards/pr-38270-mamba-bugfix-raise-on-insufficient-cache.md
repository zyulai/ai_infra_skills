# PR #38270: [Mamba][Bugfix] Raise on insufficient cache blocks instead of silently capping cudagraph sizes

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38270](https://github.com/vllm-project/vllm/pull/38270) |
| 作者 | NickLucche |
| 合并日期 | 2026-03-30 |
| labels | bug, nvidia, ready, v1 |
| 热度 | 👍 1 · 💬 6 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
Mamba cache 块不足时原先静默压缩 cudagraph 尺寸导致 prefill 退化为 eager,改为直接报 ValueError 提示调大 max-num-seqs 或显存。

## 问题现象
带 Mamba 层的混合模型在 `max_num_seqs` 超过可用 Mamba cache blocks 时,启动/捕获阶段表现为:cudagraph 捕获尺寸被静默压缩,PIECEWISE(prefill)与 FULL(decode)两种 cudagraph 捕获同时被限制,prefill 被静默地以 eager 模式运行,造成严重性能回退(正文指出该性能问题最初在 #37416 中被提出)。若按本 PR 改后触发检查,则会显式报错:`ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (512). Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed. Please lower max_num_seqs to at most 512 or increase gpu_memory_utilization.`

## 根因
正文披露:由 #37416 引入的 `adjust_cudagraph_sizes_for_mamba_cache` 会静默修改共享的 `cudagraph_capture_sizes` 列表,同时限制 PIECEWISE(prefill)与 FULL(decode)的 cudagraph 捕获;但 Mamba 约束只适用于 FULL decode graph——PIECEWISE 路径下 Mamba 算子是 eager 执行的,并不受影响。同时 FULL decode 的 dispatcher(`vllm/v1/cudagraph_dispatcher.py:217-220`)已经把 decode 捕获尺寸过滤到 `<= max_num_seqs * uniform_decode_query_len`,所以真正需要的安全检查只是 `max_num_seqs <= num_blocks`。原实现把约束错误地扩大到 prefill 路径,导致性能问题。

## 修复方案
- 删除 `vllm/config/compilation.py` 中的 `adjust_cudagraph_sizes_for_mamba_cache`(及相关单测)。
- `vllm/v1/worker/gpu_model_runner.py` 的 `_check_and_update_cudagraph_mode`:对含 Mamba 且 FULL cudagraph 的配置,不再 cap,而是当 `self.max_num_reqs > num_blocks` 时直接 raise ValueError,提示降低 `max_num_seqs` 或提高 `gpu_memory_utilization`:

```python
if cudagraph_mode.has_full_cudagraphs() and not is_profiling:
    has_mamba = any(
        isinstance(g.kv_cache_spec, MambaSpec) for g in kv_cache_groups
    )
    if has_mamba and self.kv_cache_config is not None:
        num_blocks = self.kv_cache_config.num_blocks
        if self.max_num_reqs > num_blocks:
            raise ValueError(...)
```

- 新增 `is_profiling` 参数,经 `initialize_kv_cache` → `initialize_attn_backend` 传入;`_init_minimal_kv_cache_for_profiling` 以 `initialize_kv_cache(minimal_config, is_profiling=True)` 调用,使 profiling 用的「dummy」KV cache 创建阶段跳过该检查。
- 改写 `tests/v1/worker/test_gpu_model_runner.py` 测试为断言抛出 ValueError。

## 影响与回归风险
- 回归面:含 Mamba 层模型(FULL cudagraph 模式)的启动路径——块数不足时从「静默降级」变为「启动报错」,之前依赖静默 cap 能跑起来的部署(块数略小于 max_num_seqs)升级后会无法启动,需按提示调整参数;profiling 阶段的 minimal KV cache 初始化被显式豁免,行为与之前一致。

## 关联
#37416(被部分回退)、#34094(代码注释引用)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
