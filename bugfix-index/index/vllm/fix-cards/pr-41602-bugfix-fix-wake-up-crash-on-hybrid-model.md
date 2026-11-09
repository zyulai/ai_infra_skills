# PR #41602: [Bugfix] Fix /wake_up crash on hybrid models (Mamba/DeltaNet)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#41602](https://github.com/vllm-project/vllm/pull/41602) |
| 作者 | kevglynn |
| 合并日期 | 2026-07-29 |
| labels | bug, mrv1-only, ready, v1 |
| 热度 | 👍 6 · 💬 5 |
| 分类 | Model-specific Fast Path |
| bug 类型 | crash |

## 一句话结论
`/sleep` 后调 `/wake_up` 时对 Mamba/DeltaNet 混合模型的 list 型 KV cache 条目调 `.zero_()` 崩溃,改为按 list/tensor 两种形态分别遍历清零。

## 问题现象
对使用 Mamba、DeltaNet 或其他 `MambaSpec` 层且开启量化 KV cache(`--kv-cache-dtype fp8_e4m3`)的混合模型,`/sleep` 之后调用 `/wake_up` 报错:
```
AttributeError: 'list' object has no attribute 'zero_'
```
引擎无法唤醒。Gemma-4 等纯 interleaved SWA 模型(全部条目都是 tensor)不受影响,可作为对照。

## 根因
正文披露:`init_fp8_kv_scales()`(经 `post_kv_cache_wake_up()` → `wake_up()` 调用)遍历 `self.kv_caches` 并对每个条目调 `.zero_()`,假定每个条目都是 `torch.Tensor`。但混合模型中 `MambaSpec` 层把 per-layer state 存为 tensor 列表而非单个 tensor(`_initialize_kv_cache_tensors` 中 `kv_caches[layer_name] = state_tensors`),经 `bind_kv_cache()` 混入 `self.kv_caches`。于是 `kv_caches` 同时含 `torch.Tensor`(attention 层)与 `list[torch.Tensor]`(Mamba/DeltaNet 层),循环碰到 list 即失败。

## 修复方案
改 `vllm/v1/worker/gpu_model_runner.py` 的 `init_fp8_kv_scales()`:遍历时区分条目形态,list 则逐个元素 `.zero_()`,tensor 仍直接 `.zero_()`:
```python
for cache_entry in kv_caches:
    if cache_entry is None:
        continue
    if isinstance(cache_entry, list):
        for t in cache_entry:
            t.zero_()
    else:
        cache_entry.zero_()
```
另在 `tests/v1/worker/test_gpu_model_runner.py` 新增 `TestInitFp8KvScalesHybridModels`,覆盖 tensor+list 混合、None 条目、非量化 noop 等场景。

## 影响与回归风险
- 回归面:仅 `init_fp8_kv_scales()` 的 KV cache 遍历逻辑,且仅在 KV cache 量化(fp8_e4m3)时生效;非量化路径直接 return。纯 attention 模型条目仍为 tensor,走原分支,行为不变。既有 sleep/wake 测试(`tests/basic_correctness/test_cumem.py`)用纯 attention 模型,不经过该路径(正文说明无回归风险)。

## 关联
- #41564(fixes)
- [硬件验证评论](https://github.com/vllm-project/vllm/pull/41602#issuecomment-4375996290)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
