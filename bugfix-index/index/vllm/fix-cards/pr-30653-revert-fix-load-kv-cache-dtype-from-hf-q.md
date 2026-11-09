# PR #30653: Revert "[Fix]Load kv-cache dtype from hf_quant_config.json automatically"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#30653](https://github.com/vllm-project/vllm/pull/30653) |
| 作者 | robertgshaw2-redhat |
| 合并日期 | 2025-12-15 |
| labels | ready, ci-failure |
| 热度 | 👍 2 · 💬 3 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
#29980 自动从 hf_quant_config.json 读 kv-cache dtype 导致 CI 失败(CUTLASS MLA 不支持 FP8 kv cache),整体回退。

## 问题现象
被回退的改动(#29980,自动从 `hf_quant_config.json` 加载 kv-cache dtype)合入后破坏了 CI,正文给出失败的 Buildkite 构建链接(build #43393)。正文披露失败原因:CUTLASS MLA 不支持 FP8 kv cache。由 diff 推断触发条件:当 `kv_cache_dtype=auto` 且模型 hf_config 携带 modelopt 系 `quantization_config`(含 `kv_cache_quant_algo`)时,新逻辑会自动把 kv-cache dtype 解析为该算法对应的 dtype(如 FP8),使使用 CUTLASS MLA 的用例拿到其不支持的 FP8 kv cache 而失败。

## 根因
正文披露:#29980 引入的「自动从 hf_quant_config.json 读取 kv-cache dtype」行为破坏 CI,根本原因是 CUTLASS MLA 不支持 FP8 kv cache,自动推断出的 dtype 会在该后端上出问题。

## 修复方案
完整回退 #29980 对 `vllm/utils/torch_utils.py` 的改动:
1. 删除新增的 `get_kv_cache_quant_algo_dtype()`(从 modelopt 系 quant_cfg 读取 `kv_cache_quant_algo` 并映射为 torch dtype);
2. `kv_cache_dtype_str_to_dtype()` 恢复为:仅当 `kv_cache_dtype == "auto"` 时返回 `model_config.dtype`(单测无 model_config 时回退 `torch.half`),其余走 `STR_DTYPE_TO_TORCH_DTYPE` 查表。关键 diff 摘录:

```python
if kv_cache_dtype == "auto":
    # Model config may not be specified for unit tests, default to float16
    return model_config.dtype if model_config else torch.half
return STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
```

## 影响与回归风险
- 影响窗口:自 #29980 合入至本回退合并(正文未给出版本号)。
- 回归面:kv-cache dtype 解析行为回到 #29980 之前;「从 hf_quant_config.json 自动推断 kv-cache dtype」的能力被移除,依赖该自动行为的用户需显式指定 `--kv-cache-dtype`;CUTLASS MLA 等不支持 FP8 kv cache 的后端恢复不被自动切换 dtype。

## 关联
- 被回退的 PR:https://github.com/vllm-project/vllm/pull/29980
- 失败的 CI 构建:https://buildkite.com/vllm/ci/builds/43393#019b176c-8881-47f4-aaf9-5dac38740906

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
