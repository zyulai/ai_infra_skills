# PR #30785: [Fix]Load kv-cache dtype from hf_quant_config.json automatically (fix for reverted PR)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#30785](https://github.com/vllm-project/vllm/pull/30785) |
| 作者 | danielafrimi |
| 合并日期 | 2025-12-17 |
| labels | ready |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
modelopt checkpoint 声明 FP8 KV cache 却被忽略;在创建 CacheConfig 前从 hf_quant_config.json 解析,等同显式传参。

## 问题现象
HF checkpoint 的 hf_quant_config.json(modelopt)指定 `"kv_cache_quant_algo": "FP8"`,但 vLLM 仍加载未量化的 kv-cache;只有显式设置服务端参数 `--kv-cache-dtype fp8` 才生效。

## 根因
正文披露:此前应支持该功能的 PR #29980 有 bug 被 revert(CI 在部分测试上失败;作者本地在 B200 上可以通过)。本 PR 的关键修复是把转换提前到 `arg_utils.py` 创建 `CacheConfig` 时完成,使解析后的值在全代码库可见,行为与显式传 `--kv-cache-dtype` 完全一致。

## 修复方案
`vllm/utils/torch_utils.py` 新增 `MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP`("fp8"→"fp8_e4m3")及 `get_kv_cache_quant_algo_string` / `get_kv_cache_quant_algo_dtype` / `resolve_kv_cache_dtype_string`:后者在 `kv_cache_dtype == "auto"` 时读取 `model_config.hf_config.quantization_config`,对 modelopt 取 `kv_cache_quant_algo`(兼容顶层或 `quantization` 内层字段)并映射到 vLLM cache dtype 字符串;未知值打 warning 并安全回退 "auto"。`vllm/engine/arg_utils.py` `create_engine_config` 中:
```python
# Resolve "auto" kv_cache_dtype to actual value from model config
resolved_cache_dtype = resolve_kv_cache_dtype_string(
    self.kv_cache_dtype, model_config)
cache_config = CacheConfig(
    ...
    cache_dtype=resolved_cache_dtype,
```

## 影响与回归风险
- 回归面:`kv_cache_dtype="auto"` 且 checkpoint 带 modelopt quantization_config 时,kv-cache dtype 可能自动变为 `fp8_e4m3`(此前保持 auto);显式传参行为不变;未知 kv_cache_quant_algo 告警回退 "auto",不支持更多 modelopt KV dtype(映射表留有 TODO)。

## 关联
前次被 revert 的 PR #29980;CI 失败记录(buildkite build 43393)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
