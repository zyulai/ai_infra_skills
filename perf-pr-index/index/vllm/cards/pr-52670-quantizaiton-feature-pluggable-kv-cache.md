# PR #52670: [Quantizaiton][Feature] Pluggable KV Cache Data Types

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @lcfenglinwan |
| 合并 | 2026-10-20 |
| 分类 | Quantization(tier: major) |
| Label | performance, rocm, intel-gpu, ready, deepseek, cpu, nvidia, quantization, dflash, kimi, k3, minimax, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明「14 file(s) dropped from diff for size」 |
| 链接 | https://github.com/vllm-project/vllm/pull/52670 |

## 一句话结论

把 KV cache dtype 从封闭 Literal 改为可插拔注册，平台后端可注册自定义 `--kv-cache-dtype`。

## 动机

PR 正文未注明动机：Purpose 仅写「This PR is the implementation of https://github.com/vllm-project/vllm/issues/51751」，Test Plan 与 Test Result 两节均为空，未描述改动前的具体瓶颈，也未给出任何验证说明。本卡的机制描述全部来自 diff。

## 优化点

从 diff 看，核心是把原本硬编码的 KV cache dtype 判定改造成可插拔注册。(1) `vllm/config/cache.py` 新增可变列表 `KV_CACHE_DTYPES`、`KVCacheDTypeHandler` Protocol（`name` / `torch_dtype()` / `quant_mode()`，后者可返回 `KVQuantMode.CUSTOM` 由后端自管 kernel 派发）、`_KV_CACHE_DTYPE_HANDLERS` 字典与 `register_kv_cache_dtype(name)` 注册装饰器，供平台后端在注册期注入自定义 `--kv-cache-dtype`。(2) `AttentionConfig.indexer_kv_dtype` 与 `CacheDType` 由封闭 Literal 放宽为 `| str`，`resolve_indexer_kv_dtype` 签名同步放宽。(3) `STR_DTYPE_TO_TORCH_DTYPE` 被标注为 registration-time-mutable，明确禁止快照（不 `dict(...)`/`.copy()`/`list(.keys())`），以便后注册项对所有消费者可见。(4) `is_quantized_kv_cache` / `kv_cache_uses_per_token_head_scales` 从 `vllm/utils/torch_utils.py` 迁到 `vllm/v1/kv_cache_interface.py`，全仓大量 attention backend 与模型文件的 import 路径随之调整。

## 关键改动

```text
vllm/config/cache.py: 新增 KV_CACHE_DTYPES、KVCacheDTypeHandler Protocol、register_kv_cache_dtype 及自定义 dtype 处理
vllm/config/attention.py: indexer_kv_dtype 放宽为 IndexerKVDType | str
vllm/utils/torch_utils.py: STR_DTYPE_TO_TORCH_DTYPE 标注可变，移除两个 kv-cache 判定函数
vllm/v1/kv_cache_interface.py: 承接 is_quantized_kv_cache / kv_cache_uses_per_token_head_scales
vllm/v1/attention/backends/* 与 vllm/models/*: 大量 backend/模型 import 路径改用新位置
```

关键片段（可插拔注册契约）：

```diff
+KV_CACHE_DTYPES: list[str] = list(get_args(CacheDType))
+
+
+@runtime_checkable
+class KVCacheDTypeHandler(Protocol):
+    """Contract for a custom ``--kv-cache-dtype`` value.
+
+    A platform backend implements this protocol to answer the questions vLLM
+    currently answers with closed tables (``STR_DTYPE_TO_TORCH_DTYPE`` /
+    ``get_kv_quant_mode`` / ``is_quantized_kv_cache``). A single handler may
+    be device-aware (e.g., resolving different torch dtypes per device
+    generation).
+    """
```

## 性能声明

PR 正文未给出定量性能数据（perf claims 检测为 none，Test Result 为空）；PR 未注明测试条件。正文无任何 benchmark 数字。

## 验证与回归风险

PR 正文未注明测试与验证结果。diff 中含 `tests/config/test_kv_cache_dtype.py`、`tests/v1/attention/test_attention_backends.py`、`benchmarks/kernels/cpu/benchmark_cpu_attn.py` 等测试/基准文件的改动，但正文未说明其运行结果。回归风险：把全局判定函数从 `torch_utils` 迁到 `kv_cache_interface`、把 `CacheDType`/`indexer_kv_dtype` 由 Literal 放宽为 `str`，波及配置校验、量化判定与大量 attention backend 导入；pack 标注 14 file(s) dropped from diff，diff 不完整。

## 关联

正文仅引用 issue #51751（本 PR 的实现来源），无同分类其他 PR 编号线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
