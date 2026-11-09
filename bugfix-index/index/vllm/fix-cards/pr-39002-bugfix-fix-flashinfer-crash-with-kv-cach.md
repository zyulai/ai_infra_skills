# PR #39002: [Bugfix] Fix FlashInfer crash with kv_cache_dtype_skip_layers

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39002](https://github.com/vllm-project/vllm/pull/39002) |
| 作者 | yzong-rh |
| 合并日期 | 2026-04-10 |
| labels | bug, nvidia, ready, v1 |
| 热度 | 👍 1 · 💬 5 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
kv_cache_dtype_skip_layers 下 FlashInfer 把被跳过的层仍当 fp8 处理,q/kv dtype 断言崩溃;改按 per-group kv_quant_mode 判定。

## 问题现象
使用 `kv_cache_dtype_skip_layers`(#33695 引入)时 FlashInfer attention 崩溃。修复前的报错日志:

```
File ".../vllm/v1/attention/backends/flashinfer.py", line 1313, in forward
    assert attn_metadata.q_data_type == query.dtype, (
AssertionError: Query dtype mismatch: expected torch.float8_e4m3fn, got torch.float16
```

正文在 B200(FlashInfer attention 为默认后端)上用 `pytest tests/quantization/test_fp8.py::test_kv_cache_dtype_skip_layers -v` 复现;修复后该测试通过。

## 根因
正文披露:metadata builder 读取的是全局 `cache_config.cache_dtype`(例如 `"fp8"`),而不是 per-group 的 `kv_cache_spec.kv_quant_mode`,因此配置了 skip 的层仍被当作 fp8 量化层处理,q/kv dtype 不匹配触发断言失败。

## 修复方案
改动 `vllm/v1/attention/backends/flashinfer.py` 的 `__init__`:量化判定从 `is_quantized_kv_cache(self.cache_dtype)`(看全局 dtype)改为 `self.kv_cache_spec.kv_quant_mode != KVQuantMode.NONE`(看 per-group 量化模式);非量化分支将 `self.cache_dtype` 设为 `"auto"`,并注释说明不能直接用 `kv_cache_spec.dtype`——存储 dtype 与 op dtype 可能不同(uint8 vs fp8_e4m3)。关键 diff 摘录:

```python
if self.kv_cache_spec.kv_quant_mode != KVQuantMode.NONE:
    self.cache_dtype = self.cache_config.cache_dtype
    self.kv_cache_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
        self.cache_dtype
    )
else:
    self.cache_dtype = "auto"
    self.kv_cache_dtype = self.kv_cache_spec.dtype
```

同时更新 `tests/compile/passes/test_fusion_attn.py`:测试用 `AttentionSpec` 改为传入 `kv_quant_mode=get_kv_quant_mode(self.attn.kv_cache_dtype)`、`dtype=self.attn.kv_cache_torch_dtype`,并删除构造函数中冗余的 `kv_cache_dtype` 参数。

## 影响与回归风险
- 影响窗口:自 #33695 引入 `kv_cache_dtype_skip_layers` 起至本修复(正文未给出版本号)。
- 回归面:FlashInfer backend 对 cache_dtype/kv_cache_dtype 的判定路径;被 skip 的层将走非量化分支(`kv_cache_dtype` 取 spec dtype);未使用 skip 配置的常规 FP8 KV cache 场景判断结果不变(kv_quant_mode 非 NONE)。测试辅助类的构造签名有变,依赖该测试类的代码需同步。

## 关联
- https://github.com/vllm-project/vllm/pull/33695

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
