# PR #53695: [ROCm][Feature] Support KV connectors with ROCM_AITER_UNIFIED_ATTN

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @simondanielsson |
| 合并 | 2026-09-10 |
| 分类 | Attention / Kernel（tier: major） |
| Label | rocm, ready, kv-connector, verified |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/53695 |

## 一句话结论

让 ROCM_AITER_UNIFIED_ATTN 支持 KV connectors，MI300X 上 Gemma4+MoRI 的 GSM8K 达 0.7415。

## 动机

正文 Purpose 只有两句：「It's a block first layout so should be safe to enable. Previously it was inheriting `False` from `ROCM_ATTN` which is not a block first layout.」即 ROCM_AITER_UNIFIED_ATTN 属于 block-first layout，启用 KV connector 支持是安全的；此前它从 ROCM_ATTN 继承了 `False`（而 ROCM_ATTN 并非 block-first layout）。这是功能性放开，正文未描述性能瓶颈或量化退化。

## 优化点

`vllm/v1/attention/backends/rocm_aiter_unified_attn.py` 新增 `_kv_connector_enabled()`：读 `get_current_vllm_config_or_none()` 的 `kv_transfer_config`，判断其存在且为 `is_kv_transfer_instance`；给后端新增 `supports_kv_connector(cls) -> bool: return True`；`supported_kv_cache_layouts()` 在启用 KV connector 时只返回 `(KVCacheLayout.LBHNC,)`（注释：Connectors like MoRI assume contiguous blocks），否则仍返回 `(LBHNC, LHBNC)`。`vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py` 的 `build_offloading_config` 把原来的 `assert tokens_per_chunk_int % tokens_per_block == 0` 改成条件分支：不整除时抛 ValueError，提示 `'block_size'` 必须是 GPU KV cache block size 的整数倍、建议改用 `round_up(...)` 或设置 `blocks_per_chunk`，并为此引入 `round_up`。测试涉及 `tests/v1/attention/test_rocm_attention_backends_selection.py` 与 `tests/v1/kv_connector/unit/test_offloading_connector.py`。

## 关键改动

```text
vllm/v1/attention/backends/rocm_aiter_unified_attn.py: 新增 _kv_connector_enabled() 与 supports_kv_connector()，启用 connector 时把 layout 收窄为 LBHNC
vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py: assert 改为明确 ValueError，提示 block_size 须为 block size 倍数（用 round_up）
tests/v1/attention/test_rocm_attention_backends_selection.py: ROCm backend 选择单测
tests/v1/kv_connector/unit/test_offloading_connector.py: offloading connector 单测

# 机制摘录 (rocm_aiter_unified_attn.py)
    @classmethod
    def supports_kv_connector(cls) -> bool:
        return True

    @staticmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        if _kv_connector_enabled():
            # Connectors like MoRI assume contiguous blocks
            return (KVCacheLayout.LBHNC,)
        return (KVCacheLayout.LBHNC, KVCacheLayout.LHBNC)
```

## 性能声明

PR 正文未给出吞吐/延迟类定量性能数据；给出的是 GSM8K 正确性分数。条件：模型 `RedHatAI/gemma-4-31B-it-FP8-block`（Gemma4）、`--attention-backend ROCM_AITER_UNIFIED_ATTN`、`--kv-cache-dtype fp8`、`--block-size 16`、TP1，MoRI 场景在 MI300X 上以 P/D 分离 + router 运行，NIXL 场景在 2×MI300X 上，lm_eval gsm8k 5-shot。Test Result：正文称「Everything starts fine and we get the expected GSM8k score (~74%)」，表格给出 gsm8k flexible-extract exact_match `0.7415 ± 0.0121`、strict-match `0.7324 ± 0.0122`。这些是 accuracy 数字，非吞吐/延迟性能指标；正文未给出其它硬件/精度/workload 的定量性能结果。

## 验证与回归风险

Test Plan 为在 MI300 上用 ROCM_AITER_UNIFIED_ATTN 分别对 MoRI connector、DecodeBenchConnector、NIXL connector 跑 Gemma4 的 GSM8k，正文给出三组 P/D 实例与 router 的启动命令、DecodeBenchConnector 的 lm_eval 命令、以及 NIXL 的 chat completions curl 请求。Test Result 三点：1) 一切正常启动、得到预期 GSM8k 分数（~74%，见性能声明表格）；2) DecodeBenchConnector 下 Engine 正常启动并处理请求、不崩溃；3) NIXL 正常返回带 prefill/decode 地址的 chat completion（completion_tokens 10、prompt_tokens 31、total 41，system_fingerprint `vllm-0.27.2rc1.dev193+gaa9903490-f71e9648`）。回归范围：改动限定在 ROCM_AITER_UNIFIED_ATTN 后端与 offloading connector config；启用 KV connector 时 layout 由 `(LBHNC, LHBNC)` 收窄为仅 `LBHNC`；offloading config 由 assert 改为 ValueError，会影响 block_size 非 GPU KV cache block size 倍数的配置。正文未给出其它后端/硬件的回归评估，也未给出吞吐/延迟数据。

## 关联

无。PR 正文与 diff 未给出同区域其它 PR 编号或前序/后续线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
