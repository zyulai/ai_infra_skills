# PR #55053: [Bugfix][KV Connector][ROCm] Fix MoRI-IO KV block-offset for speculative decoding (MTP)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MohitAMD |
| 合并 | 2026-10-07 |
| 分类 | Speculative Decoding |
| Label | bug, rocm, ready, kv-connector |
| Diff | pack 未提供 diffstat(+/- 与 files 数缺失,详见 PR 页面) |
| 链接 | https://github.com/vllm-project/vllm/pull/55053 |

## 一句话结论

修复 MoRI-IO PD 分离 + MTP 下 lookahead block 导致 KV 传输崩溃,使 MTP TPOT 收益(1.36-1.45x)可用。

## 动机

在 MoRI-IO PD 分离上启用 speculative decoding(如 MTP)时,prefill(producer)侧 KV cache manager 会为 `num_speculative_tokens` 预留 lookahead block,而 decode(consumer)侧从不分配这些块,producer 的本地 block 列表比 consumer 的远端列表长,首次 prefill→decode KV 写即抛 `ValueError: local_block_ids longer than remote_block_ids`(n=3 时如 8003 vs 8000),MTP 在 MoRI-IO 上完全不可用。

## 优化点

 surplus 是尾部的 lookahead scratch 而非 prompt KV,因此应在 producer 保存前裁掉而非传输时掩盖。三处机制:其一,WRITE 保存路径新增 `_clamp_to_prompt_blocks`,按每个 full-attention KV 组各自的 block_size 截断到 `ceil(num_prompt_tokens / block_size)`,sliding-window 组不动,与 hybrid 多组管理器兼容;其二,final-chunk 判定从块计数改为 token 进度 `_is_final_prefill_chunk`(`num_computed_tokens + num_scheduled >= num_prompt_tokens`),对 lookahead 膨胀免疫,且正确处理不分配新块的 final chunk(`new_block_ids is None`,此前会漏掉保存导致 decode 挂死);其三,保持 READ 路径"local 真的更长即抛错"的响亮不变量,并为 WRITE + sliding-window(+spec)组合抛 `NotImplementedError` fail closed(位置式 prompt-prefix 保存会送出窗口外空块)。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py: _clamp_to_prompt_blocks + _is_final_prefill_chunk + SWA/spec 守卫
vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py: 保留 longer-local ValueError,注释澄清 READ partial transfer
tests/v1/kv_connector/unit/test_moriio_connector.py: clamp/chunked-prefill/lookahead 回归/SWA 守卫等单测
```

```python
num_prompt_blocks = math.ceil(
    req.num_prompt_tokens / self._group_block_sizes[g]
)
if len(group_blocks) > num_prompt_blocks:
    clamped[g] = group_blocks[:num_prompt_blocks]
```

## 性能声明

验证环境:GLM-5.3-FP8 WideEP 1P1D / 1P2D PD-disagg(vLLM v0.27.1 + AITER v0.1.19),MTP n=3:KV transfer 成功(如 8003→8000 / 8→5),无 token-0("!")损坏,FULL cudagraph 生效;8k/8k decode-heavy、低并发下呈现预期的 MTP TPOT 收益 1.36-1.45x(即 MTP 本身的收益,本 PR 使其在 MoRI-IO 上恢复可用)。单测 62 passed。

## 验证与回归风险

新增单测覆盖 WRITE-mode clamp(单组/多组、block_size 4/16 与整除/非整除)、chunked-prefill 累积、lookahead final-chunk 回归(final chunk 无新块)、WRITE spec+SWA 守卫;`test_moriio_kv_layout.py` 保留 longer-local ValueError 与 READ 短列表 partial-transfer 行为。风险面限于 MoRI-IO connector 的 WRITE 保存路径;非 spec PD 分离回归(等长 local/remote)不变。

## 关联

同区域(kv-connector):PR #54689(NIXL 在途 transfer 防误驱逐);构建于上游 hybrid/multi-KV-group MoRI-IO 重构之上。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
