# PR #35758: [Core][KVConnector] Support HMA+NixlConnector

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @NickLucche |
| 合并 | 2026-03-06 |
| 分类 | Overlap / Scheduler \| tier: major |
| Label | ready, v1, kv-connector |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)，pack 末尾标注 6 file(s) dropped from diff for size |
| 链接 | https://github.com/vllm-project/vllm/pull/35758 |

## 一句话结论

NixlConnector 支持 HMA，SWA+FA 模型 8xH100 ShareGPT 吞吐最高约 +7%。

## 动机

PR 正文 Overview：当前 connector 无法充分利用采用混合注意力（FA+SWA）的模型，因为 Hybrid KV Cache Manager 被禁用，connector 把所有层都当作 FA 处理，导致一次 xfer 移动的 bytes/regions 过多。示例对比：NON-HMA（当前 master）`get_block_descs_ids num output 4284`，而 HMA（本 PR）为 `1650`。本 PR 让 NixlConnector 与 HMA 协同，大幅减少 SWA+FA 模型每次 xfer 的 bytes/regions，并为 state-based（mamba 等）模型铺路；与 #32204 相同，但去掉了 xfer 失败时的 kv blocks recovery。

## 优化点

让 NixlConnector 声明 `SupportsHMA`，按 `FullAttentionSpec` / `SlidingWindowSpec` / `MambaSpec` 区分 KV cache group，新增 `BlockIds` 与 `cdiv` 处理逻辑↔物理 block 映射；在 `compute_nixl_compatibility_hash` 中加入 `is_hma_enabled = not scheduler_config.disable_hybrid_kv_cache_manager` 因子，使 HMA 开关进入兼容性哈希。`kv_cache_manager.py` 与两个 accuracy 测试脚本相应适配。需以 `--no-disable-hybrid-kv-cache-manager` 显式开启 HMA 实验支持。TODO 列出报告/处理 block 级失败、验证逻辑↔物理 kernel block 路径、HMA 关闭无回归、多配置 lm-eval 等（多数已勾选）。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py: 声明 SupportsHMA、按 KV cache spec 分组、加入 is_hma_enabled 兼容哈希因子。
vllm/distributed/kv_transfer/kv_connector/utils.py: 新增 BlockIds 供 HMA 路径使用。
vllm/v1/core/kv_cache_manager.py: HMA 下的 KV cache 管理适配。
tests/v1/kv_connector/nixl_integration/config_sweep_accuracy_test.sh、run_accuracy_test.sh: HMA accuracy 回归脚本。

  is_hma_enabled = not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
  factors = {
      ...
      "is_hma_enabled": is_hma_enabled,
  }
```

pack 有 6 个文件因体积被 dropped from diff。

## 性能声明

测试条件（Benchmarks 段）：ShareGPT、no-prefix-caching、8xH100，1000 requests、max concurrency 10、Failed 0；配置标注为 DTP4-PTP4 与 "WideEP" D DPEP4 - PTP4；PR 未注明该 benchmark 的模型与精度（"Test with" 段以 `google/gemma-3-4b-it` 为例，lm-eval gsm8k 得 0.74±0.0441）。Main DTP4-PTP4：duration 74.16 s，Request throughput 13.48 req/s，Output 2614.71 tok/s，Total 5518.14 tok/s，Mean/Median/P99 TTFT 92.65/36.87/2410.05 ms，Mean TPOT 3.31 ms。Main WideEP：duration 99.42 s，10.06 req/s，Output 2001.95 tok/s，Total 4167.65 tok/s，Mean TTFT 90.22 ms，Mean TPOT 4.51 ms。This PR HMA DTP4-PTP4：duration 68.28 s，14.65 req/s，Output 2798.60 tok/s，Total 5951.91 tok/s，Mean/Median/P99 TTFT 45.78/37.02/563.35 ms，Mean TPOT 3.32 ms。This PR HMA WideEP：duration 98.65 s，10.14 req/s，Output 2017.54 tok/s，Total 4200.10 tok/s，Mean TTFT 88.58 ms，Mean TPOT 4.48 ms。PR 结论：「so up to ~7% throughput in this small-scale intra-node setup. Inter-node one would be more interesting to analyze.」

## 验证与回归风险

测试：`pytest -x -v -s tests/v1/kv_connector/unit/test_nixl_connector_hma.py`；`cd tests && ENABLE_HMA_FLAG=1 bash v1/kv_connector/nixl_integration/config_sweep_accuracy_test.sh`（本地验证部分 lm-eval CI 的不同 tracked configs）；`python -m pytest -s -v -x tests/v1/kv_connector/unit/test_invalid_blocks_correctness.py::test_hma_sync_recompute_evicts_all_blocks`（HMA 下 invalid block 处理）；lm-eval gsm8k 0.74/0.74。TODO 中已勾选 pre-commit+mypy、报告处理 block 级失败、验证逻辑↔物理 kernel block 路径、HMA 关闭无回归、多配置 lm-eval；未完成：verify with llama4（旧优化已移除）、verify host-backed transfers (D2H->H2D)，mamba-like 模型与 block_size_ratio !=1 推迟到单独 PR。风险：需显式开启 HMA 开关；与 #32204 相比去掉了失败时 kv blocks recovery；仅小规模 intra-node 验证，inter-node 未测。

## 关联

PR 正文明确提及：#32204（相同实现思路，但保留 xfer 失败时的 kv blocks recovery；本 PR 去掉该恢复）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
