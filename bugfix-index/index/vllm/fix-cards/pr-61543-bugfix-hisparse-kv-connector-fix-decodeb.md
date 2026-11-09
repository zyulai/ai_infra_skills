# PR #61543: [Bugfix][HiSparse][KV Connector] Fix DecodeBenchConnector crash on bounded KV cache groups

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#61543](https://github.com/vllm-project/vllm/pull/61543) |
| 作者 | grYe99 |
| 合并日期 | 2026-10-27 |
| labels | bug, hisparse, kv-connector, ready |
| 热度 | 👍 0 · 💬 5 |
| 分类 | KV Cache / Memory |
| bug 类型 | crash |

## 一句话结论
DecodeBenchConnector 对 HiSparse 关闭 KV transfer 的有界 hot group 套用 token-paged 断言,ISL>4096 即 EngineCore 致命崩溃;改为跳过非 transfer group。

## 问题现象
`HiSparse` + `DecodeBenchConnector` 在 GLM-5.3 上、ISL 大于 device_buffer_size(4096) 时崩溃:`AssertionError: DecodeBenchConnector block range exceeds allocated blocks: request=chatcmpl-bd27629bd71bb94d-9ff77f54, group=22, range=[0, 87), allocated=64`。断言在 scheduler 中于第一个符合条件(约 5.5k token)的请求上触发,EngineCore 直接 fatal,整个 serve 实例死亡。复现环境:单机 8×H200、TP8、vllm 0.32.0,MultiConnector(kv_both)组合 DecodeBenchConnector 与 HiSparseConnector(host_pool_gib 64)。ISL ≤ 4096 时只有巧合地通过。

## 根因
正文披露:HiSparse(v0.32.0 起进主干,#60072)通过 `HiSparseHotSpec`(直接继承 `KVCacheSpec`)添加每请求 hot-cache group,其分配有界:`blocks_per_request = cdiv(device_buffer_size, block_size)`;这些 group `enable_kv_transfer=False`,是每次 connector load 后从可 transfer 源 group 重建的临时副本,不参与外部 KV 传输。而 `DecodeBenchConnector.update_state_after_alloc` 假设所有非 `CircularBufferSpec` 的 group 都是 token-paged,并对计算出的 token range 断言必须落在分配内。崩溃数据可解码:allocated=64 是 hot group 每请求上限 cdiv(4096,64)=64,与序列长度无关;range=[0,87) 是对约 5.5k token 用 cdiv(total_computed_tokens,64) 的 token-paged 公式算出。因此 ISL>4096 必崩,ISL≤4096 只因 cdiv(ISL,64)≤64 巧合通过。group=22 是 GLM-5.3 的 78 个 `hisparse_hot` group 之一。

## 修复方案
改 `vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py`:初始化时收集 `enable_kv_transfer=False` 的 group 到 `_non_transfer_group_ids`;`update_state_after_alloc` 中对这些 group(与 startup-fill group 同样处理)选块列表置空并跳过,不再做 token-paged 断言。与 connector load 侧声明一致(`get_loaded_kv_cache_group_ids()` 默认即排除这些 group);NIXL/Mooncake 本来就以 `transfer_group_ids` 为准,不引入新的 spec 形状启发式。`enable_kv_transfer=True` 的 group(circular buffer、Mamba、sliding-window)路径与断言完全不变。

```diff
+        self._non_transfer_group_ids = {
+            group_idx
+            for group_idx, group in enumerate(kv_cache_config.kv_cache_groups)
+            if not group.enable_kv_transfer
+        }
...
-            if group_idx in self._startup_fill_group_ids:
+            if (
+                group_idx in self._startup_fill_group_ids
+                or group_idx in self._non_transfer_group_ids
+            ):
```

## 影响与回归风险
- 影响窗口:正文说明 HiSparse 自 v0.32.0(#60072)进主干,故影响 v0.32.0 起、本修复之前的版本。
- 回归面:仅 `decode_bench_connector.py` 的 fill 选块路径。非 transfer group 现在选零块(此前会在每个请求上警告选中 78 个不在 connector `kv_caches` 里的 `hisparse_hot` layer,现一并避免);所有 `enable_kv_transfer=True` 配置的 token-range 数学与 in-range 断言保持原样,行为不变。
- 验证:新增回归测试 `test_decode_bench_connector_skips_non_transfer_groups`(用真实 `HiSparseHotSpec`、enable_kv_transfer=False,断言 `reqs_to_fill` 对应项为空;在 main 上失败、本分支通过);单测 30/30 通过;复现配置 e2e 从必崩变为正常处理。

## 关联
- #60072(HiSparse 进主干的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
