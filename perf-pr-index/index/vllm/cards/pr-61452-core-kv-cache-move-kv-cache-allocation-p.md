# PR #61452: [Core][KV Cache] Move KV cache allocation policy out of KVCacheSpecs (RFC #42449 part 2)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LucasWilkinson |
| 合并 | 2026-11-13 |
| 分类 | KV Cache / Memory(tier: major) |
| Label | documentation, performance, rocm, speculative-decoding, torch.compile, ci/build, multi-modality, qwen, deepseek, cpu, kv-connector, nvidia, mrv2, mistral, DSv4, dflash, kimi, k3, scheduler, kv-cache-manager, minimax, inkling, DSv4.1, mooncake, pooling, hisparse, nixl |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/61452 |

## 一句话结论

KV cache 接口重构(RFC #42449 part 2):`KVCacheSpec` 收敛为只描述 layer 的 tensor layout,分配 policy(windows、recycling、prefix caching、Mamba modes)移到 KV cache manager(frozen `Params` dataclass 声明);15 个 spec 类(8 个纯 policy)变为 5 个 layout 类,`UniformTypeKVCacheSpecs`、`KVCacheSpecRegistry`、`Platform.register_custom_kv_cache_specs` 被移除。正文未给加速数字,属结构/可维护性重构(非测试代码 136 文件 +2514/−3510)。

## 动机

原 `KVCacheSpec` 同时承载 tensor layout 与分配 policy,衍生出 policy-typed specs(`FullAttentionSpec`、`SlidingWindowSpec` 等)、per-spec sizing 方法和 wrapper 聚合,以及 `UniformTypeKVCacheSpecs` 这类聚合结构。重构目标:每个 layer 声明自己的 policy 与 layout,group 就是"共享一张 block table 的 layer 集合";给插件/外树维护者提供了 drop-in shim gist(旧到新的完整映射;shim 为过渡方案,树内无 compat shim)。

## 优化点

- 数据结构:`ManagerConfig(manager_cls, params)`(frozen,如 `SlidingWindowManager.Params(sliding_window=128)`)表达 policy;`KVCacheLayerConfig(manager_config, spec)` 是 `get_kv_cache_config()` 的返回;`KVCacheGroup.from_layers(...)` 由 planner 按 manager_config、block_size、dcp_sharded 相同来分组,一个 group 一张 block table。scheduler 侧 `group.manager_cls(group, ...)` 负责分配,worker 侧 `group.kv_cache_specs[layer]` 决定 KV tensor 形状。
- Specs:15 个类(8 个 encoding-only policy)收敛为 5 个 layout 类:`AttentionSpec`、`MLAAttentionSpec`、`HiddenStateCacheSpec`、`MambaSpec`、`HiSparsePageSpec`;`SlidingWindowMLASpec`、`CircularBufferSpec`、`EncoderOnlyAttentionSpec` 等被 `MLAAttentionSpec`/`AttentionSpec` 取代,窗口等参数改由 layer/manager 提供。
- Sizing:`manager_cls.max_blocks_per_request(manager_config, ...) x group.page_size_bytes` 取代 per-spec 方法与 wrapper 聚合。
- Backend builders:从 layers 与 `vllm_config` 读取所需信息(如 `MambaBase.get_kv_cache_checkpoint_alignment()`、layer 的 sliding window),任何 backend 都看不到 manager params。
- 行为注记:cache-hit lookup 仅在共享 manager config、block size、DCP sharding 与 layout 时才 batch KV cache groups(layout 项在 EAGLE last-block drop 消失后可移除,#64096 把 lookahead block hashes 扩展到 KV connectors);Mooncake store coordinator 改用同一规则(不再 batch 首层共享 spec 的 packed groups);PP 下 group 投影到某 stage 仍保留整个 group 的 distinct specs(`KVCacheGroup.distinct_specs`)。

## 关键改动

> pack 的 diff 被截断(263 个文件未展示,核心的 `kv_cache_interface.py` 等实现文件未在展示部分),以下仅覆盖已展示部分(以 benchmark/tests 的 API 迁移为主)。

```text
vllm/v1/kv_cache_interface.py(+206/−945,正文给出,文件本体在未展示部分): spec 收敛为 5 个 layout 类
vllm/v1/core/single_type_kv_cache_manager.py(未展示): Manager.layer_config / Params / max_blocks_per_request
benchmarks/attention_benchmarks/{common,mla_runner,runner}.py: MockLayer/benchmark 改 get_kv_cache_config + AttentionSpec
tests/...(大量迁移): FullAttentionSpec→AttentionSpec、SlidingWindowMLASpec→MLAAttentionSpec、KVCacheGroupSpec→KVCacheGroup.from_layers、get_kv_cache_spec→get_kv_cache_config 等
```

```diff
-    def get_kv_cache_spec(self):
-        """Get the KV cache spec (required by AttentionLayerBase)."""
-        return self._kv_cache_spec
+    def get_kv_cache_config(self, vllm_config=None):
+        """Get the KV cache config (required by AttentionLayerBase)."""
+        from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
+
+        return FullAttentionManager.layer_config(self._kv_cache_spec)
```

## 性能声明

- 正文未提供任何加速/吞吐数字(regex 检测为 none);收益为结构性的:消除 `UniformTypeKVCacheSpecs`、统一分组与 sizing 规则。
- 等价性(非性能):Pipeline parallelism 2 下 Qwen3-4B 与 Qwen3.5-0.8B(hybrid)KV cache 大小与 greedy 输出同 main 完全一致;E2E 精度(prefix caching,GLM-5.3-Flash / DeepSeek-V4.1-Flash / P/D NIXL)与 main 相当(见验证)。

## 验证与回归风险

- 当前 head(rebased on `e2421d2ac5`,1x B300):`pre-commit run mypy --all-files` 与 ruff 通过;`df5d80339b` 上 core/kv_connector/worker 相关套件 2635 passed、165 skipped,外加一个 main 上也失败的用例;另有四个文件在 main 上同树同错地失败。
- 更早 base(`9a77b4201b`):1490 passed、149 skipped(`test_finalize_extensible_kv_cache_auto_max_model_len_warns_when_short[True]` main 上同败;两例单独跑双侧通过)。`2a14a5e477` 上约 12,000 passed,所有失败要么 main 也有要么单独跑通过。
- CPU sweep(`tests/v1/{core,kv_connector/unit,worker,attention,engine,simple_kv_offload,kv_offload}`)与 base 对比:branch-only 失败均为设备原因(无 CUDA、NCCL、world size、CLI 不在 PATH);两个真实 test bug 已修。
- E2E 精度(prefix caching,branch vs main;12 个关于约 9.4K-token 文档的问题,cold/warm-hit/batched;未在当前 head 重跑):GLM-5.3-Flash 34/36 vs 34/36;+MTP(3 tokens)33/36 vs 32/36;DeepSeek-V4.1-Flash TP4 36/36(token-identical)vs 36/36;GLM-5.3-Flash P/D(NIXL,TP2→TP2)23/24 vs 22/24。
- 风险:正文自述为 Draft 时仍有 pending——hybrid SWA 模型、HiSparse 与 Mooncake 的 E2E,以及上述精度 run 在当前 head 的重跑;接口破坏性变更(worker/engine core 之间),外树 spec 与 platform-registered manager 需迁移(gist shim 为过渡);cache-hit lookup 的 layout 项与 EAGLE last-block drop 语义有耦合(见行为注记)。

## 关联

- RFC #42449(本 PR 为 part 2);迁移 gist(LucasWilkinson/bb5cd70fc41357fe4245006214c8f6af);#64096(lookahead block hashes 扩展到 KV connectors,届时可去掉分组中的 layout 项)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
