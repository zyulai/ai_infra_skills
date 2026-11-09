# PR #30393: [HiCache] Support packed and sidecar draft caches for MTP/EAGLE/DSpark

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @1e4ves |
| 合并 | 2026-08-06 |
| 分类 | Speculative Decoding (tier: major) |
| Label | high priority, hicache, run-ci, bypass-fastfail, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +/− 行数与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/30393 |

## 一句话结论

HiCache 补齐 draft 侧 KV/indexer/SWA 状态的 L2/L3 持久化,L3 重放后 spec 接受率恢复到冷启动水平。

## 动机

HiCache 此前只恢复目标模型的 KV cache,而 MTP/EAGLE draft 模型可能把 KV、DSA indexer 或 SWA 状态放在独立的 device pool。只恢复 target pool 时,前缀缓存会报告命中但 draft 侧状态缺失或槽位映射不一致,导致 L2/L3 重放后 speculative acceptance 明显下降(下表:重放后接受率从约 46-50% 掉到 33-35%)。draft 拓扑多样(NextN 打包式、独立 EAGLE、DSpark),需要按拓扑选择合适的缓存集成路径。

## 优化点

新增 HiCache draft plan,在构建 tree cache 前选定 packed 或 sidecar 模式。**Packed** 路径面向标准 NextN MTP/EAGLE(DeepSeek-V3.2、GLM-5.x、DeepSeek-V4、MiMo-V2.5)与 DeepSeek-V4 DSpark:把 draft KV/indexer/SWA buffer 作为尾层追加到对应 target host pool,与 target 共享 host slot、由同一次 HiCache 操作搬移——实现上用 `_with_mtp_layer_mapping` 把 draft 层映射追加到 layer mapping,`transfer_layer_num` 加上 draft 层数,L2 load-back 时按 `mtp_draft_device_pools` 恢复尾层。**Sidecar** 路径面向独立 EAGLE/EAGLE3、DFlash 与非 DeepSeek-V4 DSpark:为非空 draft 状态动态注册独立 host pool 条目(`DRAFT`/`DRAFT_INDEXER`/`DRAFT_SWA`),slot 数与源 target pool 一一对应,full KV/indexer 跟随 target KV 索引、draft SWA 跟随 target SWA 的 trailing-page 语义,并与源 pool 一起挂到同一 L2/L3 操作(含 Mooncake 存储后端)。另修正 `_page_backup` 的 rank 分片语义:`should_backup` 让 Mamba 等 rank-sharded sidecar 在每个 TP rank 备份,Mooncake 下 MHA draft/draft-SWA 用 rank-specific key,MLA/DSv4 draft 仍仅 TP0。Inkling HiCache MTP 明确不支持。

## 关键改动

```text
python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py:
  packed MTP 尾层映射(_with_mtp_layer_mapping)、build_full_draft_pools/
  build_swa_draft_pools/build_hicache_draft_sidecars sidecar 构建
python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py:
  register_host_pool_entry 动态注册、should_backup 备份判定、
  start_loading 恢复 packed draft 尾层
python/sglang/srt/mem_cache/kv_cache_builder.py:
  HiCacheDraftPlan 驱动注册,legacy 非 UnifiedRadixCache 路径回退
python/sglang/srt/mem_cache/memory_pool_host.py:
  HostPoolGroup.add_entry;load_to_device_per_layer 支持 is_draft
```

```python
class SidecarPoolSpec:
    pool_name=PoolName.DRAFT,        indices_from_pool=PoolName.KV
    pool_name=PoolName.DRAFT_INDEXER, indices_from_pool=PoolName.KV
    pool_name=PoolName.DRAFT_SWA,    indices_from_pool=PoolName.SWA
                                     hit_policy=PoolHitPolicy.TRAILING_PAGES
```

## 性能声明

L3 重放验证(同一 200 条 ShareGPT prompts,每请求 128 output tokens,cold run → flush device cache → L3 replay,对比 main@39537885961c):DeepSeek-V4-Flash-0731(DSpark)冷启动接受长度/率 3.317/46.77%,main 重放 2.660/33.57%,本 PR 重放 3.323/46.86%;DeepSeek-V4-Flash 冷启动 2.506/50.19%,main 重放 2.060/35.38%,本 PR 重放 2.515/50.51%——即重放后接受率恢复到冷启动水平。精度:GSM8K 上 DeepSeek-V4-Flash-FP8 97.0%(194/200,缓存 137,728 tokens,命中率 62.1%),DeepSeek-V4-Flash-0731 97.5%(195/200,125,184 tokens,62.3%)。无吞吐/延迟倍数声明。

## 验证与回归风险

PR 测试由 CI Base/Extra 两级通过(pack 中 CI run 均绿)。风险面:packed 尾层映射要求 draft host pool 与 target 的 slot 对应关系严格一致;sidecar 的 `add_entry` 要求 HostPoolGroup 且不与既有条目重名;`should_backup` 按 storage backend 类型(mooncake)分支,新存储后端需要补 rank-sharded 判定;legacy 非 UnifiedRadixCache 路径仅支持 MHA/MLA draft pool,其他类型静默跳过并 warning;MiMo draft 配置强制 `num_nextn_predict_layers = 1`。

## 关联

同区域(Speculative Decoding + HiCache):本批次 #23958(PD 分离非对称 MTP)同样处理 spec 元数据跨边界传递;与 #32219(spec-v2 host-seam)同属 spec 基础设施。pack 未列出直接前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
