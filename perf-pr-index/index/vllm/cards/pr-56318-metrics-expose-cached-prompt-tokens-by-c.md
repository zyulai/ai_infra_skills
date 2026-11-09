# PR #56318: [Metrics] Expose cached prompt tokens by cache tier

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @cquil11 |
| 合并 | 2026-10-06 |
| 分类 | KV Cache / Memory(tier: major) |
| Label | ready, kv-connector, verified, rust, scheduler, mooncake |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files;diff 可用但另有 23 个文件因超长未收录) |
| 链接 | https://github.com/vllm-project/vllm/pull/56318 |

## 一句话结论

新增 `vllm:prompt_tokens_cached_by_source_total{source}` 指标,把 prefix cache 命中按 device/host/p2p/disk 五档来源分层计数,支撑 KV 分层缓存的性能可观测性。

## 动机

随着 KV offload(device→host→disk)与 P/D、跨实例共享(NIXL、Mooncake、MoRIIO 等 connector)的普及,`vllm:prefix_cache_hits` 与 `vllm:external_prefix_cache_hits` 只给总量,无法回答"命中到底来自 HBM、CPU 内存、磁盘还是远端实例"——而这直接决定命中 token 的加载代价与真实收益。缺一个按缓存层级归因的计数器,运维无法评估 offload 配置与 connector 选型的效果(closes #53368)。

## 优化点

归因走既有 `PrefixCacheStats` 通道,不改变任何现有计数器语义:本地 GPU prefix cache 命中计为 `device`;connector 侧命中通过新基类接口 `get_external_cache_hit_sources(request, num_external_tokens)` 返回 `dict[CacheHitSource, int]`,在 admission(alloc 成功)后调用一次,必须与 `num_external_tokens` 对账,不符则整体回退为 `external_unspecified` 并发一次性告警。默认实现按连接器类属性 `_cache_hit_source` 全量归类:Simple offload 按 backend 报 `host`/`disk`;NIXL、Mooncake transfer、MoRIIO 报 `p2p`;Mooncake Store 与未接入的 connector 报 `external_unspecified`;MultiConnector 委托给实际中标的子 connector。Native offload 的精细实现基于实际选择的 load key:`RequestOffloadState.load_key_ranges` 记录每个加载 key 的 token 区间,用端点扫描(sweep)合并区间,重叠 KV 组按"最外层 tier"归因(host+disk 混合报 disk),未覆盖区间(含被省略的 sliding-window 历史)报 `external_unspecified`,promotion 保持发起请求的原 tier。Python 与 Rust 两个前端同步实现(Rust 侧为 `CachedTokensBySource` 结构 + `CacheHitSourceLabels` label family,稀疏 dict 缺档默认 0,旧引擎序列化兼容)。

## 关键改动

```text
vllm/v1/metrics/cache_hit_source.py: CacheHitSource 枚举与 outermost 归并(23 文件未收录,推断自引用)
vllm/distributed/kv_transfer/kv_connector/v1/base.py: get_external_cache_hit_sources() 默认实现与 _cache_hit_source
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py: load_key_ranges 记录 + 区间扫描分层归因
vllm/distributed/kv_transfer/kv_connector/v1/{nixl,mooncake,moriio,multi_connector}: 各 connector 声明 p2p/委托
rust/src/engine-core-client/src/{metrics,protocol/stats}.rs: Rust 指标管线与 serde 兼容
```

区间扫描归因核心(节选):

```python
for boundary, delta, source in sorted(events):
    if boundary > position:
        tiers = [tier for tier, count in active.items() if count > 0]
        tier = (CacheHitSource.outermost(tiers)
                if tiers else CacheHitSource.EXTERNAL_UNSPECIFIED)
        sources[tier] = sources.get(tier, 0) + boundary - position
    active[source] += delta
    position = boundary
```

## 性能声明

无吞吐/延迟性能声明(regex 检测为 none),本 PR 是纯可观测性改动。Validation 为功能验证:SemiAnalysisAI/InferenceX Run Sweep,DeepSeek-V4-Pro、1 小时 AIPerf AgentX profile、关闭 eval、AIPerf 抓取 Prometheus,JSON/CSV 来源计数与对账检查通过;B200 上跑 DRAM c8、Simple NVMe c14、native DRAM+NVMe c14 三种配置,GB300 上跑 NIXL + Mooncake Store c256(4 个请求错误 0.197%、430 次运行末取消,在预设通过标准内)。这些为正确性验收而非性能对比。PR 未注明其余测试条件。

## 验证与回归风险

新增测试:`test_kv_connector_records_external_cache_hit_sources`(同步/异步 connector 的 admission 归因、失败分配不归因、异步加载完成后不计重复命中);offloading connector 的部分尾部来源保持、不同组 chunk size 对账、区间边界三组参数化、recurrent-only 状态、SWA/chunked-local 稀疏注意力加载范围 × 5 来源 × 多参数共数十例;simple_kv_offload 的 backend→source 映射;Rust 侧 serde 缺省字段兼容测试。回归风险小但面广:所有 KV connector 基类新增带默认值的方法(不实现也不破坏);指标注册新增 label family,抓取端 cardinality 增加 5 倍;`update_state_after_alloc` 内清空并重建 `load_key_ranges`,若 connector 在 alloc 后再次调用可能重复归因(测试覆盖了单次调用约定);torch profiler 测试补 `torch_profiler_with_stack=False` 为无关小修。

## 关联

Closes: vllm#53368。同区域:本批次 PR #57705(MoRI-IO)、本 PR 中 MoRIIO/NIXL/Mooncake connector 均归属 P2P 档;KV offload 体系(native offload、SimpleCPUOffloadConnector)是其 device/host/disk 档的数据源。
