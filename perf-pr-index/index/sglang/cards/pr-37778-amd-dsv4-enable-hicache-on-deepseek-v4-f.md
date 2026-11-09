# PR #37778: [AMD][DSV4] Enable hicache on deepseek-v4 fp8 unified attn

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @1am9trash |
| 合并 | 2026-09-17 |
| 分类 | Quantization(tier: major) |
| Label | quant, deepseek, hicache, sgl-kernel, run-ci, jit-kernel, run-ci-extra, memory-pool |
| Diff | diffstat 未提供(pack 素材无增删行数字) |
| 链接 | https://github.com/sgl-project/sglang/pull/37778 |

## 一句话结论

让 DeepSeek-V4 fp8 unified attention 的 KV 布局支持 hicache 分层缓存,服务长上下文/大并发场景。

## 动机

基于 #37413 引入的 DeepSeek-V4 fp8 unified attention:fp8 unified_kv 把一行 KV 拆成 fp8 nope([512B])加并行 bf16 rope([128B])两个 pool,而 hicache 的 host pool 假设单一 unified pool——只 offload nope 半区会让取回的页面带着旧 rope,产生错误输出且不崩溃。因此 fp8 路径此前直接 `raise NotImplementedError`,长上下文/大并发场景无法使用 hierarchical cache。

## 优化点

核心是把 hicache 的镜像机制泛化到"一行两池"布局。将原本只按 kv_buffer 构造页面字节视图的 `unified_region_buffers` 抽出通用的 `_unified_page_views(buffers, ratio)`,行宽改为读 `buf.shape[1]`;新增 `unified_rope_region_buffers`,在 fp8 布局下对 `kv_buffer_rope` 镜像出同样的页面视图。hicache 栈为 c4/c128 两个压缩区各构建一个 rope sibling host pool(`DEEPSEEK_V4_C4_ROPE`/`DEEPSEEK_V4_C128_ROPE`),nope 与 rope 两个 host pool 行索引一一对应;unified_kv 下hicache 只允许整页搬运(`page_aligned_only=is_unified_kv`)。新 PoolName 同步注册进 mooncake store 的 hybrid page component keys 与 rust radix tree 的枚举/解析,bf16 布局 rope region 返回 None、行为不变。

## 关键改动

```text
python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py: 抽出 _unified_page_views,新增 unified_rope_region_buffers
python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py: 为 c4/c128 构建 rope sibling host pool entry,unified_kv 下 page_aligned_only
python/sglang/srt/mem_cache/hicache_storage.py: 新增 DEEPSEEK_V4_C4_ROPE / DEEPSEEK_V4_C128_ROPE PoolName
python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py: hybrid page keys 加入两个 rope pool
rust/sglang-radix-tree/src/unified_tree_core.rs (+ python_bindings.rs): rust 侧 PoolName 枚举与字符串解析同步
```

rope 半区需要独立 host pool 的原因(摘自新方法 docstring):

```python
"""
The bf16 rope half of an fp8 two-pool row, or None when there isn't one.
...
It needs its own host pool: offloading the nope half alone leaves whatever
rope the row held before, which is wrong output rather than a crash.
"""
```

## 性能声明

PR 正文 "Speed Tests and Profiling" 一节为 "None.",未提供前后对比吞吐数字(正文中检测到的 95.7% 是 theoretical cache hit rate,非加速比)。提供的功能性/准确性数据(硬件与模型规模 PR 未注明):

- GSM8k,1319 题,hicache 开启:Accuracy 0.946,Invalid 0.000,Latency 97.326 s,Output throughput 1234.024 token/s。
- Agentic workload(来自 SA)短测,20 分钟 replay,`mem-frac` 0.70:881/1588 请求成功(707 warmup,0 error dropped),QPS mean=1.42 p75=2.00 p95=4.00,GPU cache hit rate 92.3%,CPU/offload 3.0%,External 3.0%,GPU KV usage 100%(容量 14,359,040 tokens),theoretical cache hit rate 95.7%,throughput per GPU 17198 tok/s。作者注明仅为 hicache 功能验证,不是 perf sweep。

## 验证与回归风险

验证:`test/registered/unit/mem_cache/test_dsv4_unified_fp8_pool.py` 新增 4 个 hicache 相关 unittest(整文件 13 passed),钉住 fp8 nope/rope 配对、bf16 不增生第二个 host pool、视图起始越过 SWA ring、fp8 两半合计字节数等于 bf16(0.625x);`test_dsv4_hicache_l2.py` 的 stub 补上 `unified_rope_region_buffers=lambda ratio: None`。风险:触及 DSV4 hicache 全部布局路径——bf16/fp8 unified、NPU sidecar、mooncake store、rust radix tree;nope/rope pool 若配对错误会静默产出错误结果(测试明确针对这一点);CI 状态中 AMD ROCm 10 run 失败,Base/Extra 通过。

## 关联

前序:#37413(deepseek v4 fp8 unified attn,本 PR 在其上启用 hicache)。同区域:hicache/memory-pool 相关改动、DeepSeek-V4 KV 布局优化;同属 Quantization 分类。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
