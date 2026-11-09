# PR #34012: Add Agentic-Aware Tail-Optimized LRU eviction to the unified radix cache

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yl3469 |
| 合并 | 2026-09-17 |
| 分类 | KV Cache / Memory(major) |
| Label | documentation, run-ci, bypass-fastfail, run-ci-extra, unified-radix-cache |
| Diff | pack 未提供 diffstat(PR 正文注明 +325/-7) |
| 链接 | https://github.com/sgl-project/sglang/pull/34012 |

## 一句话结论

给 unified radix cache 增加 T-LRU 驱逐策略,agentic 负载下 ITL p99 最多降 43.9%。

## 动机

Agentic/多轮对话负载下,一条对话的"尾部"(下一次 prefill 实际不依赖的缓存)恰好是普通 LRU 最保护的部分,代价是其他对话的尾部延迟最多恶化 43%。T-LRU(Tail-Optimized LRU,NeurIPS 25,arXiv:2510.15152)的思路是:历史长度 `L`、下一轮预计新增 `Q_hat` token 的对话,只要保留 `L + Q_hat - xi` 个 token 就能让下次 prefill 的重算开销落在 TTFT 预算 `xi` 之内,超出部分("TEL-safe")先驱逐,再退回普通 recency 顺序。此前 SGLang 的 radix cache 只有 lru/lfu/slru/priority 四种策略,没有针对尾部延迟的选项。

## 优化点

实现为优先级函数而非独立的驱逐流程:TEL-safe 节点的 `get_priority` 返回 `(-1, last_access_time)`(视作无限老),现有 device/host 驱逐循环天然先清空这类节点(论文 phase 1)再按 recency 继续(论文 phase 2),驱逐 driver 完全不用改。关键簿记是 `_tlru_history_len`——分支的高水位深度,在 insert/split 时维护,且刻意不随尾部被裁剪而收缩:若从当前缓存内容推导,被裁过的对话会每轮继续超预算、被一路裁到零,而不是停在 `xi - Q_hat` token 处。只在 `tlru` 策略下执行 per-insert 簿记,其他策略零开销;启动时校验拒绝两个会退化成 LRU 的配置(`xi <= Q_hat`、非 unified radix tree),并新增 `--radix-eviction-policy tlru` 与 `threshold`/`next_prompt_estimate` 两个参数(以 token 计,论文以 block 计,需乘 page_size 换算)。只有差值 `delta = xi - Q_hat` 作用于策略。

## 关键改动

```text
python/sglang/srt/mem_cache/evict_policy.py: 新增 TLRUStrategy(TEL-safe 判定与优先级)
python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py: _tlru_history_len 高水位簿记(insert/split)
python/sglang/srt/mem_cache/registry.py: 强制 tlru 仅用于 UnifiedRadixCache
python/sglang/srt/arg_groups/kv_cache_hook.py: 启动参数校验
docs/docs/advanced_features/radix_eviction_policy.mdx: tlru 参数文档
```

## 性能声明

测试条件:DeepSeek-V4-Pro FP4,4x B300,TP4/EP4/DP-attention,HiCache DRAM ratio 4,SemiAnalysis agentic trace(HuggingFace `semianalysisai/cc-traces-weka-062126`,9,843 个 Claude Code 会话)经 aiperf 回放,3,600 s 测量窗口,`xi=14592`/`Q_hat=2304`,客户端种子固定做 paired 比较(每并发 ~6,000+ 配对请求,取中间 80%)。C=32:TTFT p99 -8.6%、p95 -6.9%;ITL p90 -8.7%、p95 -17.2%、p99 -43.9%(代价:ITL p50 +2.2%)。C=40:TTFT p99 -13.6%,ITL p99 -11.0%(代价:TTFT p50 +5.1%)。吞吐与 prefix hit rate 不变(所有 delta 在重复实验散布内:C=32 tok/s/GPU -1.2%,散布 12.2%)。Write-back 研究显示 T-LRU 使 DRAM 写回量从 30.0M 增至 48.1M tokens(C=32),绝对带宽 10-20 MB/s,可忽略。作者注明该配置工作集仅 ~0.28x KV 容量、96% 命中率,是 cache 压力场景的下界;`xi`/`Q_hat` 是负载相对量,换负载需重新推导。

## 验证与回归风险

模型前向计算与 kernel 无任何改动:策略只改变哪些前缀被驱逐,被驱逐的前缀下次请求时精确重算,输出按构造不变;未开启 flag 时默认 `lru` 路径不变。新增 9 个 CPU-only 单元测试 `test/registered/unit/mem_cache/test_tlru_eviction_policy.py`(无需 GPU):覆盖精确释放尾部预算、裁剪后停止、phase 1 跨对话分布后回退 LRU、`Q_hat=xi` 时严格退化为 LRU、超大尾节点保护而非部分释放、compaction 分支共享前缀保持保护、budget 为零钳位、优先级可排序等,其中"裁剪后停止"是针对高水位推导错误的回归守卫。风险面主要是 split/insert 路径新增的簿记逻辑(仅 tlru 下激活)与策略参数的负载敏感性。

## 关联

同区域:unified radix cache 驱逐策略族(lfu/slru/priority 为既有策略);论文 arXiv:2510.15152。pack 未列出其他同分类 PR 编号。
