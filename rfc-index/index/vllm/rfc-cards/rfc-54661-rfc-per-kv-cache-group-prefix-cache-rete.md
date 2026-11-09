# RFC #54661: [RFC]: Per-KV-cache-group prefix-cache retention, with a hit-min exemption for hit-inert EAGLE sliding-window draft groups

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54661](https://github.com/vllm-project/vllm/issues/54661) |
| 作者 | nood-co1 |
| 创建日期 | 2026-09-01 |
| 状态 | 🟢 open |
| labels | speculative-decoding |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |

## 一句话结论
把 prefix-cache retention 按每个 KV cache 组分别解析,并把「hit 无用」的 EAGLE 滑窗草稿组豁免出 hit min,以恢复多对话共存容量、保留细粒度 hit 网格。

## 提案动机
携带 EAGLE 式滑窗草稿组(DFlash/DSpark 形状 drafter)的 hybrid 模型中,该组在每个 prefix-cache 边界 hash `cdiv(sliding_window-1, block_size)+1` 个 block id,并把释放的块挂到全体共享 LRU 的受保护尾部,同时(一旦豁免 hit min)对 reconciled hit 长度毫无贡献。GLM-5.3-Flash 部署(2× DGX Spark,TP=2,DFlash2 draft k=7,fork 实测)上,该组独占 87% 的 prefix cache,把共存对话压到 ~50K tokens each:每 cached 3584-token segment,草稿滑窗组占 33 个 block id(`cdiv(2047,64)+1`),四组类型合计 38,两对话仅当 `642 >= 43·S` 即 S≤14.9 才能共存(实测 14 segments OK / 17 evicting;45K/56K/66K 梯度 99%/0%/0%)。把全局 `--prefix-cache-retention-interval` 调到 14336 可恢复 99/98/97/96%(45K/56K/66K/80K),但粗化了所有组的 hit 网格(分化 sub-agent 前缀对 ~29K prompt 只命中 45%)。改为 per-group retention(草稿组 retention 0)+ hit-min 豁免后,每 segment 成本降到 5 个 id(每轮边界 +33),预计 S≤54(~195K tokens/对话)且保留 3584 hit 网格;第三方在另一 2× DGX Spark 部署复现两个 120K 对话交替,每次返回均 119,936/119,995 cached(99.95%),TTFT 冷 174.522s → 热 0.581s。所有数字为 vLLM fork 的部署证据而非上游验证;代码阅读针对 main @ `22df3a3`。

## 提出的方案
三部分,先就 1 和 3 征求政策同意、再开 PR:
1. 把满足「hit-inert」谓词的滑窗组豁免出 reconciled hit min,并定义显式 per-group lookup/consumption 契约:谓词为 EAGLE 注记(`KVCacheGroupSpec.is_eagle_group`)、window 短于 `_cache_hit_alignment_tokens`、稀疏 retention 真实可达(`need < alignment_tokens // block_size`)、`block_size` 整除 alignment、`extra_retained_tokens == 0`。组自身 hit 达到 candidate boundary 时照常消费其块;更短时不得缩短 `curr_hit_length` 也不得膨胀 `longest_hit_length`,其 per-group 结果须是显式「no reusable blocks; allocate a fresh window」而非现在会返回的空列表。作者明确指出未证明风险:drafter 的 acceptance 依赖窗口 KV 有效,设计必须保证窗口在 prefill 未缓存余量时被重算而非读到陈旧数据。
2. `KVCacheCoordinator` 增加 `retention_interval_by_group: tuple[int | None, ...]`,在 `__init__` 解析一次并在两处 `cache_blocks` 调用点索引;identity 解析下行为 byte-identical(seam patch 已 rebase 到 `22df3a3`:+37/−4、tests +72,故意暂不开 PR)。
3. 政策:满足谓词的组 `retention_interval = 0`(只缓存 reachable-boundary 尾部),其余组用配置值;真 SWA target 模型不受影响。

## 优化点与收益
部署证据数字:draft 滑窗组占 87% prefix cache;共存容量 ~50K → 预计 ~195K tokens/对话(约 4×);第三方复现 99.95% cached、TTFT 174.522s(冷)→ 0.581s(热)。条件:GLM-5.3-Flash,2× DGX Spark,TP=2,DFlash2 k=7,fork `0.1.dev20051+g487ecf187`;非上游验证,~195K 为容量模型推算(该模型在 retention 7168 处约 25% 偏乐观,120K×2 复现是实测点)。

## 现状与落地
仍 open。明确「先就政策达成一致再发 PR」;feedback period 一周;part 2 的 seam patch 已在作者 fork 就绪。

## 关联
- #48435(hybrid-SWA prefix caching 在多会话轮转下崩溃)
- PR #50457(all-sliding DFlash drafter + prefix caching)
- PR #54397(fine-grained SWA hits)
- #53786(SWA fine-grained hits,同部署形状)
- #53906(GLM-5.3-Flash 上游支持)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
