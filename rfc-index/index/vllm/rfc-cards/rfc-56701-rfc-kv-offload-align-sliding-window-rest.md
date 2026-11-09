# RFC #56701: [RFC][KV Offload]: Align sliding-window restore coverage with MTP-retained history

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56701](https://github.com/vllm-project/vllm/issues/56701) |
| 作者 | jacklin78911-collab |
| 创建日期 | 2026-09-13 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Speculative Decoding |

## 一句话结论
修复原生 KV offload 的覆盖契约:让滑窗 lookup/store 覆盖 MTP `extra_retained_tokens` 扩展出的保留历史,消除 restore 阶段的断言失败。

## 提案动机
原生 KV offload 可能先报告一个 sliding-window prefix hit,随后却尝试加载从未包含在该 hit 中的更旧 chunk。作者通过真实的 Python offload scheduler、CPU manager 与 GPU block allocator 类(无 mock)复现了 `Chunk ... not found in cache` 与 `Chunk ... is not ready for reading` 两种断言,无需并发驱逐或任何 GPU kernel。触发条件是 `SlidingWindowSpec.extra_retained_tokens > 0`:`get_kv_cache_configs()` 对 multi-module MTP 把它设为 `num_speculative_tokens - 1`,block manager 据此多保留历史,但 offload lookup 与 producer 的 reachable-store mask 仍按未扩展的 attention window 计算。最小布局(full-attn 64 tok/block;SWA 16 tok/block、window 32、extra 3;65-token prompt、零本地命中)下:64-token hit 边界处 attention 需要从 token 33 起的历史(SWA chunk 2),allocator 实际保留从 token 30 起(chunk 1);extra=3 时 chunk 1 为 missing/pending/ready 分别导致 not found 断言 / not ready 断言 / 成功;且 producer 的 `_reachable_store_block_mask()` 此时返回 `[False, False, True, True]`,过滤器本身就漏掉 chunk 1。作者注明这是 bookkeeping 复现而非完整模型服务复现(prompt hash 与配置对象为合成,不实例化 MTP drafter),代码检视针对 main @ `fa1b3b19`。

## 提出的方案
保持现有 allocator 与 MTP retention 语义,让原生 offload 覆盖这些语义导致 external restore 会读取的每个 chunk:
1. offload lookup 覆盖从 retained interval 导出(含 partial-chunk 对齐);同一 interval 也用于 in-flight-load 冲突检查与相关 recency touch。
2. 同步扩展 producer 的 offload store mask 以供给该 interval——只改 lookup 窗口会把受影响的 restore 变成 miss,因为 producer 仍可能省略必需 chunk。
3. 保持现有 outcome:missing 必需 chunk 缩短/拒绝 hit,pending 必需 chunk 推迟 lookup,完整 hit 正常分配并加载;保留现有 `prepare_load` 断言。
4. 模型实际 attention window、wire format、cache keys 与 transfer ownership 均不变;内部正确性契约不需要新用户选项。
不变量:「每个被 admitted restore 消费的 chunk 必须被 confirmed lookup 覆盖;每个必需 chunk 必须经对应 producer store policy 可达」。诊断实验(仅把 offload coverage descriptor 从 window 32 扩到 35、保留真实 allocator 的 `window=32, extra=3` spec)使 mask 变为 `[False, True, True, True]`,12 个用例全部符合预期;该描述符替换仅是实验,不是生产实现建议。

## 优化点与收益
属正确性修复,正文未给出性能提升数字;并明确指出更宽覆盖有成本——本例需 store/load 三个 SWA chunk 而非两个,「needs measurement, not an assumption that extra history is free」。

## 现状与落地
仍 open。核心待决:原生 offload 应恢复整个 retained interval(本提案),还是 initial external-hit allocation 区分 retention-only 历史与恢复所需历史(后者须另证 MTP replay 与缓存发布正确性)。model/performance 验证尚未运行、也未声称完成。feedback period 一周。

## 关联
- 检视基线 commit fa1b3b1922ccb2d7a597925765c4bec7f358d748
- #55818(滑窗分配 scaling,保留 extra retention)
- #52287(EAGLE volatile-tail pop 的 store 窗口)
- #56486(invalid suffix 遗留 pending flag)
- #54914(DSpark 下类似 load 断言,机制未证实相同)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
