# PR #61019: [Bugfix][KV Connector] Serve and store OffloadingConnector partial tails with EAGLE/MTP groups

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @kimseunghyun-kr |
| 合并 | 2026-10-25 |
| 分类 | Speculative Decoding |
| Label | bug, ready, kv-connector |
| Diff | diffstat 缺失(pack 未提供);diff 可用,`offloading/scheduler.py` +161/-32(正文注明,无核心文件改动)、测试 +400/-1 |
| 链接 | https://github.com/vllm-project/vllm/pull/61019 |

## 一句话结论

修复 EAGLE/MTP + mamba align 下 OffloadingConnector 部分尾块从不 serve/store 的问题,CPU 层命中不再停在整块边界,MLPerf agentic SLO pass 91.70% → 92.42%。

## 动机

带 EAGLE/MTP draft 且 `mamba_cache_mode="align"` 时,原生 CPU OffloadingConnector 的 CPU 层命中只能停在 scheduler block 边界,其后的 token 每轮重算。两个根因:(1) lookup——`supports_partial_tail` 因 EAGLE 组直接为 False,complete-chunk 查找为 EAGLE 多查一个 chunk 再丢弃,命中比最后存储 chunk 少一块且从不尝试 partial tail;(2) store gate——CoW hand-off 到达的步 `num_computed_tokens == T`,gate 判 `T + h > T` 恒真,唯一能存尾块的那一步永远被跳过。

## 优化点

三部分:(1) 从 `supports_partial_tail` 移除 EAGLE 排除;EAGLE 非递归组的 partial-tail 键改为 `boundary + eagle_proof_margin`(与 core #60533 的对齐规则一致:细粒度命中时一个 hash unit,否则一个块),Mamba 与非 EAGLE 组仍按 boundary;(2) 修 store gate:只在 proof token 由本步计算时存(`min(num_computed_tokens + num_scheduled_tokens, num_prompt_tokens)` 封顶,防投机 token 计入),partial 搜索加宽一块并用 bridge key(被弹出的 chunk 每次查找只 probe 一次);(3) store 策略——EAGLE 组只对从 CPU 层 load 过的请求(`loaded_from_offload`)存尾块,否则 parts 1+2 会使 GPU→CPU 流量翻倍(A100 上排队的大块 store 会按序占用 copy engine,延迟每步输出)。无 EAGLE 组、DCP>1、blocks_per_chunk>1、混合块大小、SWA 组行为不变,无新配置项。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py: 三处修复(lookup keying/store gate/store policy)
tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py: +400 行 EAGLE partial tail 测试
```

```python
+            if self._max_eagle_proof_margin and not req_status.loaded_from_offload:
+                # Requests that did not load from the offload tier keep their
+                # EAGLE tail in the GPU prefix cache; skip the store.
+                continue
...
+                if boundary + self._max_eagle_proof_margin > prompt_tokens_after_step:
+                    # The EAGLE proof is not computed by the end of this step.
+                    continue
```

## 性能声明

8x A100-SXM4-40GB,Qwen3.8-Flash-Next-FP8(36 GDN + 12 FA + 1 MTP 层),TP8+EP、MTP 3、block 800 tokens、`prefix_match_unit` 16、native offloading 256 GB,MLPerf agentic 多轮、20 并发会话、3x60 min/臂:

- SLO pass(人工分析口径:P95 TTFT≤2.93s 且 P95 TPOT≤21.74ms):without 91.70% → 本 PR **92.42%**;goodput 1.866 → 1.922 req/s;CPU→GPU load 19.9 → 16.0 TB;GPU prefix hit 79.0% → 85.2%。作者明确只 claim "no regression"(+0.7 点在两次重复波动内)。
- 同日对照(单次 60 min/臂):store gate 使每请求 GPU→CPU 存储 247 → 126 MB,完成请求 7,450 → 7,721,TTFT P95 2,297 → 2,025 ms(-12%),TPOT P95 23.41 → 22.56 ms(-4%)。
- 反例证据:只上 parts 1+2(无 store 策略)时 SLO pass 掉到 86.6%、P95 TTFT 比 without 高约 13%(即 part 3 存在的必要性);GPU→CPU store 3.00 → 6.93 TB。

## 验证与回归风险

单元测试覆盖 proof margin 规则、store/lookup 键位、round-trip、proof 必须在 prompt 内(5 例)、popped chunk 只查一次(6/7 次 vs 7/9 次)、store 策略两种情形;A100 上 `offloading_connector/` 全目录 589 通过,macOS CPU 603 通过,ruff 干净。风险面:hand-off 步若在 `T+h` 前停止则尾块不存(多模块 MTP 损失 0.2-0.45%、块超 token 预算 ≤1.7%,修复需 core 配合、未做);store 策略是启发式——GPU 前缀命中过的请求不存 EAGLE 尾块,GPU 逐出后下一轮才能部分命中;旧 CPU 缓存无 EAGLE 尾块,无需迁移。E2E 收益幅度小且作者自述在噪声范围内。

## 关联

依赖/对齐:#60533(core 的 EAGLE proof 对齐)、#60695(Mooncake 侧同类修复,键粒度不同)、#60533/#52832/#54165/#61375(相关或同文件其他 bug)。同批 KV offloading:#38460(批量 cuMemcpyBatchAsync swap)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
