# PR #46168: [Perf] mooncake: Cut the host time of direct linker layer-wise loads

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @alphabetc1 |
| 合并 | 2026-11-09 |
| 分类 | Speculative Decoding（tier: major） |
| Label | hicache, run-ci, bypass-fail-fast, mooncake, run-ci-nv-extra |
| Diff | +227/-76, 6 files（pack diffstat 缺失，按 pack diff 文本计数） |
| 链接 | https://github.com/sgl-project/sglang/pull/46168 |

## 一句话结论

合并 Mooncake direct linker 的跨 pool get session 并 NumPy 化 range 元数据，480K 前缀分层加载 host 时间 1.4s→0.41s。

## 动机

Mooncake direct linker 的 layer-wise prefix restore 在 host 侧每个 pool 各开一个 get session，并逐行构建 range 元数据；分层加载的 host 时间因此成为恢复外部前缀时的瓶颈。

## 优化点

三处机制改动：(1) 一次 load 把所有 pool 的 keys 合并成单次 `batch_get_session_start`，同时保留 main 已有的分组异步 range 传输；(2) keys 准备移到 GPU slot fence 之前、session start 移到 fence 之后，避免 fence 等待消耗 session lease，location 读取与 range 传输也放到 fence 之后；(3) `finally` 中对每个提交过的 key 结束 session——批量启动部分成功时也要清理（Mooncake session-end 忽略不存在的 key）。range 元数据改用 NumPy：同一层每行 emit 的 (size, offset) 完全相同，改为所有行共享只读列表，ptrs 用仿射式 `rows[:,None]*stride+base` 一次算出；分组组装会 extend 行，因此先把共享元数据拷贝成独立所有权，避免改写共享状态。

## 关键改动

```text
python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py: 跨 pool 批量 session、fence 后置、按提交 key 清理
python/sglang/srt/mem_cache/hybrid_cache/linker_pool_assembler.py: NumPy 仿射 ptrs + 共享只读 size/offset 行
test/registered/unit/mem_cache/test_mooncake_direct_linker_session.py: 新增 session 回归测试（fence 顺序、部分失败清理）
test/registered/unit/mem_cache/test_linker_pool_assembler.py: 布局/行序保持测试（packed 与 unpacked）
```

```python
base = np.array([item[0] for item in items], dtype=np.int64)
stride = np.array([item[1] for item in items], dtype=np.int64)
rows = np.asarray(locations, dtype=np.int64)
ptrs = rows[:, None] * stride + base
row_sizes = [item[2] for item in items]   # 所有行共享同一个只读列表
row_offsets = [item[3] for item in items]
```

## 性能声明

(1) 历史 ABBA 对比（baseline `27ac614f` vs 早期修订 `baf3d6bc`，DSV4.1，TP2/EP2/attention-DP2，MegaMoE、DSPARK block5、32768 new tokens/rank/step）：step 时间 475.028/484.743 → 442.530/438.533 ms，配对增益 7.34%/10.54%，均值比 8.93%；固定 batch engine-step 口径，非应用层吞吐。(2) lease 修复后的 MiniMax-M3.1-Flash TP1 NUMA-bound 对比：fixed 均值仅低 2.72%，小于 7–8% 的版本内波动，作者明示不能证明加速。(3) E2E agentic trace replay（与 #45106 合测、增益不拆分：4×TP4 GB300 + cache-aware router，MiniMax-M3.1-Flash TP4/EP4，AgentX 64 并发 900s）：output tok/s/GPU 23,994→24,891；TTFT p50 0.460→0.396 s、p90 2.299→1.193 s；480K 前缀 L3 reload 的 `load_layer_wise` 墙钟 1.37–1.49 s→0.41–0.42 s，range gets 由 125 次串行（1.31–1.39 s）变为 68 次 4 线程（0.34–0.35 s）。DeepSeek-V4.1-Flash 同设置在噪声内（其压缩/sparse KV 下 529K reload 仅 0.87 GB/rank）。

## 验证与回归风险

21 个 pool/session/grouped-load 测试与 88 个 unified-linker 测试（+30 subtests）通过；新增 session 测试覆盖 fence 后单次跨 pool batch、有序 range、部分启动失败后的清理；assembler 测试证明不拷贝共享 size/offset 行会复现失败。准确性限制：三个确定性 restore-vs-device 检查超过 selected-token logprob 容差（共享 restore 差异未定位）；早期 DSv4.1 restore 测试的 KL<0.03 结论先于 lease 修复，不覆盖新 head。风险集中在 mooncake layer-wise 加载与池元数据共享语义，默认路径行为不变。

## 关联

E2E 与 #45106（`_group_range_meta` 行构建方式调整）合并测量，同区域前序还有共享 `bench_engine_steps.py` 的 #46006。无其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
