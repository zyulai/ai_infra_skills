# PR #50045: [KV Offloading] Back-pressure detection and remediation

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @bnellnm |
| 合并 | 2026-09-18 |
| 分类 | KV Cache / Memory \| tier: major |
| Label | ready, v1, kv-connector |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注 3 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/50045 |

## 一句话结论

KV offload 二级 tier 用 EMA 检测背压，EMA 超 1.0s 时丢弃 store、低于 0.5s 才恢复。

## 动机

PR 正文 Purpose 说明，当 secondary tier（disk、shared storage、P2P）变慢时，KV cache tiering offloading manager 仍会不断把新的 store 级联到该 tier，造成 unbounded job pileup，进而劣化 serving latency。因此需要在 tier 饱和时检测并停止向该 tier 级联 store，直到其恢复。正文标注 associated RFC（issue #50031）。

## 优化点

正文 Design 分三部分。Detection：对每个 secondary tier 跟踪 store completion latency 的 EMA（指数移动平均），EMA 超过 high-water mark（1.0s）时把 tier 标记为 under pressure，跌破 low-water mark（0.5s）时清除，hysteresis band 防止震荡。Response：对 pressured tier 的 store 静默丢弃，block 仍留在 CPU primary tier 并按正常 LRU 淘汰；来自 pressured tier 的 load 照常进行。Observability：每 tier 新增两个 Prometheus 指标 `vllm:kv_offload_tiering_backpressure_active`（gauge，压力激活为 1）与 `vllm:kv_offload_tiering_backpressure_stores_dropped`（counter，被丢弃的 store 数）。

## 关键改动

```text
vllm/v1/kv_offload/tiering/backpressure.py: 新增抽象 BackpressureDetector 与 BackpressurePolicy API，实现 EMA 检测、store skipping 子类（diff 中新建）
vllm/v1/kv_offload/tiering/manager.py: 新增 BackpressureState 与 per-tier 跟踪，_process_finished_jobs() 喂 EMA，complete_store() 与 request-level 级联按压力门控，get_stats() 上报指标，reset_cache() 重置状态
vllm/v1/kv_offload/tiering/base.py: JobMetadata 增加 submit_time 字段用于时延测量，新增 BACKPRESSURE_ACTIVE / BACKPRESSURE_STORES_DROPPED 指标 key
tests/v1/kv_offload/tiering/test_backpressure.py: 新测试套件（12 个测试）
```

```python
class DropStorePolicy(DropAccountingPolicy):
    """Silently drop stores to pressured tiers."""

    def should_store(self, detector) -> bool:
        return not detector.is_under_pressure()
```

（摘自 `vllm/v1/kv_offload/tiering/backpressure.py`：压力下丢弃 store 的策略。）

## 性能声明

PR 正文未给出定量性能数据。正文仅给出 "Benchmark for Qwen with various tiering configs" 的 `backpressure-report.html` 附件链接，无表格数字，pack 的 perf claims regex 也未检测到；PR 未注明测试条件。

## 验证与回归风险

PR 引入/依赖的测试：Test plan 用 `DelayedSecondaryTierManager` 悬停已完成 job 以精确控制表观 store latency，12 个测试覆盖 EMA 更新、高压激活（慢 store 5x 阈值）、低压清除、hysteresis 防震荡、压力下 store 跳过与恢复、dropped 计数、get_stats 指标、gauge 归零、reset_cache 清状态、request-level tier 遵守背压、压力下 loads 仍返回 RETRY 并提升；命令 `pytest -sv tests/v1/kv_offload/tiering/test_backpressure.py`。PR 未注明回归风险；改动集中在 secondary tier 的 store 级联门控与 KV offload 指标，diff 中 `ThrottledDropPolicy` 注释指出丢弃 store 会影响 cache miss 与 recomputation/ITL，而保护 TTFT；loads 不受影响，其他硬件/模型路径正文未说明。

## 关联

正文标注 "See associated RFC"：https://github.com/vllm-project/vllm/issues/50031（issue #50031）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
