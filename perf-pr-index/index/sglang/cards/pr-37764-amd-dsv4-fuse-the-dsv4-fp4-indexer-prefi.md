# PR #37764: [AMD][DSv4] Fuse the DSv4 FP4 indexer prefill-schedule preamble into one kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @heachary |
| 合并 | 2026-09-04 |
| 分类 | Speculative Decoding(tier: major) |
| Label | amd, run-ci, jit-kernel |
| Diff | pack diffstat 缺失;diff 含新文件 fp4_indexer_schedule_hip.py(+365)及 fp4_indexer_hip.py 修改 |
| 链接 | https://github.com/sgl-project/sglang/pull/37764 |

## 一句话结论

把 DSv4 FP4 indexer 的 prefill-schedule preamble 从 33 次 dispatch 融合为 2 次,decode step 时间降约 3%。

## 动机

FP4 indexer(#37353)用一段 torch-op preamble 构建 persistent-grid schedule,每 C4 layer 执行一次;每个 op 只触碰几百个元素,整段纯属 launch latency bound。在 DeepSeek-V4-Pro 128k/1k、conc-64、DP8TP8 MTP trace(MI355X)上实测每 layer 每步 33 次 dispatch、146 us——按新文件 docstring 的说法占全部 GPU kernel 时间的 11.5%,而 logits kernel 本身仅 8.6 us。这 33 个 kernel 可融合为一个。

## 优化点

新增 `fp4_indexer_schedule_hip.py`,把 preamble 收敛为单个 Triton prep kernel 加 AITER 的 `_prefill_cta_info_kernel`,共两次 dispatch(preamble 中唯一真实依赖是 cta_info kernel 需要前缀和)。具体机制:split-factor 搜索原来物化 `[s_max, T]` 张量并算术提取首个可行索引(约 27 个 op 中的 18 个),利用该和随 s 单调非增的性质改为寄存器内二分搜索,结果相同且无中间张量;page-table pad 在单 kernel 内写满全部输出元素,目的地不再需要预清零;`row_to_batch`/`local_starts` 在 kernel 内生成(sglang 始终一行一 query token 覆盖其全窗口);pinned `cta_info` 让 logits kernel 跳过 −inf 预填(长度感知 top-k 只读 `[0, c4_seq_len)`,恒被 CTA 覆盖)。超过 `MAX_FUSED_ROWS=4096` 行时回退 AITER 的 torch preamble——如此宽的 prefill 是 compute-bound,不在乎约 30 次额外 launch。无新 flag,`--enable-deepseek-v4-fp4-indexer` 开启即生效(HIP)。

## 关键改动

```text
python/sglang/kernels/ops/attention/dsv4/fp4_indexer_schedule_hip.py: 新文件,融合的 prefill schedule 构建器(pad + prep + cta_info 两次 dispatch)
python/sglang/kernels/ops/attention/dsv4/fp4_indexer_hip.py: workspace 挂 schedule_buffers;无 workspace 的 fallback 路径改走融合构建
```

```python
# split-factor 二分搜索(替代 [s_max, T] 张量物化)
any_fits = tl.sum((chunks + (s_max - 1)) // s_max) <= P
lo = 1
hi = s_max
for _ in tl.static_range(32):
    searching = lo < hi
    mid = tl.where(searching, (lo + hi) // 2, lo)
    fits = tl.sum((chunks + (mid - 1)) // mid) <= P
    lo = tl.where(searching & (fits == 0), mid + 1, lo)
    hi = tl.where(searching & fits, mid, hi)
```

## 性能声明

- Fixed length isl/osl = 8k/1k,conc=4/64,mtp=on,MI355X(镜像 lmsysorg/sglang-rocm:v0.5.18-rocm720-mi35x-20260902):output throughput conc4 311.66 → 359.43 tok/s(+15.3%),conc64 2592.46 → 2677.55 tok/s(+3.3%);PR 注明 launch-latency-bound kernel 是固定成本,低延迟场景收益更明显。
- Agent-X conc64、mtp=on:per-GPU throughput 20295 → 20752 tok/s,p50 ITL 32.36 → 31.27 ms,mean accept length 2.48 → 2.49,decode step 85.0 → 82.5 ms,约 3% decode step 改善。
- GSM8k(1319 questions,DP8TP8 + MTP + FP4 indexer):0.940 / 0.920 before,0.948 / 0.931 after。

## 验证与回归风险

GSM8k accuracy(1319 题)+ bench_serving 固定长度与 Agent-X 两类负载。风险面:DSv4 FP4 indexer 的 HIP 路径;workspace 无效时的 fallback(DP padding 或 truncated activations)与超过 MAX_FUSED_ROWS 的 AITER 回退路径仍保留,需验证与原 preamble 等价;fallback 路径中 schedule scratch 通过 `fallback_schedule` 保活以覆盖 logits kernel 的读取,生命周期正确性依赖该实现。

## 关联

前序 PR #37353(DSv4 FP4 indexer)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
