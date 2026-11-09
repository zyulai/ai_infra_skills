# PR #36094: [AMD][DSV4] perf: retune decode split-K heuristic for MI355X

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @karverma-amd |
| 合并 | 2026-08-28 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | run-ci, jit-kernel |
| Diff | pack 未提供 diffstat(diff 完整可见,改 1 个文件 + 新增 1 个测试文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/36094 |

## 一句话结论

MI355X 上 DSv4 decode split-K 启发式重调参(target_wg 1.5 / cap 16),生产形状 kernel 55.7µs 降到 44.1µs(-20.8%)。

## 动机

`_kv_splits_heuristic` 决定 DSv4 decode attention 启动多少个 KV split,它在整个 decode 区间恰好多分了一个 2 的幂:在 `H=128`/`block_h=64` 时,对 `T=32/64/128` 选了 8/4/2 个 split,而实测 4/2/1 更快。Split-K 只在基础 grid 未填满 GPU 时才有收益;每个额外 split 都增加一次 partial buffer 写与 reduce kernel 的工作量,当每 split 的 K 变短时这些开销不再被摊薄。

## 优化点

改动极小:一个文件里的两个常量——`target_wg_per_cu` 2.0 → 1.5、`_MAX_KV_SPLITS` 64 → 16——且都按平台分流(`is_hip()` 时用新值,CUDA 保持 2.0/64 未测不动),无 kernel 改动、无新代码路径。调参方法:在 MI355X 上扫 `T` in {1..256} x `kv_len` in {128,512,1024} @ `H=128`,以离每形状最优的距离(regret)打分。残余 3.7% regret 是 CUDAGraph 安全性的代价:每形状最优依赖 per-token K,而捕获时不可知,所以启发式只能用捕获期标量。精度方面 selection 逻辑不变,结果仅因 split reduction 的 bf16 重结合可能有 bit 级差异。

## 关键改动

```text
python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_decode.py: _MAX_KV_SPLITS 64->16、target_wg_per_cu 2.0->1.5(仅 HIP)
test/registered/unit/layers/test_dsv4_kv_splits_heuristic.py: 新增 38 用例,锁定契约而非常量
```

```python
# Split-K heuristic constants. The (1.5, 16) pair is tuned for MI355X (see
# _kv_splits_heuristic); CUDA is unmeasured here and keeps the prior (2.0, 64).
_is_hip = is_hip()
_MAX_KV_SPLITS = 16 if _is_hip else 64  # Hard cap on kv_splits.
_TARGET_WG_PER_CU = 1.5 if _is_hip else 2.0
```

## 性能声明

Kernel 级(MI355X,生产形状 `T=32`、`top-k 1024`、`H=128`):decode attention 55.7µs → 44.1µs,**20.8% 提升**。启发式 sweep:(2.0, 64) geomean regret 33.5%/worst 119%;(1.5, 16) **3.7%/36%**。

E2E(DeepSeek-V4-Pro,MI355X gfx950,8192 in / 1024 out,TP8 + DP8 attention + EP8/MoRI + EAGLE MTP,`bench_serving` random 数据集,每格 1 次运行):

| conc | tput base | tput PR | tput | TPOT base | TPOT PR | TPOT |
|---|---|---|---|---|---|---|
| 4 | 860.07 | 873.28 | +1.5% | 39.79 | 39.11 | -1.7% |
| 8 | 1508.46 | 1545.51 | +2.5% | 42.92 | 42.17 | -1.8% |
| 16 | 2590.30 | 2610.88 | +0.8% | 50.02 | 49.62 | -0.8% |
| 32 | 4001.04 | 4073.41 | +1.8% | 65.30 | 64.24 | -1.6% |
| 64 | 5539.24 | 5776.29 | +4.3% | 96.41 | 92.41 | -4.2% |

效果特定于 DP attention:plain TP8(每 rank 16 头)同版本测得平坦(delta <1%、符号交替);DP attention 下每 rank 驻留全部 128 头,正是调参形状。增益随并发增大,因为每 rank token 更多时 decode attention 占比更大。

## 验证与回归风险

精度:GSM8K 1319 题(parallel 1319)两次运行 0.958 与 0.946,invalid 0.000。新增 38 个契约测试:结果必须是 cap 内的正 2 的幂(reduce kernel 按 pow2 stride 索引 partial buffer,坏值会污染结果而非仅变慢)、split 数不随 token 数上升、grid 饱和时不 split、CUDAGraph 安全(只依赖捕获期标量、不读 tensor)、一张明确标注的表锁定 6 个调优操作点(为未来架构重调参只需改该表)。回归面:ROCm-only 的 dsv4 unified_kv 后端;CUDA 路径常量保持不变;CI 在 PR 时点 Base/Extra 为失败、AMD ROCm 7.2 待运行(pack 记录)。

## 关联

同区域:DSv4 attention 后端(`--attention-backend dsv4`)与 unified_kv Triton decode kernel(`SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`);pack 正文未列出关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
