# PR #63891: [ROCm][Perf][DSv4.1] Run DSpark's fused Markov sampling on gfx950 with --monokernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @okorzh-amd |
| 合并 | 2026-11-05 |
| 分类 | Attention / Kernel（tier: major） |
| Label | rocm, ready, ci/build, deepseek, dflash, DSv4.1 |
| Diff | diffstat 缺失（素材记录为 +None/-None）；diff 含 6 个文件（ops/__init__.py、fused_markov.py、swa_decode_meta.py、两个测试文件，约 +110/-30 行） |
| 链接 | https://github.com/vllm-project/vllm/pull/63891 |

## 一句话结论

把 #60731 的 DSpark fused Markov sampler（每 draft 位置 1 次 launch）从 gfx942 放开到 gfx950（MI355X），DeepSeek-V4.1-Flash TP4 c1 decode step 提速 2.36%。

## 动机

`fused_markov` 此前只跑在 gfx942：#60731 的 gfx942 移植把所有 helper 用 `ops.enabled()`（gfx942-only）把关。gfx950 上 DSpark decode step 的 Markov 采样退回 vLLM 的算子循环，k=6 时每步约 51 次 kernel launch。

## 优化点

纯开关/门控改动，kernel 本体不动（fused_markov 是基于 vLLM 自有 Gumbel helper 的普通 Triton，无 gfx942 专属依赖）：(1) 新增共享 gate `ops.monokernel_cdna3_or_4()`（`--monokernel` + ROCm + CDNA 版本 3 或 4），`fused_markov` 与 #63885 的 `swa_decode_meta` 都改用它，其余 helper 保持 `ops.enabled()`；(2) gfx950 行数上限 `MAX_ROWS_GFX950 = 5`——microbench 显示 6 行起 vLLM 循环更快（其 Markov GEMV 在 n>5 时离开 wvSplitK 走通用 GEMM，而 fused launch 每行约 +54 µs），`_why_not` 按 `get_cdna_version()` 选 5 或 8 行上限，超过则回退 vLLM 循环；(3) 两架构统一以 4 warps 启动 step kernel（gfx950 上 4 warps 比 8 warps 快 2–5%，1–2 行时差 2% 内），BLOCK 不变；(4) FULL CUDA graph 下按每个 padded batch size 在 capture 时决定路径。无新 flag/env、无 speculator/model-runner/CPU-GPU sync 改动；gfx942 行为按构造不变（此时 `monokernel_cdna3_or_4()` 与 `enabled()` 等价）。

## 关键改动

```text
vllm/models/deepseek_v41/amd/ops/__init__.py: swa_decode_meta_enabled 重命名为 monokernel_cdna3_or_4（CDNA3/CDNA4 共享 gate）
vllm/models/deepseek_v41/amd/ops/fused_markov.py: gate 改 monokernel_cdna3_or_4；新增 MAX_ROWS_GFX950=5 行数上限（按 CDNA 版本选取）
vllm/models/deepseek_v41/amd/ops/swa_decode_meta.py: gate 同步替换
tests/models/test_dsv41_mono_decode.py: gate 测试参数化为 (cdna, monokernel, rows, taken) 四场景，stub kernel 驱动真实 gate
```

行数上限选择：

```python
max_rows = MAX_ROWS_GFX950 if get_cdna_version() == 4 else MAX_ROWS
if not 1 <= num_reqs <= max_rows:
    return f"{num_reqs} requests"
```

## 性能声明

微基准（MI355X 单卡，DSv4.1-Flash Markov head，vocab 129280/rank 256，k=6 全 6 个 draft 位置 = 7 次 fused launch，graph replay 9 轮×200 次中位数）：fused 4 warps 在 1–5 行为 120/178/228/281/337 µs，vLLM 循环为 225/286/339/355/(1127)/275 µs——5 行 fused 337 µs 已低于循环 4 行 355 µs。端到端（MI355X，`main` 335dd77b09 vs PR c128b8ba44，DSv4.1-Flash `--monokernel`、DSpark k=6、AV off、synthetic AL 3.87、random 32k in/1k out、Step=ITL p50，4 台服务器 ABBA 顺序 × 3 runs）：TP4 c1 5.372 → 5.245 ms（**−2.36%**）、c2 −2.10%、c4 −0.98%、c8 +0.02%（8 行 step 走循环，持平）；TP2 c1 −2.27%。逐轮 c1 提速 2.04–2.75%、c2 1.34–2.70%、c4 0.60–1.43%（正文中 2.4%/2.75%/2.70%/1.43%/2.02% 即各轮数据）。launch 数（torch profiler）：每 decode step GPU kernel 365–366 → 321（净 −44/−46），每步执行 7.00–7.01 个 `_markov_*` kernel。精度：GSM8K-1319 与 AL 不变（8 并行 0.9189/0.9156、4 并行 0.9151/0.9151，各 6 pass），数值差异仅在 draft 侧（1-bf16-ULP bias 可换 draft token，rejection sampling 保持 target 分布）。

## 验证与回归风险

gate 测试 4 场景（gfx950 5 行取 fused/6 行取循环、gfx942 8 行取 fused、无 flag 取循环）对 stub kernel 断言 launch 次数与 warp 数；6 个变异体（gate 仅 gfx942、无 gfx950 cap、cap 误用到 gfx942、cap 按 CDNA3 而非 CDNA4、缓存未清、忽略 --monokernel）均被测试击穿；`test_dsv41_gfx942_fused_markov.py` 在 gfx950 4 warps 手动跑 6 passed（main 代码 6 failed）；`test_dsv41_gfx942_decode_meta.py` 3 passed；`test_dsv41_mono_decode.py` 82 passed。无 CI 配置改动（gate 测试走既有 MI355 "Basic Models (Other)" job；无 CI job 在 gfx950 跑 fused launch，靠手动）。风险面：(1) 5/6 行边界是 microbench 拟合值，随 MI355X 驱动/时钟变化可能漂移；(2) 3–5 行的输出正确性仅通过 GSM8K/AL 间接验证（kernel 对拍只在 1–2 行、k=5）；(3) AV（acceptance estimator / confidence head）开启时回退循环，fused 路径覆盖面受限；(4) `use_fp64_gumbel` 两架构均未测试；(5) 与 #61311 的重命名（`_monokernel`→`_specialized_decode`）有落地顺序依赖。

## 关联

前序：#60731（gfx942 移植）、#63885（swa_decode_meta，gate 被本卡替换）。相关：#63181（另一 2-launch/position 融合，合并后本卡 hook 先行、其余场景由它接管）、#63486/#61311（测试文件/CI 迁移与重命名）、#61869（CUDA-only fused Markov sampler，不涉 ROCm）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
