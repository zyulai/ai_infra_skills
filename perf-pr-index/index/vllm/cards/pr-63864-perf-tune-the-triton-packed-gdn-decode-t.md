# PR #63864: [Perf] Tune the Triton packed GDN decode tile on SM90

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Rockyeast |
| 合并 | 2026-10-27 |
| 分类 | Attention / Kernel \| tier: major |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/63864 |

## 一句话结论

SM90 上 FP32 state 的 packed GDN decode 改用 BV=4，H200 kernel 快 1.02–2.16x、几何均值 1.34x。

## 动机

正文说明本 PR 为 SM90（H100/H200）上 recurrent state 为 FP32 时的 Triton packed GDN decode kernel 采用 `BV = 4`，是 #61355（SM100）的 SM90 对应版本；kernel 本身未改，只增加一个 dispatch 分支。pack 内 microbenchmark 脚本注释显示原 wrapper 固定 `BV = min(next_power_of_2(V), 32)`。正文未进一步描述此前的瓶颈。

## 优化点

在 SM90 且 recurrent state 为 FP32 时把 packed decode 的 value tile 从 `BV = 32` 降到 `BV = 4`，kernel 不变、只加一条按 state dtype 与 device capability 分派的分支，无新增 host synchronization。该 rule 与 batch 无关，SM90 FP32 state 下只有一个 tile 变体可达，graph capture 直接编译它，无需按 batch 预热。BF16 state 保持 `BV = 32`：正文给出的理由是 SM90 上无单一规则通吃（最好的规则是 `B*HV <= 96` 用 4、否则 16，几何均值 1.18x，但最多慢 6%，且在 7/99 个点上慢超过 2%）。输出与更新后的 recurrent state 在所有测量点与 `BV = 32` bitwise identical。

## 关键改动

```text
vllm/third_party/flash_linear_attention/ops/fused_recurrent.py: packed decode 增加 SM90 + FP32 state 走 BV=4 的 dispatch 分支
```

## 性能声明

H200、FP32 state、cold L2、CUPTI：在 99 个 `(H, HV, batch)` 点上 `BV = 4` 比 `BV = 32` 快 1.02–2.16x、几何均值 1.34x，且 99 个点上 `BV = 4` 全为最快。kernel 条件：H200 NVL（132 SMs、60 MiB L2）、`K = V = 128`、BF16 qkv、QK L2-norm 开、state 冷（≥ 4× L2 = 240 MiB）、CUPTI（`torch.profiler`）over graph replays、7 轮 shuffle 取中位数、轮间 spread ≤ 1.8%；`BV` ∈ {4,8,16,32} 在 198 个点输出与 state 均 bitwise identical。(16,48) 绝对时间 `BV=32→4`：B=1 6.65→3.94 µs、B=8 19.5→15.2、B=32 58.0→53.9、B=256 412.5→404.1 µs（带宽受限故收益收窄）。端到端：Qwen/Qwen3.8-27B-FP8（48 GDN layers）、vLLM 0.32.0、H200 NVL、`vllm bench serve`、random 1024 in / 128 out、prefix caching off、`max_num_seqs 64`、patched/unpatched 交错各 2 次取均值：median TPOT conc1 10.78→10.65 ms（−1.2%）、median ITL conc8 11.72→11.51 ms（−1.8%）、conc32 14.60→14.45 ms（−1.0%）；output tok/s 86.1/503.7/1085.6 → 86.6/508.4/1088.0（+0.5/+0.9/+0.2%），正文称 throughput within noise，decode latency 低 1.0–1.8%。

## 验证与回归风险

`pytest -q tests/kernels/fla/test_fused_recurrent_packed_decode.py` 报 14 passed（H200 NVL），FP32-state 用例（`K=V=128`）在 SM90 走新分支。正文称输出与 recurrent state bitwise identical、BF16 state unchanged；只在 H200 NVL 实测，分支也覆盖 H100 与 H20（更少 SM、更低带宽）但未测量。CI 的 `kernels/fla` 只在 B200 与 MI355 运行，因此 CI 未覆盖此 SM90 分支，作者注明该测试是本地针对改动在 H200 跑的。正文未列出其他受影响模型或精度路径。

## 关联

正文明确列出 sibling tunings：#61355（SM100，merged，本 PR 的对应来源）、#61823（SM89，open）、#54181（SM120，closed）、#62961（仅改该 kernel 的测试，非重复）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
