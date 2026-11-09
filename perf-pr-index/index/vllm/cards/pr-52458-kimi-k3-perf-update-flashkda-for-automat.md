# PR #52458: [Kimi-K3][Perf] Update FlashKDA for automatic K2 V-split

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @BabyDrangoner |
| 合并 | 2026-08-17 |
| 分类 | Model-specific Fast Path（tier: major） |
| Label | ci/build, kimi, k3 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/52458 |

## 一句话结论

升级 FlashKDA pin 启用自动 K2 V-split，H=12,N=1,T=4096 完整 forward 在 PRO 6000 提速 1.434x、H20 提速 1.613x。

## 动机

本 PR 仅把 pinned FlashKDA revision 从 `b5d1101` 升到 `053de1b`（一行 CMake）。新 revision 在 `2 * local_heads * sequences <= SM count` 时自动拆分 K2 的 value dimension，提升低并行度 workload 的 CTA 级并行；对被 heuristic 拒绝的 shape 保留原 K2 path，并带入上游 TMA proxy-fence fixes。无 vLLM API 或 model-code 改动，作者检索 open PRs / issues 的 exact revision 与 FlashKDA V-split 关键词后未发现重复。

## 优化点

新 revision 依据 GPU 的 SM count 做 dispatch：当 `2 * local_heads * sequences <= SM count` 时自动 split K2 的 value dimension 以增加 CTA 级并行；H=12 时 188-SM 的 RTX PRO 6000 选中 `N <= 7`、78-SM 的 H20 选中 `N <= 3`，两平台因此使用不同的选中 workload 集；被 heuristic 拒绝的 shape 仍走原有 K2 path，另含上游 TMA proxy-fence fixes。改动只有 `flashkda.cmake` 的一行 `GIT_TAG`，无 vLLM API 或 model-code 变更。

## 关键改动

```text
cmake/external_projects/flashkda.cmake: 把 FlashKDA 的 GIT_TAG 由 b5d11010ff01c1d4a683c0dde42e76cbeaa8107f 升到 053de1b716ef3255873e02d2d28f4adf09951978（本 PR 唯一改动文件，无 vLLM 侧代码变更）
```

## 性能声明

测试条件：Kimi-K3 的 TP8 rank-local geometry（`H=12`, `D=128`, production beta row stride 6288），为 SM120 与 SM90a 用 vLLM production stable-ABI flags 构建两个确切 revision，测量完整 FlashKDA forward（beta materialization、K1、K2），非 isolated K2；main 对比用 old-new-new-old schedule，每次采样前 512 MiB L2 scrub，每 revision 两个 60-sample slots；平台为 RTX PRO 6000（SM120，188 SM）与 NVIDIA H20（SM90，78 SM）。Selected `H=12` prefill/mixed cases（各 4 个、paired wins 120/120 per case）：PRO 6000 eager 1.202x-1.434x、CUDA Graph 1.198x-1.433x；H20 eager 1.453x-1.613x、CUDA Graph 1.501x-1.652x。两 GPU 都选中的 workload `H=12, N=1, T=4096`：PRO 6000 1.434x eager / 1.433x graph，H20 1.613x eager / 1.652x graph。短序列 sweep：PRO 6000 全部 12 个选中用例 +5.576%-10.571%，H20 全部 12 个 +9.704%-17.905%。被 heuristic 拒绝的长 H20 用例约不变（0.999x-1.002x），极短被拒用例 synchronized wall time 约 ±1%、最慢约 1.0% 更慢。PR 注明这些是 operator-level 测量而非 full-model throughput；引用的上游 full Kimi-K3 TEP8 on GB300 评测为 GSM8K 96.47%→96.82%、OCRBench 88.8%→90.0%，runtime 1075.8→1044.3 s 与 610.1→596.5 s。

## 验证与回归风险

正确性：`tests/models/kimi_k3/test_kda.py` 在 RTX PRO 6000 新 pin 下 51 passed、在 H20 两个 pin 下各 51 passed；old/new 输出与 final recurrent states 在长/短序列 eager 与 CUDA Graph 矩阵上 bitwise identical（含 deterministic replay、empty-sequence checks）；H20 全部 133 个 workload shapes（含 76 个选中 V-split）两 pin bitwise-identical，10 个 CUDA Graph workloads 20 replays bitwise deterministic 且匹配 eager，20 个空序列用例保持 final-state 与相邻序列语义；本地构建的 old SM90a 扩展与 wheel 当前 pin 在 6 个代表用例 bitwise 匹配；pre-commit hooks 与 `git diff --check` 通过。风险：被 heuristic 拒绝的长/极短 shape 有约 ±1% 波动、最慢约 1.0% 更慢；每次 eager 调用查询 `cudaDeviceGetAttribute` 约 0.35 us 的 host 开销（CUDA Graph replay 不付出该成本，PR 把缓存该属性列为可能的 upstream follow-up）。

## 关联

无（PR 正文未给出其他 vLLM PR 编号线索，仅引用上游 FlashKDA 仓库评测链接并说明未发现重复）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
