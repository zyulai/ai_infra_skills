# PR #30261: [Spec] Add DSpark: confidence-scheduled speculative decoding

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @sglang-bot |
| 合并 | 2026-07-12 |
| 分类 | Speculative Decoding(tier: major) |
| Label | high priority, amd, deepseek, speculative-decoding, blackwell, npu, run-ci, diffusion, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight |
| Diff | 素材未提供 diffstat(84 个文件因体积原因从 diff 中省略) |
| 链接 | https://github.com/sgl-project/sglang/pull/30261 |

## 一句话结论

新增 DSpark 投机解码:半自回归块 draft + 置信度调度的变长 verify,并附带 SPS 成本表 profiling 工具。

## 动机

素材正文以一句话给出定位:"DSpark speculative decoding — semi-autoregressive block drafting + confidence-scheduled, variable-length verify",即用半自回归的块式 draft 生成候选,再按置信度调度决定每次 verify 的长度,而不是固定 draft token 数。动机的量化论证(对比 EAGLE 等既有算法的收益数字)未包含在素材正文中。

## 优化点

按标题、正文与可见 diff 归纳:(1) 新算法 DSPARK 进入 `--speculative-algorithm` 家族,由 `dspark_components` 组件包实现(含 `dspark_sps.py` 的 `SpsAdditiveCostTable` 成本表);(2) verify 走 ragged(变长)路径,`SGLANG_RAGGED_VERIFY_MODE` 支持 static/compact/cap-accept 等模式,decode CUDA-graph runner 以 token-bucket 为键、与 bs 为键的图并行驱动 ragged verify,拥有独立的 capture/admission 路径;(3) 调度器按置信度决定 verify 预算,依赖 device-to-host 的 confidence-relay staging(放在共享 overlap-utils 模块);(4) 自带 `dspark_sps_profiler.py`:连接已运行的 DSpark 服务器(要求 static 模式、`SGLANG_SIMULATE_ACC_LEN=1.0`、CUDA graph 开启),按 batch_tokens 扫描实测 steps_per_sec,拟合 `SpsCostTable`/`SpsAdditiveCostTable`(支持 `--fracs` 离对角预算钉扎),供调度器查表定预算;(5) DSpark 专属 Triton 内核(正文称暂在 `sglang.kernels` ops 布局之外)与 dsv4 attention-metadata builder 的 Triton 重写(服务所有 dsv4 路径)。

## 关键改动

```text
python/sglang/benchmark/dspark_sps_profiler.py: SPS 成本表测量/拟合 CLI 工具(diff 可见部分,新增)
python/sglang/srt/speculative/dspark_components/: DSpark 组件与 SPS 成本表(84 个文件被省略,由正文与引用推断)
SGLANG_DSPARK_* 环境变量: FAST_KERNEL / FAST_SAMPLING / FP32_LM_HEAD 等性能开关(正文提及)
```

```python
CONVERSION_FORMULA = (
    "batch_tokens = num_running_reqs_per_rank * verify_num_draft_tokens; "
    "steps_per_sec = 1 / median(server-side step_time over aligned steady steps)"
)
STATIC_CONDITIONING_CAVEAT = (
    "Profiled with SGLANG_RAGGED_VERIFY_MODE=static: a verify step of B tokens "
    "comes from B/(gamma+1) requests, the fewest possible for that B. A "
    "compact-mode step with the same B usually spans more requests and reads "
    "more KV history, so the table slightly over-estimates steps_per_sec ..."
)
```

## 性能声明

PR 未注明测试条件:素材正文(regex 检出)不含任何性能数字,也无可引用的 benchmark 表格。SPS 成本表本身是"实测静态模式 verify 步速 → 查表调度"的基础设施,profiler 明确要求在 CUDA graph 开启、simulate_acc_len=1.0 的静态条件下测量,防止 eager 步骤污染表格。

## 验证与回归风险

可见素材未给出 accuracy 数字与具体测试文件清单(84 个文件被省略);正文 CI 快照显示 Base 测试通过、Extra 测试失败。正文以"后续独立 PR"形式列出已知风险面:fa3 backend 会就地改写 `speculative_num_draft_tokens`;dp attention 下 DSpark draft 批次 `can_run_dp_cuda_graph` 未设置,dense-draft 块不回放 CUDA graph(保守,需 GPU 验证);dsv4 metadata builders 重写为 Triton 后,非投机 dsv4 路径的专门性能验证尚未完成;confidence-relay 位于共享 overlap-utils 且 CUDA buffer 懒分配;`SGLANG_DSPARK_*` 命名待整理(发布后改名需走 alias 弃用)。

## 关联

与 DFLASH 共享 `draft_worker_common` 草稿 worker 构建层(正文提及,最终形态未定);属 EAGLE/EAGLE3/NEXTN/STANDALONE/LOOKAHEAD 之外的又一投机解码算法。正文列有多项计划中的后续 PR(重命名 `num_tokens_per_bs`→`num_tokens_per_req`、`speculative_hook.py` 去重、Triton 内核迁入 `sglang.kernels` 等),未给出编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
