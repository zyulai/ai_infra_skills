# PR #43077: [AMD][V4.1][*/N] Fix DP attention + TP MoE decode NaN and support bounded replay

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#43077](https://github.com/sgl-project/sglang/pull/43077) |
| 作者 | kkHuang-amd |
| 合并日期 | 2026-10-12 |
| labels | deepseek, run-ci, run-ci-extra |
| 热度 | 👍 0 · 💬 0 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
MI355X 上 DSv4.1 DP attention+TP MoE decode 因 IPC 读到 stale graph-pool 数据而 NaN 崩溃,改为 IPC pool 暂存并清零 pad 行,同时放开 bounded replay。

## 问题现象
在 MI355X 上以 DP attention + TP MoE 运行 DeepSeek-V4.1-Flash decode,开 CUDA graph 时 MoE 输出出现 NaN,几分钟后进程崩溃;此外 decoder SWA bounded replay 在所有 DP attention 布局下都被拒绝启用。修复前 AgentX c32 压测约 6 分钟即 crash。

## 根因
正文 Modifications 披露的机制:
1. CUDA graph capture 期间,aiter custom all-gather/reduce-scatter 直接注册 graph-pool 张量,peers 在 replay 时经 IPC 读到 stale 的 graph-pool 数据(DP attention + TP MoE 下表现为垃圾行/NaN)。
2. DP attention + TP MoE 会把每个 rank 整个 padded chunk 送入 DP gather;graph replay 不刷新 pad 行,diff 注释指出 pad 行持有 stale pool 内存(bf16-max scale),在 MoE 与 combine 中溢出产生 NaN(由 diff 推断,diff 代码注释原文)。

## 修复方案
1. `distributed/parallel_state.py`:新增开关 `SGLANG_AITER_CAPTURE_COPY_IN`(默认 0,DSv4.1 DP attention+TP MoE 设 1),capture 时将 aiter custom all-gather/reduce-scatter 输入经由预注册 IPC pool 暂存(`all_gather_unreg`/`reduce_scatter(registered=False)`),replay 时不再读 stale 数据。
2. DP pad 行掩蔽(ROCm only,`SGLANG_DP_MASK_PAD_ROWS` 默认 true):`decode_cuda_graph_runner.py` 新增 `_set_dp_real_num_tokens`,每次 decode replay 前把真实 per-rank token 数写入 `num_token_non_padded` buffer;`forward_batch_info.py` 的 `enable_num_token_non_padded` 在 DP attention+TP MoE(ROCm)下也返回 True;`deepseek_v4.py` 的 `_run_moe_ffn_dp_sync` 在 DP gather 前用新增 `_mask_dp_pad_rows` 清零 pad 行。
3. bounded replay 放开:ROCm 下,MoE 为 TP(`moe_a2a_backend none`)且 attention TP 为 1 时允许 `--enable-decoder-swa-bounded-replay` + DP attention;`deepseek_v4.py` 新增 `_late_layer_dp_counts`/`_enter_late_layer_dp`/`_exit_late_layer_dp`,eager 前向 all-gather 各 rank late-layer 行数并切换 DP gather 尺寸。CUDA 行为不变(全部门控在 ROCm)。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:aiter 通信路径(默认关闭,仅设 `SGLANG_AITER_CAPTURE_COPY_IN=1` 生效);ROCm DP attention+TP MoE 的 decode replay 与 MoE gather(新增 pad 行清零,可经 `SGLANG_DP_MASK_PAD_ROWS` 关闭);ROCm SWA bounded replay 校验逻辑(仅 TP-MoE+attention-TP1 布局放行,其他 DP 布局仍拒绝)。CUDA 路径无行为变化。验证:GSM8K 5-shot(1319 题)0.896(TP2 无 DP attention 对照 0.894-0.906);AgentX 修复后 0 errors、96,931 tok/s/GPU、P90 70.7,加 bounded replay 后 102,631 tok/s/GPU、P90 88.9。CI:Base/Extra 绿,AMD ROCm 10 红(原帖标注)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
