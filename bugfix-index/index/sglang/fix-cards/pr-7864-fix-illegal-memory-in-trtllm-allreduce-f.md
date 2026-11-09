# PR #7864: Fix illegal memory in trtllm allreduce fusion

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7864](https://github.com/sgl-project/sglang/pull/7864) |
| 作者 | BBuf |
| 合并日期 | 2025-07-08 |
| labels | (none) |
| 热度 | 👍 2 · 💬 0 |
| 分类 | Parallelism / Communication |
| bug 类型 | crash |

## 一句话结论
allreduce 融合在 min-latency oneshot 路径上 token 上限 1024 超出内核支持,触发非法内存;上限降到 128 修复。

## 问题现象
按正文的复现步骤,以 DeepSeek-R1-FP4(modelopt_fp4 量化、TP=8)启动服务,同时开启 `--enable-flashinfer-moe`、EAGLE 投机解码与 `--enable-flashinfer-allreduce-fusion`,每次 `flush_cache` 后用 bench_serving 分别以 1/4/16 并发跑 random 数据集(输入/输出长度均为 1000),出现 PR 标题所述的 illegal memory 问题(正文以截图呈现,未附文字日志)。触发条件与大 token batch 相关:融合路径此前允许 token 数最多 1024。

## 根因
正文未直接写明根因,仅引用 flashinfer issue #1223 的讨论作为依据。由 diff 推断:新增注释写明 "According to the discussion in flashinfer#1223 ... We set the max token num to 128 for allreduce fusion with min-latency case(use_oneshot=True)",即 flashinfer 的 oneshot(min-latency)allreduce fusion 内核仅支持最多 128 个 token;而 SGLang 侧此前以 `max_token_num=1024` 初始化 workspace,并在 `hidden_states.shape[0] <= 1024` 时就走融合路径,token 数超过内核实际支持范围时发生非法内存访问。

## 修复方案
把融合路径的 token 上限从 1024 收紧到 128,并同步修正调用命名:
- `python/sglang/srt/layers/communicator.py` `_gather_hidden_states_and_residual`:融合触发条件由 `hidden_states.shape[0] <= 1024` 改为 `<= 128`,并补充指向 flashinfer#1223 讨论的注释;
- `python/sglang/srt/layers/flashinfer_comm_fusion.py`:`ensure_workspace_initialized` 默认 `max_token_num` 1024→128,`flashinfer_allreduce_add_rmsnorm` 改名为 `flashinfer_allreduce_residual_rmsnorm`,其默认 `max_token_num` 同步 1024→128;
- `python/sglang/srt/layers/layernorm.py` `forward_with_allreduce_fusion`:调用点同步使用新函数名。

关键摘录:
```diff
-                and hidden_states.shape[0] <= 1024
+                and hidden_states.shape[0] <= 128
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:改动覆盖所有启用 `enable_flashinfer_allreduce_fusion` 的 TP 推理路径。token 数 >128 的 batch 现在回退到普通 allreduce + 独立 layernorm,不再享受 oneshot 融合的低延迟收益(小 batch 场景不受影响);`flashinfer_allreduce_add_rmsnorm` 改名会波及其他调用点,本次 diff 已同步 layernorm.py 调用处。

## 关联
- https://github.com/flashinfer-ai/flashinfer/issues/1223#issuecomment-3047256465

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
