# PR #38936: [Fix] Disable NCCL graph buffer registration for the TP LM-head all-to-all (pure-DP decode hang under request bursts)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#38936](https://github.com/sgl-project/sglang/pull/38936) |
| 作者 | hanming-lu |
| 合并日期 | 2026-09-11 |
| labels | run-ci, bypass-fastfail |
| 热度 | 👍 0 · 💬 3 |
| 分类 | MoE |
| bug 类型 | hang |

## 一句话结论
pure-DP decode 节点 graph-captured 的 TP LM-head all-to-all 在 NCCL 默认 graph buffer 注册下,请求突发时各 DP rank 步进不同步,某 rank 卡死在 `ncclDevKernel_SendRecv` 导致全 gang 挂死重启;强制 `NCCL_GRAPH_REGISTER=0` 修复。

## 问题现象
pure-DP PD decode 节点(`tp_size == dp_size` 且开 DP attention)自动启用 TP LM-head all-to-all(捕获进 decode CUDA graph 的 PyNccl send/recv)。当突发的大量新请求让各 DP rank 以不同速率爬坡(ramp)时,首次突发即挂死:一个 rank 离开当前 step,其余 rank 在 `ncclDevKernel_SendRecv` 里永远空转,scheduler watchdog 在每个 rank 上触发,整个 gang 重启。复现环境:tp4/dp4/ep4 unified launch + `--enable-tp-lm-head-all-to-all`(从 idle 起约 240 请求的首次突发,每次启动必挂),以及多节点 tp16/dp16/ep16 PD decode 部署。`--enable-dp-lm-head`(避开 all-to-all)不挂。cuda-gdb 显示挂死时三个 rank 停在 `ncclDevKernel_SendRecv`,第四个 rank 已进入下一步的首个跨 rank MoE kernel;各 rank Python 层决策一致,分歧发生在注册的交换内部。

## 根因
正文披露:NCCL 默认开启 graph-capture buffer 注册(`NCCL_GRAPH_REGISTER=1`),会把捕获进 CUDA graph 的每个 collective 的 send/recv buffer 注册为图生命周期的零拷贝通道;TP LM-head all-to-all 捕获在 decode graph 的 graph-pool 临时张量上,而这些地址会被 memory pool 复用给其他张量,注册的交换在此情况下无法存活,突发流量下死锁。实验矩阵支持该判断:仅改 MoE backend、专用 non-pool all-to-all buffer 都仍挂,`NCCL_GRAPH_REGISTER=0`(单独或连同 `NCCL_LOCAL_REGISTER=0`)5/5 通过,plain gather 路径(关 all-to-all)5/5 通过。正文同时明确:NCCL 侧确切机制是从实验矩阵推断,非来自 NCCL 内部。

## 修复方案
改 `python/sglang/srt/arg_groups/parallel_hook.py` 的 `handle_data_parallelism`:当解析结果 `enable_tp_lm_head_all_to_all` 为真时,调用新增的 `_disable_nccl_graph_buffer_registration()`,用 `os.environ.setdefault("NCCL_GRAPH_REGISTER", "0")` 设默认值——用户显式设置优先,此时打 warning「the graph-captured TP LM-head all-to-all can deadlock with registered buffers」。须在 scheduler 创建 NCCL communicator 之前运行,communicator 继承该环境。范围仅限 all-to-all 路径:`attn_tp > 1` 形状不会启用 all-to-all,`--enable-dp-lm-head` 与其互斥,eager NCCL op 不受影响;只有 decode graph 捕获的两个 collective(LM-head all-to-all 及喂它的 DP gather)改走 NCCL staging buffer 而非注册零拷贝。新增 `TestTpLmHeadAllToAllNcclGraphRegister` 三个单测:pure-DP decode 设 env、显式值保留并告警、unified serving / `--enable-dp-lm-head` 不动 env。

```diff
+    if resolving_view(server_args).enable_tp_lm_head_all_to_all:
+        _disable_nccl_graph_buffer_registration()
```

验证:tp4/dp4/ep4 上本 PR 默认 env 10/10 轮突发(240-256 并发)全部通过、全程 health 200、无 watchdog;tp4/dp2/ep4(all-to-all 不启用)env 不动、10/10 通过,行为无变化。单测 3 个通过。

## 影响与回归风险
- 影响窗口:正文未说明;受影响配置为启用 TP LM-head all-to-all 的 pure-DP decode(`tp_size == dp_size` + DP attention)。
- 回归面:仅 `parallel_hook.py` 的启动期环境变量设置与新增单测。all-to-all 启用时 NCCL graph 注册被关闭,graph 内两个 collective 改走 staging buffer,可能带来少量带宽/延迟开销(正文未量化);显式设置 `NCCL_GRAPH_REGISTER` 的用户仍可覆盖(并收到死锁警告);不启用 all-to-all 的所有路径 env 不变。

## 关联
正文未列出 issue/PR 链接(正文给出原始 commit `53b0218449`、`ca3830e652`)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
