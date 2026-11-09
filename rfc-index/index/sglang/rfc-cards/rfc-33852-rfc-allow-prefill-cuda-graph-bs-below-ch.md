# RFC #33852: [RFC] Allow prefill CUDA graph bs below chunked_prefill_size under post-capture KV sizing

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33852](https://github.com/sgl-project/sglang/issues/33852) |
| 作者 | hnyls2002 |
| 创建日期 | 2026-08-06 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
提议放宽 post-capture KV sizing 的启用条件:允许 prefill capture 上限低于 chunked_prefill_size,用激活 headroom 覆盖 eager prefill 而非整体禁用该特性。

## 提案动机
`post_capture_kv_sizing_planned()` 在最大可达 prefill batch 未被 capture 时整体放弃该特性:只要 `--cuda-graph-max-bs-prefill` 低于 `chunked_prefill_size`,特性就静默关闭且无任何日志。两个值本是独立旋钮(`chunked_prefill_size` 是调度上限,prefill `max_bs` 是 capture 覆盖),机制真正需要的是 eager 激活上界而非「无 eager 前向」;decode 侧已接受 `decode_max_bs < running_requests` 并改为提高 reserve。背景:#33847 中 CI 级 `--cuda-graph-max-bs-prefill 1024` 默认值在 H100 级 GPU(`chunked_prefill_size` 默认 8192)上静默关掉了特性。

## 提出的方案
- 把 prefill 条件移出硬性 gate、放到 `eager_decode_gap` 已用的 headroom 路径:保留 `backend == DISABLED` 与 `chunked_prefill_size <= 0` 为硬 gate;当 `max_prefill_buffer_tokens() > max(prefill_cfg.bs)` 时告警并抬高 `headroom_gb` 覆盖 eager prefill 激活峰值
- 无需新估算:`pre_capture_activation_reserve_mb()` 已有按 `chunked_prefill_size` 键控的 prefill 分支;其大显存 GPU 上 10GB floor 使宽松路径收益低于全 capture,但仍优于现状(现状还要再估算 graph 内存 `reserve_for_graph_mb()`)
- 更紧变体(暂缓):只预留 eager prefill 峰值与最大已 capture bucket 峰值之差,需要「激活随 token 数扩展」的模型
- 可独立落地的小项:gate 不应静默,加一条 `logger.warning` 说明两个值

## 优化点与收益
正文未给出性能量化数字(文中 `10GB floor` 为既有 reserve 公式参数,非本 RFC 收益);收益为让「更短的 capture 时间」与「post-capture KV sizing」两个正当目标不再互斥,并消除静默失效。

## 现状与落地
仍开放:提出放宽方案并附可独立合入的 warning 小项,等待社区/维护者评审。

## 关联
- #33847(问题来源)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
