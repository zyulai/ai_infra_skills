# RFC #47572: [RFC]: ReplaySSM: cache SSM inputs instead of state for faster standard and speculative decode (Mamba2 + GDN)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47572](https://github.com/vllm-project/vllm/issues/47572) |
| 作者 | Johnny-Liou |
| 创建日期 | 2026-07-03 |
| 状态 | 🟢 open |
| labels | RFC, stale |
| 热度 | 👍 4 · 💬 5 |
| 分类 | Speculative Decoding |

## 一句话结论
提议缓存 SSM 近期输入而非完整状态:标准 decode 每 L 步才写回状态,推测解码回滚降为指针移动,降低内存带宽压力。

## 提案动机
SSM/线性注意力层的 decode 是内存带宽受限的:当前每步都要读写完整递归状态。在混合模型中,SSM 层是短到中等输出长度下的 decode 瓶颈。推测解码还面临回滚难题:SSM 状态更新不可逆,被拒绝的 draft token 无法从状态中"减去",现实现为每个 draft token 保留一份独立状态,带来约 (1 + num_spec)× 的状态内存占用与额外流量,伤害延迟与最大并发。ReplaySSM(Dao AI Lab 与 NVIDIA 合作,TensorRT-LLM 有并行实现 PR #14203)通过缓存输入而非状态同时解决两者。

## 提出的方案
- 保留单一 checkpoint 状态 S0,把近期 SSM 输入缓存进 ring buffer;flush 步(每 L 步一次)把缓存输入折入 checkpoint,其余步直接由 checkpoint + 缓存输入算输出,不物化完整状态。
- 范围:Mamba2 与 Gated DeltaNet(GDN)层;标准与推测解码;新增 Triton kernel 及集成(mixer dispatch、attention-metadata cursors、state-shape sizing);conv 路径复用现有 `causal_conv1d_update` 不变。
- opt-in 配置项(全部默认关):`--use-replayssm`、`--use-replayssm-spec`(需推测解码且 `--mamba-cache-mode none`)、`--replayssm-buffer-len`(默认 16)、`--replayssm-route`(Mamba2 标准解码专用,默认 `output_only`,精度研究用的临时开关);默认关闭时 decode 路径与基线逐字节一致。
- 集成要点:输入缓存与 checkpoint 状态放在现有 paged mamba state 上,复用 paged-block 分配与 block-table 索引;推测解码用两次设备端路由 launch(verify 与 flush),flush 判定与回滚在设备端计算,每个 batch size 一张 CUDA graph。
- 验证:三级正确性检查(kernel 对精确参考、prefill/decode 全序列等价、engine 级 `check_logprobs_close`)+ GSM8K 上的接受长度匹配;完整数字在 draft PR #47576。

## 优化点与收益
标准 decode 整模型每步加速比(baseline/ReplaySSM,fp32 状态,buffer length 16):Nemotron-Nano-4B(Mamba2,H100)bs 1/32/128/512 为 0.99/1.17/1.40/1.55;Qwen3.5-4B(GDN,H100)0.99/1.08/1.21/1.31;Nemotron-Super-120B(Mamba2,B300)0.98/1.10/1.27/1.67;Qwen3.5-122B(GDN,B300)0.98/1.07/1.14/1.20。即大批次下收益最大;正文称标准解码下每 L 步才写回状态约可砍半主导性内存流量。

推测解码端到端吞吐(B,T)=(16,4)、B300 + MTP drafter:对原生推测解码(c/std)Nemotron-Super-120B 为 1.09/1.07/1.88(bs 1/32/256),Qwen3.5-122B 为 0.99/1.19/1.80;对无推测自回归基线(c/AR)分别 2.12/1.90/2.21 与 1.90/1.93/1.85。回滚成本从 O(T) 降到 O(1)(T = num_spec_tokens + 1)。待办:TP 测试、fp16/fp8 低精度状态评估(fp16 checkpoint 重量化频率约为基线的 1/L)。

## 现状与落地
状态 open,已有 draft PR(#47576)实现;计划分三阶段集成:Mamba2 标准解码 -> GDN 标准解码 -> Mamba2/GDN 推测解码。开放问题:分阶段时间线、缓存是否应附加在 mamba state 上还是独立分配。

## 关联
- [draft PR #47576](https://github.com/vllm-project/vllm/pull/47576)(关联编号,以正文提及为准)
- [ReplaySSM 博客](https://dao-lab.ai/blog/2026/replayssm/)
- [TensorRT-LLM PR #14203](https://github.com/NVIDIA/TensorRT-LLM/pull/14203)
- [ReplaySSM 参考仓库](https://github.com/Johnny-Liou/ReplaySSM)
- vLLM feature request #46187

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
