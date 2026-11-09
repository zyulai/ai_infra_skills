# RFC #64776: [RFC]: Single Batch Overlap (down GEMM ↔ combine) for DeepEP LL

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#64776](https://github.com/vllm-project/vllm/issues/64776) |
| 作者 | algebra-MCX |
| 创建日期 | 2026-10-30 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | MoE |

## 一句话结论
提议在 DeepEP 低延迟路径上落地 Single Batch Overlap(down GEMM 与 combine 重叠):分两阶段推进且不 pin fork,重启 #34963 的方向。

## 提案动机
SBO 面向对 DBO 不划算的小 decode batch(crossover 见 #64100):单 batch 没有第二个 microbatch 遮挡 all-to-all,SBO 让 DeepEP LL combine 在 down GEMM 完成每个 expert 的行后即开始发送,经共享信号张量协调。此方向 revisit #34963(closed as stale)与原型 PR #29211(因依赖两个未合并 fork 被弃)。当前 vLLM(异步 prepare/finalize,DeepEP LL + NIXL EP)中 down GEMM 完全结束后 combine 才开始;shared-expert 侧的重叠已存在,缺的是 down GEMM ↔ combine send。上游内核现状(2026-10-30 核查):DeepEP main 无 src_signals/signal-aware combine,DeepEP #249/#388 与 DeepGEMM #183 均开放但相互冲突,SGLang 靠自带 fork 实现。

## 提出的方案
- Phase 0(内核,先决条件,在 vLLM 之外):推动 signal-aware LL combine 与 signal-producing masked FP8 grouped GEMM 上游落地(DeepEP #249/#388、DeepGEMM #183 或后继);上游发布可用版本前 vLLM 不开始合并
- Phase 1(vLLM,opt-in,Hopper,DeepEP LL + DeepGEMM masked FP8):
  1. 能力检测、不 pin fork:仅当已装 DeepEP/DeepGEMM 暴露信号参数时启用,否则保持原路径;opt-in flag,默认关闭
  2. 每层从既有 workspace 分配归零 int32 信号张量(num_local_experts × ceil(max_tokens_per_rank/block_m)),可图捕获,无每步 torch.zeros
  3. down GEMM 用 num_sms − comm_sms 个 SM,每完成一个 block 递增信号;combine 在现有 aux stream 上以 overlap=True 启动,等 start event 后发送信号达标的行
  4. shared experts 移入 dispatch 窗口(dispatch 发送与 receive hook 之间),因 combine 窗口被 down GEMM 占用
  5. DBO 互斥:一步只用 DBO 或 SBO;首版在 microbatched 时禁用 SBO
- Phase 1 范围外:NIXL EP(1.5 无信号参数)、FlashInfer CuteDSL/Blackwell(#34963 路径)、DeepEP HT/v2、BF16 masked GEMM
- 完成标准含:DeepSeek-V3/R1 FP8 DP/EP decode on H200 上 SBO 开关输出/logprobs 一致、FULL CUDA graph 捕获与回放、bench serve TPOT/吞吐及显示重叠的 trace

## 优化点与收益
正文未给出量化数字;正文明确目前没有实现、GPU 运行或性能结果,目标是让 combine 发送与 down GEMM 重叠以缩短小 batch decode 的关键路径。

## 现状与落地
🟢 open,反馈期两周,实现无论如何等待 Phase 0 内核;向维护者提出三个问题(能力检测式 opt-in 是否可接受、等 DeepEP/DeepGEMM main 还是接受 released tag、SBO 开启时 shared experts 放 dispatch 窗口是否合适);正文说明调查与起草使用了 AI 辅助。

## 关联
- #34963、PR #29211、#64100
- DeepEP #249/#388、DeepGEMM #183

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
