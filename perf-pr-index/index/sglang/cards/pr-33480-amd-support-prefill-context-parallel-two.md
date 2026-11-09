# PR #33480: [AMD] Support prefill context parallel two batch overlap for DeepSeek V4

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @At1a8 |
| 合并 | 2026-08-17 |
| 分类 | Overlap / Scheduler(tier: major) |
| Label | deepseek, run-ci, bypass-fastfail |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及约 9 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/33480 |

## 一句话结论

为 AMD/ROCm DeepSeek V4 DSA prefill CP 启用 TBO:256k 输入吞吐 +7.2%,64k 以上 TTFT 稳定改善约 8–11%。

## 动机

DeepSeek V4 的 prefill context parallelism(CP)把长上下文 prefill 分摊到 attention CP ranks,但围绕 attention 与 MoE 引入每层 CP collectives。此前 TBO 只支持非 CP 的 DP/EP 工作流:`--enable-prefill-cp` 与 `--enable-two-batch-overlap` 在 DP attention 关闭时被直接拒绝;CP 批被 runtime TBO gate 显式排除;既有 TBO operation 只处理 DP/EP 通信,无法调度 CP all-gather/reduce-scatter。目标是在保持 CP-only 精度、不改 CUDA 与非 CP 行为的前提下,用 TBO 把一个 sub-batch 的 CP 通信藏进另一个 sub-batch 的计算里,降低 TTFT、提升长上下文 prefill 吞吐。

## 优化点

新增 HIP-only 的 CP-aware TBO 前向路径(`_forward_layers_tbo_cp`):每个 prefill batch 拆成两个独立 sub-batch,重建 CP metadata、round-robin 拆分 hidden_states/positions/input_ids,重叠执行后从 CP 布局恢复并合并输出。配套 CP 专属 op schedule:一个 sub-batch 的 MoE all-gather(`op_cp_gather`)与另一个 sub-batch 的 attention/expert 计算重叠,reduce-scatter(`op_cp_combine`)与另一个的 MoE 计算重叠,用持久缓冲与 stream event 管理异步通信。新增 `attn_cp_overlap` 重复 communicator(同一组 CP ranks):让通信流上的 collective 与计算流上的 attention collectives 并发,避免从两个 stream 驱动同一 RCCL communicator 导致死锁。新增 split-phase CP all-gather helpers(launch/finish):KV 与 kv_score 的 all-gather 在输入就绪时立即发射、同步推迟到消费前,handle 持有输入防止 allocator 复用;kv_score 只依赖层入口的 x,可在 q/kv projection 之前发射、由 projection 隐藏。TBO padding 对齐 lcm(attention_tp_size, CP align)。门控严格:仅 HIP、DSA prefill CP round-robin split、CP-v1、非 EP TP-MoE、单 stage PP、子批可拆;不满足或过小的批回落原 eager 路径。

## 关键改动

```text
python/sglang/srt/models/deepseek_v4.py: CP+TBO 前向路径、op_cp_* 操作序列与 TBO 门控扩展
python/sglang/srt/batch_overlap/operations_strategy.py: CP 专属 TBO op schedule(use_cp 分支)
python/sglang/srt/distributed/parallel_state.py: attn_cp_overlap 重复 communicator
python/sglang/srt/layers/utils/cp_utils.py: split-phase CP all-gather launch/finish
python/sglang/srt/layers/attention/dsv4/compressor.py: kv_score all-gather 预发射(prelaunch_kv_score)
```

```python
ops = [
    layer.op_mhc_prepare_attn,
    layer.self_attn.op_attn,
    layer.op_mhc_post_attn_pre_mlp,
    layer.op_cp_gather_a,
    operations.YieldOperation(),
    layer.op_cp_gather_b,
    layer.op_cp_moe,
    layer.op_cp_combine_a,
    operations.YieldOperation(),
    layer.op_cp_combine_b,
    layer.op_mhc_postprocess,
]
```

## 性能声明

DeepSeek-V4-Pro,TP8,AMD(`unified_kv_triton` backend,fp8 KV,page-size 256,chunk 51200)。① I/O=8k/1k~256k/1k、conc=8:TBO 收益随输入长度增长,吞吐从 8k 的 -3.6% 到 256k 的 +7.2%(29004→31102),TTFT 在 ≥64k 后稳定约 -11%(256k:27542→24383 ms);② 128k/1k、conc 2→16:吞吐 -1.2%→+4.6%,TTFT 各档一致改善 8–11%;③ 50k/1 纯 prefill:吞吐 +10.1%–13.9%,TTFT -8.2%–-11.9%。accuracy:GSM8K 1319,cp 0.944 vs cp+tbo 0.936。

## 验证与回归风险

新增 AMD 8-GPU DeepSeek V4 Pro FP4 nightly 测试(CP+TBO 启动、1319 并发 GSM8K 断言 >0.92、per-sub-batch CP metadata、并发 communicator 稳定性)。DSpark aux capture 期间跳过 TBO(perf-only 降级,非正确性);非 CP/CUDA 路径与原行为对齐。注意 8k/1k 与低并发下 TBO 吞吐为负收益(-3.6%/-1.2%),收益随输入长度与并发增长。pack 中 CI:Base 通过,Extra 为失败标记。

## 关联

同批索引 #36527 亦含 two-batch overlap 门控逻辑(互斥条件)。正文未引用其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
