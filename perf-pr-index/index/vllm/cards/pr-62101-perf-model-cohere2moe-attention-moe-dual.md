# PR #62101: [Perf][Model] Cohere2MoE: attention/MoE dual stream, single TP all-reduce per layer, fused add + RMSNorm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @XiaochunTong-Cohere |
| 合并 | 2026-10-21 |
| 分类 | Speculative Decoding(classify.py 给出;实际为 Cohere2MoE 模型级解码优化,触及 Eagle3 aux 路径) |
| Label | ready, ci/build, cohere |
| Diff | diffstat 缺失(pack 未提供 +N/-N 与文件数;diff 可用,注明 1 个文件因篇幅省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/62101 |

## 一句话结论

Cohere2MoE 并行块三连改:dual stream 重叠 attention/MoE + 每层单次 all-reduce + fused add+RMSNorm,decode TPOT 提速 1.14–1.36x。

## 动机

Cohere2MoE 的层是并行块:attention 与 MLP/MoE 读同一个 normed 输入,输出相加。此前两条支路串行执行、各自做 TP all-reduce,decode 时 GPU 利用率低;模型用 torch.compile 手写 RMSNorm 也无法接入 vLLM 的 `allreduce_rms` 融合 pass。改动从 Cohere 内部 vLLM fork 移植到 upstream。

## 优化点

三处机制改动:(1) **attention/MoE dual stream**——用 `direct_register_custom_op` 注册 `vllm::cohere2_mlp_fork`/`vllm::cohere2_mlp_join` 两个 custom op(Dynamo 对其不透明,保住 stream 切换;attention 仍是 piecewise 分割 op),fork 在专用 side stream 上跑 MLP/MoE 与 attention 重叠,join 让主 stream 等待后做该层唯一 all-reduce;超过 `VLLM_COHERE2_ATTN_MOE_DUAL_STREAM_TOKEN_THRESHOLD`(默认 256,0 关闭)、piecewise CUDA graph(prefill/混合 batch,fork 不能跨 graph 边界)、breakable capture 时回退 inline;TP 下内部自约简的 MoE(共享 expert、EP all2all、sequence parallel)与 DBO 微批、非 CUDA 平台保持单流。(2) **每层单次 all-reduce**——并行块两支都是 row-parallel 部分和,`all_reduce(attn + mlp)` 数学等价于分开约简:`o_proj`/dense MLP 返回未约简输出,routed-only MoE 在 TP>1 时向 `FusedMoEFactory` 传 `reduce_results=False`,MoE 被迫内部约简时由 `output_is_reduced` 记录、decoder 只约简 attention。(3) **fused add + RMSNorm**——模型内手写 torch RMSNorm 换成 vLLM `RMSNorm`,每层返回约简后的块输出与 residual,由下一层 norm 融合相加,使 `allreduce_rms` pass 从 `Replaced 0 patterns` 变为每编译子图 `Replaced 1 patterns`;Eagle3 aux states 经 `_maybe_add_hidden_state` 加回 residual,起始层 capture 因 in-place residual 更新而克隆。

## 关键改动

```text
vllm/model_executor/models/cohere2_moe.py: fork/join custom op、单 all-reduce、vLLM RMSNorm 接入
vllm/envs.py: VLLM_COHERE2_ATTN_MOE_DUAL_STREAM_TOKEN_THRESHOLD(默认 256)
tests/models/test_cohere2_moe_parallel_block.py: 并行块单测(含 CUDA side-stream 竞态测试)
.buildkite/test_areas/models_basic.yaml: 新测试纳入 CPU job
```

```python
state.fork_event.record(torch.cuda.current_stream())
with torch.cuda.stream(state.stream):
    state.fork_event.wait()
    mlp_output = self.mlp(mlp_input)
    state.join_event.record()
...
if state.join_pending:
    torch.cuda.current_stream().wait_event(state.join_event)
return attn_output + mlp_output
```

## 性能声明

(TP2,2x H100 80GB,`--max-model-len 8192`;TPOT 来自 `vllm bench serve` random 数据集 1024 输入/512 输出 `--ignore-eos`;speedup = main TPOT / 本 PR TPOT;每配置跑一次)

| Checkpoint | gsm8k main→PR | TPOT c=1 | c=8 | c=32 | c=64 |
| --- | --- | --- | --- | --- | --- |
| North-Mini-Code-1.0 (bf16) | 0.899→0.898 | 4.07→3.04ms (1.34x) | 7.00→5.61 (1.25x) | 10.59→9.29 (1.14x) | 14.04→11.97 (1.17x) |
| -fp8(VLLM_CUTLASS MoE) | 0.895→0.908 | 5.34→4.31 (1.24x) | 6.93→5.89 (1.18x) | 10.00→8.80 (1.14x) | 12.32→10.68 (1.15x) |
| -w4a16(NVFP4 权重) | 0.905→0.908 | 4.19→3.07 (1.36x) | 5.57→4.62 (1.21x) | 8.00→6.59 (1.21x) | 10.17→8.93 (1.14x) |

- 消融(bf16):+单 all-reduce 最高 1.05x;+fused add+RMSNorm 约 1.00–1.05x;+dual stream 后达 1.14–1.34x——提速几乎全部来自 dual stream。关掉 dual stream(阈值=0)时单 all-reduce + fusion 贡献最高 1.05x。
- rebase 后 bf16 复测:1.45x/1.20x/1.17x/1.12x(c=1/8/32/64)。
- 精度:gsm8k 5-shot 1319 题、SE 约 ±0.8 点,均在噪声内;投机解码 acceptance 不变(EAGLE 0.503→0.503,Eagle3 0.289→0.290;负对照残差丢弃后 0.005,证明比较能抓住 aux capture 损坏)。
- 所有 dual-stream 配置通过 `VLLM_GPU_SYNC_CHECK=error`(无新增 GPU→CPU sync)。

## 验证与回归风险

新增 `tests/models/test_cohere2_moe_parallel_block.py`(CPU 可跑,进 Buildkite "Basic Models Other" job):覆盖延迟约简决策与单 all-reduce、跨层 residual 流、in-place residual 下的 Eagle3 aux capture、dual-stream 门控与 inline 回退,以及 CUDA side-stream 竞态测试(join 不等待即失败)。风险面:dual stream 默认开启(仅 CUDA),fork/join 的 stream/event 生命周期是主要复杂度(token threshold 未按投机解码调优,有投机时每请求每步多 1+k token,更低并发即达阈值);DBO、EP all2all、内部约简 MoE、非 CUDA 均自动单流;Cohere LayerNorm checkpoint 保留旧 residual 流。

## 关联

#61508(`swiglu_limit`,也改 MLP/MoE 构造函数,与本 PR 不重叠)。同分类(投机解码)无直接改动,但验证了 EAGLE/Eagle3 acceptance 不受影响。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
