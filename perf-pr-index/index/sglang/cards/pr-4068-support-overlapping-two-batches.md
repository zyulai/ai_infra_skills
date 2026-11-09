# PR #4068: Support overlapping two batches

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @fzyzcjy |
| 合并 | 2025-05-25 |
| 分类 | MoE(tier: major) |
| Label | high priority |
| Diff | diffstat 缺失(pack 未记录);diff 可用,另有 6 个文件被截断(含核心的 two_batch_overlap.py,详见 PR) |
| 链接 | https://github.com/sgl-project/sglang/pull/4068 |

## 一句话结论

引入 two-batch-overlap(TBO):把一个 batch 拆成两个子 batch 交错执行,让一个 batch 的 MoE 通信与另一个 batch 的计算重叠,8xH200 预测量得 +6.4% 吞吐。

## 动机

DeepSeek V3 这类 EP MoE 模型走 DeepEP dispatch/combine 时,通信阶段 GPU 大量空转。若把一个 batch 拆成两个子 batch,一个子 batch 做 dispatch/combine 通信时另一个子 batch 可以做专家计算,即可掩盖通信延迟。该 PR 依赖 DeepGEMM 与 DeepEP 集成作为前置(两者 2025.03 就绪后才更新代码),作者明言这是等 kernel 就绪前的初步版本,未做精细调优(DeepGEMM grouped gemm、DeepEP low-latency、CUDA graph 支持当时均未到位)。

## 优化点

核心机制(主文件 `two_batch_overlap.py` 在被截断的 diff 中,从其余文件可见其接线方式):(1) `ForwardBatch` 增加 `tbo_split_seq_index`、`tbo_children` 等字段,`TboForwardBatchPreparer.prepare` 在 batch 构建时把父 batch 沿 split 点拆成两个子 batch;(2) `TboAttnBackend` 包装任意 attention backend,持有一个 primary + 两个 children,forward metadata、CUDA graph capture/replay 都按 split 点切分后分发给子 batch;(3) scheduler 侧 `TboDPAttentionPreparer` 在 DP attention 的 all-gather 里多带 2 个 int(tbo split 信息),由 `compute_output` 算出全局一致的 `tbo_split_seq_index` 与 `global_forward_mode`;(4) DeepSeek V2 模型的 op 序列(`op_dispatch_a/b`、`op_combine_a/b`、`op_experts`)全部改穿 `tbo_subbatch_index`,`deepep_dispatcher` 换成 `MaybeTboDeepEPDispatcher`,`model_forward_maybe_tbo` 交错执行两个子 batch 的 op 流;(5) DeepGEMM 侧新增 `configure_deep_gemm_num_sms` context manager,可在重叠场景下调整 DeepGEMM 占用的 SM 数,为通信/计算并行让资源;(6) CUDA graph capture 的 batch size 列表在 TBO 开启时过滤掉 bs<2。

## 关键改动

```text
python/sglang/srt/two_batch_overlap.py: TBO 核心实现(split 计算、op 交错调度、Tbo* 工具类;diff 被截断)
python/sglang/srt/layers/attention/tbo_backend.py: TboAttnBackend,primary+children 的 attention backend 包装
python/sglang/srt/models/deepseek_v2.py: op 序列穿 tbo_subbatch_index,MaybeTboDeepEPDispatcher,model_forward_maybe_tbo
python/sglang/srt/managers/scheduler.py: prepare_dp_attn_batch 集成 TboDPAttentionPreparer(all-gather 4->6 字段)
python/sglang/srt/layers/quantization/deep_gemm.py: configure_deep_gemm_num_sms
```

split 信息跨 DP rank 对齐:

```python
tbo_split_seq_index, global_forward_mode = tbo_preparer.compute_output(
    global_info[:, :, 4:6]
)
```

## 性能声明

条件:8xH200,DeepSeek-V3-0324,`--tp 8 --dp 8 --enable-dp-attention --enable-deepep-moe --disable-cuda-graph --enable-flashmla --chunked-prefill-size 65536`;bench_serving random 数据集,input 1000 / output 1(output=1 是刻意禁用 decode,因 decode 依赖当时尚不存在的 low-latency kernel 与 CUDA graph),1024 prompts,max-concurrency 1024;bench 跑 5 次丢第 1 次(含 JIT 编译)。结果:baseline 吞吐 14492.08/14657.42/14515.49/14590.32,ours 15446.02/15528.12/15322.22/15724.91,平均 +6.4% 吞吐。作者强调这是依赖 PR 未就绪、未做调优的初步数字。

## 验证与回归风险

验证:MMLU 通过(2025.03.21);profiler timeline 确认两个 batch 确实交错、一个 batch 的通信与另一个 batch 的计算重叠。风险:TBO 开启时 CUDA graph 仅支持 num_tokens==bs 的 capture 路径(children 的 metadata 切分 assert);`encoder_lens`、`spec_info` 在 graph 切分路径显式 assert None 不支持;`BumpAllocator` buffer 在 TBO 时需翻倍;非 MoE 的前 `first_k_dense_replace` 层在 TBO 下走单独循环。默认关闭(`--enable-two-batch-overlap` opt-in),不开时 `can_run_tbo` 为 False 走原路径。

## 关联

正文提到聚合分支(fzyzcjy/sglang feat/dev_branch,合并 PD + EPLB + TBO 等)。依赖前提:DeepGEMM、DeepEP 集成(编号未注明);待其就绪的后续:DeepGEMM grouped gemm、DeepEP low-latency、CUDA graph 支持。同区域后续 TBO 相关 PR 编号在本 PR 正文中未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
