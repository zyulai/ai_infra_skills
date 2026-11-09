# PR #61867: [Perf] Kimi K3 Specialized Decode: CuTeDSL Optimized GB300 (1.5-2x)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @benchislett |
| 合并 | 2026-10-16 |
| 分类 | Speculative Decoding |
| Label | ready, kimi, k3 |
| Diff | diffstat 缺失(pack 标注 +None/-None);diff 规模大,37 个文件被截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/61867 |

## 一句话结论

Kimi K3 专属 GB300 CuTeDSL decode 运行时(组合式接入,opt-in),NoSpec decode step 快 1.79~1.92x,DSpark 验证 decode tok/s 快 1.53~1.75x。

## 动机

Kimi K3 是 hybrid(KDA 线性注意力 + MLA)大模型,decode 延迟在小 batch 下由通用 vLLM 路径的 kernel 组合与同步开销主导。NVIDIA 为 GB300 开发了一支专门的 decode 实现:针对 NoSpec 与 DSpark 验证两种模式、TP8/TP16 与 DCP(Decode Context Parallelism,DCP=TP)拓扑做延迟特化;要求不动权重、可随时按 batch 回退 stock 前向。

## 优化点

集成采用组合(composition):模型持有 `SpecializedDecode` 对象(经 `KimiK3ModelState`,继承 `MambaHybridModelState`,在权重加载后构建),`forward()` 里 `decode.plan(...)` 返回 None(如 token 数过多)即回退 stock forward,prefill/混合批/大批自动走原实现。GPU 侧是一组 CuTeDSL kernel(`vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/`):7-CTA cluster AttnRes(每 token 一个 cluster、每 CTA 1024 hidden 列,mbarrier 交换部分和,Lamport mailbox 以 -0.0 哨兵发布 16B 片段并 poll/re-arm),KDA/MLA 相关组件按子目录组织;另有 CuTe 词表 embedding gather(7x1024 BF16 列 + 框架 TP all-reduce)、L2 权重 prefetch kernel(持久 grid 走 (address, bytes) 表)、PDL(grid-dependent launch)串联相邻 kernel,并对需要跨 CTA 等待的持久 grid 做 co-residency 检查。经 `--additional-config '{"decode_backend": "latency_specialized"}'` opt-in。

## 关键改动

```text
vllm/models/kimi_k3/nvidia/model.py: 挂接 specialized_decode + forward 分流 + norm 位置调整
vllm/models/kimi_k3/nvidia/model_state.py: KimiK3ModelState 构建/初始化 specialized decode
vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/attn_res/kernel.py: 7-CTA cluster AttnRes 融合 kernel(残差混合+RMSNorm)
vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/common/: mailbox(Lamport)/prefetch(L2)/primitives(PTX)/host(TMA、residency)
```

```python
decode = self.specialized_decode
if decode is not None:
    step = decode.plan(positions, intermediate_tensors)
    if step is not None:
        return decode.run(step, input_ids, inputs_embeds)
hidden_states = self.model(...)
```

## 性能声明

条件:FP8 KV cache、FP32 SSM cache、model-provider 权重,GB300。

- TP16、无投机、ISL 1、OSL 1024、greedy(decode step = 中位 ITL,nsys graph-replay 交叉校验):B1 3.90 vs 7.38 ms(1.89x),B2 1.92x,B4 1.80x,B8 6.82 vs 12.19 ms(1.79x);decode tok/s B1 255 vs 135,B8 1156 vs 649。
- TP16、DSpark DL7、合成接受长度 6:decode tok/s B1 838 vs 489(1.71x)至 B8 3094 vs 2014(1.54x)。
- DL7、ISL 32k、DCP on、合成接受 3.78:TP8+DCP8 decode tok/s 1.36~1.50x,TP16+DCP16 1.53~1.75x(B1 483 vs 276,1.75x);TTFT 持平。
- 全矩阵(TP8,SPEED-Bench,ISL 8k/32k/100k):无投机 step time 1.31~1.66x(B1 8673→5254 µs @8k);DL3/DL7 验证 step 1.28~1.61x。
- 精度:AIME25/GPQA-Diamond/GSM8k/MRCR v2 8-needle 与 stock 差异在约一个标准误内(带/不带 DSpark DL7);per-step 误差等于 1-ulp 噪声底,prefill 与 stock bit-identical。

## 验证与回归风险

验证以离线评测与数值验证为主(AIME25/GPQA/GSM8k/MRCR 大样本对比、greedy logprob 漂移分析、decode-vs-prefill rescoring),作者自述 checklist 的 CI 项未勾满("尚不清楚 CI 里该放多少,可能补 GSM8k sanity check")。风险面:opt-in 且按 batch 回退,stock 路径行为不变;`needs_norm_in_logits` 状态切换使 final norm 的应用位置在 specialized/stock 两条路径间不同;Lamport mailbox/持久 grid 依赖 GB300(SM103)cluster 特性与 co-residency 检查;非 2 维连续 BF16 [V,7168] embedding shard 会直接 ValueError;MRCR 32k 输出截断在高方差下略多(64k cap 下可忽略)。

## 关联

同区域(Kimi K3 / DSpark):索引内 PR #50000(Kimi K3 模型支持,本 PR 的基座)、PR #53396(Kimi K3 DS-conv state layer kernel)、PR #46995(DSpark)、PR #47808(DSpark confidence scheduled)、PR #52861(DSA 模型路由 CUDA attention)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
