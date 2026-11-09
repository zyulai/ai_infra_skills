# PR #4918: Add DeepSeek V3/R1 shared experts fusion

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2025-04-04 |
| 分类 | Model-specific Fast Path |
| Label | high priority |
| Diff | diffstat 缺失(素材未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/4918 |

## 一句话结论

把 DeepSeek V3/R1 shared experts 复制成路由专家融合进 MoE,QPS=4 时吞吐 +4%,TTFT/ITL 降 15-20%。

## 动机

思路主要来自 vLLM #15502;sgl-kernel 的 moe_align_kernel 已支持 num_experts > 256(sgl-project/sglang#4348),使把 shared experts 融合进 256 个路由专家变得容易。此前 shared experts 是与 routed experts 并行的独立 MLP,每个 token 每层都要额外算一遍,融合后可走同一条 fused MoE 路径。

## 优化点

新增 `--n-share-experts-fusion`(默认 None 时取 tp_size)与 `--disable-shared-experts-fusion`(HIP 默认禁用);仅在 DeepseekV3ForCausalLM 且 n_routed_experts==256、routed_scaling_factor==2.5 时生效,默认启用。load_weights 把每层 shared experts 的 gate/up/down 权重(含 fp8 scale)按 n 份 clone 成 expert 256..255+n 并移除原 shared experts 权重;MoE 的 num_experts 变为 256+n、top_k+1。grouped_topk / biased_grouped_topk_impl 把 topk 最后一个 slot 的 id 随机指向融合专家,权重取其余 top-k 权重之和除以 2.5(routed_scaling_factor),renormalize 只对前 topk-1 求和。另附 H200 fp8_w8a8 的 E=257/E=264 Triton MoE tuned config。

## 关键改动

```text
python/sglang/srt/models/deepseek_v2.py: 融合开关与启用条件;load_weights 克隆 shared expert 权重为路由专家
python/sglang/srt/layers/moe/topk.py: grouped_topk/biased_grouped_topk_impl 增加 n_share_experts_fusion 逻辑
python/sglang/srt/server_args.py: 新增 --n-share-experts-fusion / --disable-shared-experts-fusion
python/sglang/srt/layers/moe/fused_moe_triton/configs/E=257|264,N=256,...H200....json: 新 tuned config
```

```python
if n_share_experts_fusion:
    topk_ids[:, -1] = torch.randint(
        low=num_experts,
        high=num_experts + n_share_experts_fusion,
        size=(topk_ids.size(0),),
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )
    topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / 2.5
```

## 性能声明

H200,DeepSeek-V3,TP8,bench_serving 300 prompts(share-gpt):QPS=1:总吞吐 483.47→485.72 tok/s,mean TTFT 949.18→664.25 ms,mean ITL 54.69→50.20 ms;QPS=4:总吞吐 1088.59→1132.73 tok/s(+4.0%),mean TTFT 2630.26→2144.08 ms,mean ITL 156.21→132.75 ms;QPS=8:总吞吐 1188.77→1235.63 tok/s(+3.9%),mean TTFT 6320.67→3443.59 ms,mean ITL 188.29→178.94 ms。PR 结论:QPS 提高到 4 时吞吐提升 4%,TTFT 与 ITL 约降 15-20%。GSM8K 2000 题 accuracy 0.952(H200)。

## 验证与回归风险

GSM8K accuracy 0.952;sgl-kernel test_moe_align.py 的 num_experts 参数扩展到 257/260/264。影响面:改变 DeepSeek V3/R1 的默认行为(默认启用,可用 --disable-shared-experts-fusion 关闭);topk 最后一个 slot 随机指向 shared expert 副本会改变专家负载分布与 renormalize 语义;load_weights 为每层克隆 n 份权重增加内存与加载时间;bench_serving 新增 --warmup-requests(默认 1);覆盖范围为 FusedMoE(triton)路径。

## 关联

#4348(moe_align_kernel 支持 >256 experts);vLLM #15502(思路来源)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
