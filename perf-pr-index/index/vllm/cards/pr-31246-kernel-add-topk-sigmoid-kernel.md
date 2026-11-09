# PR #31246: [Kernel] Add topk_sigmoid kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyang16 |
| 合并 | 2026-01-21 |
| 分类 | Attention / Kernel |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 未提供,正文 diff 另有 8 个文件被裁剪) |
| 链接 | https://github.com/vllm-project/vllm/pull/31246 |

## 一句话结论

为 MoE fused topk 增加 sigmoid 打分与 e_score_correction_bias 支持,MiniMax-M2 在 8xH200 上 e2e 输出吞吐 +11%,gating kernel 11 µs → 6 µs。

## 动机

现有 fused topk kernel 只支持 softmax 打分。MiniMax-M2 的路由使用 sigmoid(其 config.json 的 scoring_func),此前只能用 `grouped_topk` 并设 `num_expert_group=1` 来模拟 `fused_topk`,多绕一层;另外 DeepSeek 系路由需要的 `e_score_correction_bias`(带 bias 选择、返回原始分数作为权重)在 fused kernel 中缺失。

## 优化点

CUDA 侧把原 `topkGatingSoftmax` 重构为统一模板 `topkGating`,新增 `ScoringFunc`(SOFTMAX/SIGMOID)编译期分派:sigmoid 分支直接逐元素 `1/(1+__expf(-x))`,省去 softmax 的 max-reduce、exp 与 sum 归约;新增 `moeSigmoid` kernel 供非 2 次幂 expert 数的两段式 fallback 使用。`moeTopK` 增加可选 `bias`:选择时用 `score + bias[expert]` 比较大小,但输出的 `topk_weights` 取未加 bias 的原始分数(与 torch 参考实现对齐),renormalize 也基于原始分数。Python 侧暴露 `ops.topk_sigmoid(..., bias?)` 并给 `ops.topk_softmax` 加 bias 参数,`fused_topk`/`fused_topk_bias` 打通 `scoring_func`;ROCm 上 `fused_topk` 优先用 aiter 的 `topk_sigmoid`(aiter 不支持 bias,故 ROCm 的 `fused_topk_bias` 保持 torch 实现)。MiniMax-M2 模型改为直接使用 fused `topk_sigmoid`。

## 关键改动

```text
csrc/moe/topk_softmax_kernels.cu: topkGating 统一模板(SF 分派)+ moeSigmoid + bias 选择逻辑
csrc/moe/torch_bindings.cpp: 注册 topk_sigmoid、topk_softmax 增加 bias 参数
vllm/model_executor/layers/fused_moe/router/: fused_topk/fused_topk_bias 打通 scoring_func(文件在裁剪部分)
tests/kernels/moe/test_fused_topk.py: 新增单元测试(softmax/sigmoid x bias x 3 dtype x renormalize)
benchmarks/kernels/benchmark_fused_topk.py: 新增 micro benchmark
```

```cuda
} else if constexpr (SF == SCORING_SIGMOID) {
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = 1.0f / (1.0f + __expf(-row_chunk[ii]));
  }
}
```

## 性能声明

e2e(MiniMax-M2,8x H200 TP8 + expert-parallel,`vllm bench serve` sharegpt,sharegpt-output-len 300,max-concurrency 16,1000 prompts):output token throughput 1655.81 → 1837.95 tok/s(+11%),TPOT 18.51 → 16.44 ms,TTFT 155.76 → 209.99 ms(变慢)。kernel profile:main 的 `grouped_topk_fused_kernel` 约 11 µs,PR 的 `topkGating` 约 6 µs。gsm8k 评测:精度两侧均 0.943,总延迟 183.035 → 159.336 s,output tok/s 701.406 → 803.968。

## 验证与回归风险

新增 `tests/kernels/moe/test_fused_topk.py` 与 `test_fused_topk_bias`:对 torch 参考实现比对 topk_weights(atol/rtol 1e-2)与 topk_ids(精确一致),覆盖 sigmoid/softmax、bias 有无、bf16/half/fp32、renormalize 开关;micro benchmark 覆盖 token 数 1-64、expert 数 16-512、topk 3/4。风险:MoE gating 逻辑变更影响所有 sigmoid 路由模型的专家选择正确性(数值容差内);`topk_softmax` 的 torch binding 签名加了 `Tensor? bias`,调用方需同步;ROCm bias 路径回落 torch 实现,性能与 CUDA 不对称。

## 关联

无(本批索引中无其他 MoE gating/topk PR 编号;MiniMax-M2 此前以 grouped_topk num_expert_group=1 模拟 fused_topk)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
