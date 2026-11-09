# PR #63076: [ROCm][Perf] Triton sigmoid+bias top-k when MoE grouping eliminates nothing

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @chennyiiis |
| 合并 | 2026-11-03 |
| 分类 | MoE (tier: major) |
| Label | rocm, ready, verified |
| Diff | diffstat 缺失(pack 未提供 +None/-None);正文自述三个文件、纯新增(new router kernel 206 行 + _aiter_ops.py gate + 测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/63076 |

## 一句话结论

分组路由退化为纯 top-k 时(n_group>=topk_group)用 register-resident Triton kernel 替换 AITER biased_grouped_topk,CUDA graph 下路由 op 快 1.16x-2.15x,MI355X 单请求 decode 各模型快 0.4%-4.2%。

## 动机

多个 DeepSeek-V3 架构 checkpoint(如 Moonlight-16B-A3B 64 experts/top-6、Kimi-K2 384/top-8)用 sigmoid+noaux_tc 路由但 num_expert_group=1、topk_group=1,分组阶段不淘汰任何 expert;FusedTopKBiasRouter 也会合成 topk_group==num_expert_group 的形状。此时 AITER biased_grouped_topk 仍在 LDS 上做每轮 group scan + shuffle argmax + invalidate 的串行循环,纯属浪费;该场景就是 sigmoid(logits)+bias 后做纯 top-k。

## 优化点

新增 vllm/model_executor/layers/fused_moe/router/rocm_sigmoid_bias_flat_topk.py:单 token 单 wave(_NUM_WARPS=1)的 Triton kernel,全部分数常驻寄存器,把带 bias 分数与 expert index 打包成保序 int64 key,TOPK 轮 tl.max + 等值淘汰即完成选择(无 argmax 配对归约、无 LDS 往返,tie 落向低 id);出现精确 tie 时走 int64 key 排序 fallback 重选该行。替换点收在 rocm_aiter_ops.biased_grouped_topk 这一个两路 router 共用的调用处;_flat_sigmoid_bias_topk_is_profitable gate 要求 topk_group >= num_expert_group、token<=1(_MAX_TOKENS=1)、experts<=512、topk<=8,且 `torch.cuda.is_current_stream_capturing()` 为真——eager 下 Triton Python launch 比 AITER 单次 C++ dispatch 慢(enforce-eager 实测 -1.8% 到 -3.5%),必须只在 graph 捕获时替换(用 stream capture 而非 forward context 判断,否则 V2 runner 的 FULL graph 捕不到)。bias 与 gating dtype 的 cast 行为保持与 AITER 一致。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/router/rocm_sigmoid_bias_flat_topk.py: 新增 kernel + can_use gate(_MAX_TOKENS=1/_MAX_EXPERTS=512/_MAX_TOPK=8)
vllm/_aiter_ops.py: _flat_sigmoid_bias_topk_is_profitable + biased_grouped_topk 分支替换
tests/kernels/moe/test_rocm_aiter_topk.py: 新增 5 组测试(40 形状对齐 AITER、tie、bias cast、scaling、捕获判定)
```

## 性能声明

路由 op(CUDA graph replay 下 per-call device time):gfx942/MI300X 1.67x-2.15x(256/384 experts,bf16/fp32 gating);gfx950/MI355X 每形状 1.16x-1.57x(64 experts top-6 最优 1.57x),top-16 或 >=896 experts 会输(最低 0.62x),故 gate 止于 512 experts/top-8。E2E:MI355X,五模型(64-384 experts,1k-32k context),单请求 decode 全部快 0.4%-4.2%(GLM-5.3-Flash 1024/conc1 +4.17%,Kimi-K2.6 +0.36%),高并发在噪声内;GSM8K GLM 92.12%->92.04%、Kimi-K2.6 93.48%->93.71%。注意:初期顺序 A/B 曾测得 +16-24% 吞吐,但 interleaved 复测未能重现,作者已撤回该结论。路由一致性:Moonlight 3323/3328、Kimi-VL 1248/1248 调用选中 id 与 AITER 一致;已知差异:NaN gating 输入下与 AITER 行为实质不同(6/10 合成配置,最大 weight delta 0.93)。

## 验证与回归风险

61 个新测试在 MI355X 通过(含 prime expert 数、tie、renorm、routed_scaling、真 CUDA graph 捕获判定);两个 live 模型的 routing comparison 与 NIAH/GPQA/interleaved e2e。回归面:所有 ROCm AITER 路由(两 router 共用替换点);eager 步骤与大 batch、>512 experts、top>16 模型仍走 AITER;NaN 输入行为与 AITER 不同(作者已注明可选 gate 加固);enforce-eager 部署不受影响(eager 不替换)。

## 关联

同区域:GroupedTopKRouter / FusedTopKBiasRouter 共用的 rocm_aiter_ops.biased_grouped_topk;stream-capture 判定先例(custom_all_reduce.py、sparse_attn_indexer.py)。无具体关联 PR 编号。
