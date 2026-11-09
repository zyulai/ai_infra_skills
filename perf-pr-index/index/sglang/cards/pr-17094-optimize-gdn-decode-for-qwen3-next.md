# PR #17094: Optimize GDN decode for Qwen3 Next

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @samuellees |
| 合并 | 2026-01-30 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 数;diff 本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/17094 |

## 一句话结论

把 Qwen3-Next GDN decode 的 fused recurrent Triton kernel 中 V 维分块 BV 从 8 提到 32,B200 NVFP4 推理下 gsm8k e2e 吞吐 2967 -> 3266 token/s(约 +10%)。

## 动机

在 MoE/FullAttn/GEMM 相继优化后,GDN(Gated DeltaNet)kernel 成为 Qwen3-Next(Q3N)的新瓶颈:B200 上 NVFP4 推理、ISL=1k、cc=64/256 时,decode 线性注意力约占 15-23% 时间。优化对象是 `fla/fused_recurrent.py` 的 decode 线性注意力 kernel。参考 vLLM 的同型优化 vllm-project/vllm#31722。

## 优化点

单行改动:decode 路径的 `fused_recurrent_gated_delta_rule_fwd` 中,`BV`(V 维的 Triton block 大小)从 `min(next_power_of_2(V), 8)` 改为 `min(next_power_of_2(V), 32)`。BV 决定每次迭代处理的 value 通道数——从 8 提到 32 后单次循环覆盖更多 V 维列,减少主循环迭代次数,提高每 warp 的有效计算/访存比;K 维分块 BK 保持 `next_power_of_2(K)`(NK 恒为 1 的约束不变),`num_stages=3` 不变。prefill chunked 路径不受影响(decode-only 调整)。

## 关键改动

```text
python/sglang/srt/layers/attention/fla/fused_recurrent.py: fused_recurrent_gated_delta_rule_fwd 的 BV 上限 8 -> 32
```

关键片段(完整改动):

```python
- BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
+ BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
```

## 性能声明

测试条件:TP4(TEP4)、MTP OFF、B200,gsm8k 数据集:

- Before:Accuracy 0.945,Invalid 0.000,Latency 74.870 s,Output throughput 2967.035 token/s
- After:Accuracy 0.943,Invalid 0.000,Latency 67.961 s,Output throughput 3266.447 token/s
- 约 10% e2e 提升;PR 未注明 ISL/OSL 等请求负载细节

## 验证与回归风险

验证为 gsm8k 前后对比(0.945 -> 0.943,PR 视为正常波动范围,未给出噪声界);未注明新增单元测试,Checklist 中 Add unit tests/Provide benchmark results 两项未勾选。回归面:所有走 `fused_recurrent_gated_delta_rule_fwd` decode 的 GDN 模型(Qwen3-Next 类 hybrid 线性注意力);BV=32 需要更多共享内存/寄存器,若 V 维较小由 `min(next_power_of_2(V), 32)` 兜底;prefill 路径与 CUDA graph 无直接改动。

## 关联

参考实现:vllm-project/vllm PR #31722(同类 GDN kernel 优化);相关人 @yizhang2077。同区域为 Qwen3-Next GDN/hybrid 线性注意力优化(同 batch 有 #10586 的 MTP verify 状态回写优化);pack 未列出其他直接关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
