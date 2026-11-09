# PR #24123: [Compilation][WideEP] Enable Piecewise CUDAGraph for DeepEPHT

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2025-09-09 |
| 分类 | CUDA Graph / Compiler |
| Label | performance, ready, deepseek |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计,diff 内容有保留) |
| 链接 | https://github.com/vllm-project/vllm/pull/24123 |

## 一句话结论

DP + DeepEP high-throughput 场景从"禁用 CUDA graph + enforce_eager"改为 Piecewise CUDA graph,decode 吞吐提升约 33%。

## 动机

此前 `VLLM_ALL2ALL_BACKEND=deepep_high_throughput` 且 data_parallel_size>1 时,DeepEP HT kernel 与 CUDA graph capture 不兼容,平台检查直接把 `cudagraph_mode` 设为 NONE 并强制 `enforce_eager=True`。通信确实无法 capture,但 attention 等纯计算部分也一并失去了 graph 化的 CPU 开销优化,decode 路径损失明显。

## 优化点

把 MoE 的 dispatch/combine 边界显式暴露为 splitting ops:在 `set_splitting_ops_for_v1` 中,当 all2all backend 为 `deepep_high_throughput` 时向 `splitting_ops` 追加 `vllm.moe_forward` 与 `vllm.moe_forward_shared`,使编译切分点落在 MoE 通信处——通信留在 CUDA graph 之外,两侧的 compute 仍被 piecewise capture。CUDA 平台检查相应从"禁用"改为把 `cudagraph_mode` 强制为 `CUDAGraphMode.PIECEWISE`(NONE 除外),不再设 enforce_eager。另修复 `splitting_ops = self._attention_ops` 直接引用类级 list 的隐患(改为拷贝)。

## 关键改动

```text
vllm/config/compilation.py: deepep_high_throughput 时向 splitting_ops 追加 moe_forward/moe_forward_shared
vllm/platforms/cuda.py: DP+DeepEPHT 从禁用 cudagraph 改为强制 PIECEWISE 模式
```

## 性能声明

deepseek-ai/DeepSeek-R1,DP=8 + EP,max-num-seqs 512,full_cuda_graph 配置,`--cuda-graph-sizes 16...512`。Decode benchmark(random input-len 1 / output-len 200,1024 prompts,request-rate inf):output token throughput 8133.06 → 10819.65 tok/s,TPOT mean 121.50 → 89.55 ms(PR 称约 33% E2E decode 提升)。Prefill benchmark(input 200 / output 1)基本持平(3226.53 vs 3143.05 ms TTFT)。正确性:GSM8K exact_match 0.95(PR 前 main 为 0.9530/0.9515)。PR 未注明测试 GPU 型号。

## 验证与回归风险

PR 正文用 lm_eval GSM8K 与 vllm bench serve 做回归,无新增单元测试(diff 仅两处配置改动)。风险面:仅影响 `deepep_high_throughput` + DP>1 的路径;需要 MoE 通信也被 capture 的用户仍需切到 `deepep_low_latency`;splitting_ops 追加逻辑对所有使用该 backend 的 compilation 配置生效,若其他 splitting_ops 组合与之冲突需依赖 `splitting_ops_contain_attention` 等既有校验。

## 关联

同区域:Piecewise CUDA graph / compilation 体系(引用 #20059 的 full cudagraph 处理)、DeepEP 相关 WideEP 优化;与后续 full cuda graph for DeepEP 类 PR 同属一条线。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
