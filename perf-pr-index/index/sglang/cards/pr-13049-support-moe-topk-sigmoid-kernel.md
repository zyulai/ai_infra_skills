# PR #13049: Support moe topk sigmoid kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @rogeryoungh |
| 合并 | 2025-11-19 |
| 分类 | MoE (tier: major) |
| Label | amd, sgl-kernel, ready-to-merge, run-ci |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 被截断(1 个文件未收入),含新 kernel .cu、benchmark、测试与注册接线 |
| 链接 | https://github.com/sgl-project/sglang/pull/13049 |

## 一句话结论

新增 `topk_sigmoid` CUDA kernel 支持基于 sigmoid 的 MoE 专家路由(MiniMax-M2),把此前 8 个 kernel 的 `grouped_topk(group_size=1)` workaround 合并为 1 个 kernel,整体吞吐提升约 10%。

## 动机

MiniMax-M2 等模型使用 sigmoid 路由(而非 softmax)选择专家,此前只能用 `grouped_topk` 配 `group_size=1` 变通实现,一次前向要 launch 8 个 kernel(sigmoid、bias、topk、gather、renormalize 等散在多个 torch/Triton 调用中),launch 开销与中间张量读写成为 routing 路径的额外成本。

## 优化点

仿照 vLLM/TensorRT-LLM 的 topk_softmax kernel 结构新增 `topk_sigmoid`:对 2 的幂个专家(1–256)用模板特化的 `topkGatingSigmoid` kernel,一次完成 sigmoid、correction_bias 加法、warp 内 argmax 归约(蝴蝶 shfl 模式、低索引平局优先)、逐轮清 max、以及 renormalize 融合(累计 top-k 权重和后原地归一),并支持 interleaved load 提升访存合并;非 2 的幂或 >256 专家时回退 `moeSigmoid` + `moeTopK` 两 kernel + workspace 路径(cub/hipcub BlockReduce)。支持 fp32/fp16/bf16 gating 输入(fp32 输出权重),correction_bias 必须 fp32。注册到 CUDA 与 ROCm 两套 extension,Python 侧以 out 参数(`topk_weights`/`topk_ids`)形式暴露。

## 关键改动

```text
sgl-kernel/csrc/moe/moe_topk_sigmoid_kernels.cu: 新增,2 的幂专家模板 kernel(sigmoid+bias+topk+renormalize 融合)与非 2 的幂回退路径
sgl-kernel/benchmark/bench_moe_topk_sigmoid.py: 新增,SGLang vs Torch 对比基准(num_tokens 128–32768,num_experts 32–512,topk 1–8)
sgl-kernel/tests/test_moe_topk_sigmoid.py: 新增,数值/索引/dtype/renormalize/correction_bias 参数化测试
sgl-kernel/csrc/common_extension.cc 与 common_extension_rocm.cc: 注册 topk_sigmoid op
```

```cuda
// fused: sigmoid -> bias -> warp argmax -> 逐轮清 max -> renormalize
val = 1.0f / (1.0f + expf(-val));
if (correction_bias != nullptr) { val = val + correction_bias[expert_idx]; }
...
if (renormalize && thread_group_idx == 0) {
  float inv = 1.f / row_sum_for_renormalize;
  output[idx] = output[idx] * inv;
}
```

## 性能声明

- 端到端:MiniMax-M2 上整体吞吐提升约 **10%**(作者正文;benchmark 表给出 max_concurrency 1/16/64 下 input_throughput 124.640/807.450/1305.240、output_throughput 80.140/485.430/862.120、mean TTFT 152.460/204.190/333.290 ms、median TPOT 11.770/21.480/35.460 ms,PR 未注明对照基线、硬件与测试时长)。
- kernel 数量:先前 8 个 kernel launch,现在仅 1 个(附 profiler 截图)。
- 精度:MiniMax-M2 上 GSM8K **0.93**、AIME2025 **0.803**。

## 验证与回归风险

新增 `test_moe_topk_sigmoid.py` 参数化覆盖:num_tokens 1–2048 x num_experts 4–256 x topk 1–4 的 torch 参照比对(weights atol/rtol 1e-3,indices 精确相等)、fp16/bf16/fp32 dtype 回归、fused renormalize 与 correction_bias 组合;`bench_moe_topk_sigmoid.py` 注册为基准脚本。风险:非 2 的幂专家数走两 kernel workspace 路径,收益较小;模板 kernel 假定 experts 为小 2 的幂(编译期断言);correction_bias 只支持 fp32;影响所有 sigmoid 路由的 MoE 模型权重选择数值(阈值 1e-3 内)。

## 关联

同区域:`topk_softmax` MoE routing kernel(本 PR 直接以其为结构蓝本)、grouped_topk、#9477(moe_sum_reduce);pack 未列出其他直接关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
