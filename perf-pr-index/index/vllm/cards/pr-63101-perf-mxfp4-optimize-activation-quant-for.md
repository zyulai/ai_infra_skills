# PR #63101: [Perf][MXFP4] Optimize activation quant for mxfp4. 3x kernel speedup, 5% throughput improvement

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ilmarkov |
| 合并 | 2026-10-26 |
| 分类 | Quantization(major) |
| Label | performance, ready, ci/build, nvidia, quantization |
| Diff | 素材未提供 diffstat 数字;pack 保留部分 diff,另有 3 个文件因超长未收录 |
| 链接 | https://github.com/vllm-project/vllm/pull/63101 |

## 一句话结论

MXFP4 MoE 激活量化 kernel 直接读取 sorted_expert_ids 跳过 expert offset 搜索,256 experts 时 kernel 延迟最高降 3.39x,GLM-5.3-MXFP4 E2E 输出吞吐 +5.7%。

## 动机

MoE routing 之后,激活行已按 expert 分组排序。原实现在量化 kernel 里为每一行重新解析所属 expert:小 expert 数走线性 offset 搜索,大 expert 数走共享内存上的二分搜索。而 Cutlass MoE 的路由 metadata 阶段(`get_cutlass_moe_mm_data`)本来就知道每行排序后的 expert id,这个搜索完全冗余,且开销随 expert 数增长:256 experts 时 microbench 中仅量化前的 offset 解析就耗约 7 µs,优化后降到约 2.2 µs。

## 优化点

核心改动是把路由排序阶段已经算出的 `sorted_expert_ids` 传进激活量化 kernel。`compute_arg_sorts` 在生成 `input_permutation`/`output_permutation` 的同时顺带写出每行排序后的 expert id;量化 kernel 有了这个张量就直接 `expert_idx = sorted_expert_ids[rowIdx]`,再用 expert offset 表做一次减法得到行内索引,取代逐行线性/二分搜索。`sorted_expert_ids` 是 optional 参数,不传时 kernel 行为与原 offset_search 完全一致(microbenchmark 显示 bitwise identical),保证兼容。

## 关键改动

```text
csrc/quantization/fp4/mxfp4_experts_quant.cu: 两个量化 kernel(mxfp4、fused silu_and_mul)增加 sorted_expert_ids 快速路径,绕过 offset 搜索
csrc/quantization/w8a8/cutlass/moe/moe_data.cu: compute_arg_sorts 增写 sorted_expert_ids 输出
csrc/torch_bindings.cpp / csrc/ops.h: mxfp4_experts_quant、silu_and_mul_mxfp4_experts_quant、get_cutlass_moe_mm_data 签名新增 Tensor? sorted_expert_ids=None
benchmarks/kernels/benchmark_mxfp4_experts_activation_quant.py: 新增 GLM-5.3 形状微基准(E=16..256 扫描)
tests/kernels/moe/test_mxfp4_moe.py: 新增 sorted path 与 offset_search 一致性测试
```

关键片段(mxfp4_experts_quant.cu,小 expert 分支):

```cuda
if (sorted_expert_ids != nullptr) {
  expert_idx = __ldca(&sorted_expert_ids[rowIdx]);
  rowIdx_in_expert = rowIdx - __ldca(&input_offset_by_experts[expert_idx]);
} else if constexpr (SMALL_NUM_EXPERTS) {
  /* 原线性 offset 搜索 */
```

## 性能声明

Kernel 微基准(NVIDIA B300,`benchmark_mxfp4_experts_activation_quant.py`,32 routed rows、topk=8,median latency):
- 仅量化:E=256 时 mxfp4 7.33→2.16 µs(3.39x),silu+mul 7.44→2.43 µs(3.07x);E=128 为 2.14x/1.99x;E=16 基本持平。
- 含 `get_cutlass_moe_mm_data`+shuffle+quant 全链路:受 metadata 开销(~14–15 µs)摊薄,整体约 1.4x;PR 自述 E2E 链路中"speedup is ~1.4×"。
- E2E:vllm serve RedHatAI/GLM-5.3-MXFP4,TP=4、无 speculative decoding、kv-cache fp8_e4m3、random 数据集 input 256/output 1024、32 prompts、并发 1。输出 tok/s 85.55→91.51(+5.7%),TPOT 11.43→10.67 ms;标题口径为 5% throughput improvement。
- Gsm8k 验证:flexible-extract 0.9052→0.9060,strict-match 0.9052→0.9083。

## 验证与回归风险

新增 `tests/kernels/moe/test_mxfp4_moe.py` 用例:`get_cutlass_moe_mm_data` 写出的 sorted ids 与 `torch.sort` 结果一致(SM100 限定);E=16/64/256 下 sorted 路径与 offset_search 路径输出一致。风险面:仅影响 SM100/SM11x 上启用 Cutlass MXFP4 MoE 的模型(如 GLM-5.3 MXFP4);op 签名改为带默认值 optional 参数,未传时行为不变;作者提出后续可在 large-m_topk 变体中移除冗余的 shared-memory expert_offsets staging。

## 关联

同属 MoE 激活量化/逐 block 量化 kernel 优化:#61641(FP8 量化 kernel 单次 cvt 转换两个 FP8 值)、#65791(SiLU-and-Mul per-block 量化向量化);依赖的 metadata 路径为 Cutlass MoE `get_cutlass_moe_mm_data`。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
