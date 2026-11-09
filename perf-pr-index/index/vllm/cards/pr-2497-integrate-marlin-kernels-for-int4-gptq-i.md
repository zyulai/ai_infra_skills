# PR #2497: Integrate Marlin Kernels for Int4 GPTQ inference

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @robertgshaw2-redhat |
| 合并 | 2024-03-01 |
| 分类 | Quantization(major) |
| Label | 无 |
| Diff | diffstat 缺失(diff 全文已提供但裁剪了 9 个文件,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/2497 |

## 一句话结论

把 IST-DASLab 的 Marlin Int4 GPTQ kernel 接入 vLLM:高 batch decode 下比原 gptq kernel 扩展性更好,为后续 GPTQ 推理默认提速路径奠基。

## 动机

vLLM 此前对 GPTQ 模型走 `gptq_gemm` kernel,decode 在大 batch 下扩展性不佳。IST-DASLab 发布的 Marlin kernel(经 Neural Magic 修改)针对 Ampere 及以上(`__CUDA_ARCH__ >= 800` 门控)做了 int4 权重 + fp16 激活的 tensor-core 优化,PR 目标是把该 kernel 端到端接入 vLLM 的 GPTQ 量化路径。

## 优化点

新增 `csrc/quantization/marlin/marlin_cuda_kernel.cu`(约 1145 行),核心机制:cp.async 多级流水线做 global→shared 预取,A 用 predicated 拷贝处理非 16 倍 batch,B(量化权重)用 `L2::evict_first` cache hint 拷贝避免污染 L2;int4 反量化用 LOP3 查表指令把 -8 零点直接融进 SUB/ADD 常数,配 `mma.m16n8k16` fp16 tensor core 指令;线程块按 stripe 划分 B 矩阵以在不同形状/GPU 配置下保持 SM 利用率,大 GEMM 时按 batch 64 分片并行,跨线程块先 shared memory 归约再经 L2 全局归约;group scale 只在 tile 落入新 group 时取一次。pybind 新增 `marlin_gemm(a, b_q_weight, b_scales, workspace, size_m, size_n, size_k)` 入口(非 ROCm 编译)。

## 关键改动

```text
csrc/quantization/marlin/marlin_cuda_kernel.cu: Marlin kernel 全量引入(cp.async 流水线 + LOP3 反量化 + tensor core mma)
csrc/pybind.cpp / csrc/ops.h: marlin_gemm 绑定与声明
csrc/quantization/marlin/LICENSE: IST-DASLab marlin Apache-2.0 来源声明
(Python 侧 wrapper 等其余 9 个文件在素材 diff 中被裁剪,见 PR 全文)
```

反量化核心:

```cuda
// Guarantee that the `(a & b) | c` operations are LOP3s.
int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
const int SUB = 0x64086408;   // fuse -8 zero point
frag_b[0] = __hsub2(*reinterpret_cast<half2 *>(&lo), ...);
```

## 性能声明

PR 正文未给出具体数字:仅称"Marlin 在 decode 高 batch 下比 gptq kernel 扩展性更好",配一张 GitHub 截图(数值未在文本中);benchmark 列为待办("Benchmark performance")。当前状态:e2e 在 `enforce_eager=True`、TP=1 下可用。GPU/模型/workload 等测试条件未注明。

## 验证与回归风险

这是 draft 性质的集成 PR,PR 正文明确列出未完成项:CUDA Graphs 开启时输出乱码(疑似临时分配所致)、benchmark 未做、`desc_act=true` 未支持(计划 torch 层 permutation 再融合)、带 bias 的模型与 channelwise 量化(kernel 已支持)待验证。风险面:kernel 仅编译进 sm80+;workspace/锁机制与多 TP 场景正确性依赖后续 PR 打磨。

## 关联

上游 kernel:IST-DASLab/marlin(Neural Magic 修改版)。后续 Marlin 系列扩展(Machete、AWQ/Marlin 变体等)素材未列编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
