# PR #54809: [Quant][Kernel] Remove GPTQ Group/Dynamic Activation Ordering

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Roderick-Wu |
| 合并 | 2026-09-08 |
| 分类 | Quantization (tier: major) |
| Label | bug, performance, rocm, intel-gpu, torch.compile, ci/build, cpu, nvidia, quantization, verified |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 可用但被截断(93 个文件未收录) |
| 链接 | https://github.com/vllm-project/vllm/pull/54809 |

## 一句话结论

彻底移除 GPTQ act-order(g_idx/激活排序)支持,删掉 Marlin/Machete/CPU/ROCm 量化内核中的死分支与模板特化,精简 repack 接口。

## 动机

GPTQ 的 group/dynamic activation ordering(act-order)特性已从上游生态移除:llm-compressor 与 compressed-tensors 均已删除该支持(对应 compressed-tensors#840、llm-compressor#3038,关联 issue #48148)。vLLM 内继续保留它意味着量化配置、checkpoint 解析(g_idx)和各后端内核都要维护一套基本不再被新 checkpoint 使用的分支与模板特化(如 Marlin 的 `group_blocks == 0`),徒增代码量、编译时间与维护成本,故做一次性彻底清理。

## 优化点

这是一个减法型优化:Python 侧删除 `MPLinearLayerConfig` 的 `has_g_idx` 字段、`MPLinearKernel.__init__` 的 `w_gidx_param_name` 参数、各量化配置里 `has_g_idx=False` 的赋值以及 kernel 实现中检查 `c.has_g_idx` 的死分支;CUDA 侧 Marlin dense/MoE 内核删除 `g_idx` 参数、`has_act_order` 模板逻辑与 act-order scale 乘法,改为 `static_assert(group_blocks != 0)` 明确不支持运行时激活排序,`generate_kernels.py` 因此少生成一组 `group_blocks=0` 的内核特化,直接减少编译产物;CPU `cpu_wna16.cpp` 的 `Dequantizer4b` 去掉 `use_desc_act` 模板参数,AMX/VEC/RVV 三条 ISA 路径的分支数量减半;ROCm `q_gemm_rdna3.cu` 与 `gptq_marlin_repack`/`marlin_gemm` 等 op 签名同步去掉 `g_idx`/`sort_indices`/`is_k_full` 参数。

## 关键改动

```text
csrc/libtorch_stable/quantization/marlin/*: 移除 g_idx/act-order 分支与 repack sort_indices 参数(文件整体删除式清理)
csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h: 删除 has_act_order 模板逻辑,g_idx 参数移除,group_blocks==0 改为 static_assert
csrc/libtorch_stable/moe/marlin_moe_wna16/generate_kernels.py: group_blocks 列表去掉 0,减少内核特化
csrc/cpu/cpu_wna16.cpp: Dequantizer4b 去掉 use_desc_act 模板参数,三条 ISA 反量化路径简化
benchmarks/kernels/benchmark_marlin.py: 移除 act_order/k_full 基准维度
```

核心机制片段:

```cuda
constexpr bool has_act_order = group_blocks == 0;
// 改为:
static_assert(group_blocks != 0,
              "runtime activation ordering is not supported");
```

## 性能声明

PR 正文未给出任何 benchmark 数字(素材中 perf claims 检测为 none),性能收益仅以"删除死代码分支、减少内核模板特化(少编译 group_blocks=0 一组 Marlin MoE 内核)"的形式隐含存在。PR 未注明测试条件。

## 验证与回归风险

Test Plan 为"Ran core tests",结果"Passed",PR 未列出具体测试清单;diff 中同步更新了 `benchmark_machete.py`、`benchmark_marlin.py` 等 benchmark 脚本与相关单测调用签名。回归风险:主要风险是不再支持带 act-order 的旧 GPTQ checkpoint(`g_idx` 现被忽略),依赖该特性的用户加载会得到错误结果而非报错;同时影响 Marlin dense/MoE、Machete、CPU WNA16、ROCm RDNA3 多条后端的 op ABI(参数签名变更),下游 fork 需同步。

## 关联

上游配套:compressed-tensors#840、llm-compressor#3038、issue #48148;同分类相关可参考索引中 Marlin/GPTQ 内核优化卡(如 pr-14138、pr-14447、pr-16850)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
