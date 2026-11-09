# PR #25860: add git gemm warpper for dispatch_bf16_fp32_backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BJWang-ant |
| 合并 | 2026-05-20 |
| 分类 | Attention / Kernel(major) |
| Label | deepseek, jit-kernel |
| Diff | pack 未提供 diffstat(additions/deletions/files 均缺失),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/25860 |

## 一句话结论

为 `_dispatch_bf16_fp32_backend` 增加 JIT DeepGemm 分支并复用 wrapper,运行期减少 cache 生成,DeepSeek-V4-Pro 吞吐约 +2-3%。

## 动机

DeepSeek V4 的 `linear_bf16_fp32`(BF16 输入、FP32 输出的线性层 GEMM)每次调用都会:重新读取 `SGLANG_OPT_BF16_FP32_GEMM_ALGO` 环境变量,且 `deep_gemm` 分支直接 `import deep_gemm` 并调用 `deep_gemm.bf16_gemm_nt`,在运行期触发 DeepGEMM 的 cache 生成(compilation cache miss)开销。作者为该 dispatch 增加 JIT DeepGemm 代码分支,以减少运行期 cache 生成。

## 优化点

改动很小但有两层。第一,algo 常量化:`linear_bf16_fp32` 把每调用一次的 `envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()` 提升为模块导入时一次性读取的 `_linear_bf16_fp32_algo`,dispatch 时直接用缓存值。第二,DeepGemm 调用走 wrapper:deep_gemm 分支从直接 `import deep_gemm; deep_gemm.bf16_gemm_nt(...)` 改为经 `deep_gemm_wrapper.gemm_nt_bf16bf16f32(x, y, z)` 调用,输出张量由 `x.new_empty` 改为显式 `torch.empty(..., device=x.device)`;通过 wrapper 复用已编译的 JIT DeepGemm kernel,避免每次调用在运行期重复产生编译 cache,降低 JIT 路径的开销。dispatch 其余分支(torch cublas JIT、aiter tgemm fallback)保持不变。

## 关键改动

```text
python/sglang/jit_kernel/deepseek_v4.py:
  _linear_bf16_fp32_algo 模块级缓存 algo 配置
  deep_gemm 分支改用 deep_gemm_wrapper.gemm_nt_bf16bf16f32
  linear_bf16_fp32 移除函数内 envs 读取
```

## 性能声明

测试条件(来自 server 命令):DeepSeek-V4-Pro,TP8/DP8 + dp-attention,megamoe A2A backend,EAGLE 投机(steps 3, topk 1, draft tokens 4),chunked prefill 65536,`SGLANG_OPT_DEEPGEMM_MEGA_MOE_*` 等环境变量开启;硬件未注明。`bench_serving` 对 isl4k-osl1k 与 isl32k-osl2k 两组 workload 测试,结论为吞吐约提升 2% 到 3%(正文文字部分先写 "approximately 2%",表格后总结写 "approximately 2% to 3%"):

- 32 并发 / 32k 输入 2k 输出:TTFT/TPOT 4683.08/19.10 ms -> 4716.73/18.46 ms;输入吞吐 23,105 -> 23,789 tok/s;输出吞吐 1,444 -> 1,486 tok/s。
- 128 并发 / 4k 输入 1k 输出:TTFT/TPOT 1950.26/17.19 ms -> 2021.52/16.87 ms;输入吞吐 23,791 -> 24,268 tok/s;输出吞吐 5,947 -> 6,067 tok/s。

TTFT 在两组中略升,TPOT 与吞吐改善。

## 验证与回归风险

PR 正文无 accuracy 测试,Checklist 未包含,pack diff 未含新增单测。风险:`algo` 配置改为导入时读取后,运行期修改 `SGLANG_OPT_BF16_FP32_GEMM_ALGO` 环境变量不再生效(进程内热改配置的场景);DeepGemm 路径输出张量分配方式变化(`new_empty` -> `torch.empty(device=x.device)`)语义等价但依赖 wrapper 与直接调用行为一致;影响面限于 DeepSeek V4 `linear_bf16_fp32` 调用点,dispatch 到错误分支的行为由 algo 配置决定。

## 关联

同区域:#27059、#26788、#24584(DeepSeek V4 索引/注意力系列)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
