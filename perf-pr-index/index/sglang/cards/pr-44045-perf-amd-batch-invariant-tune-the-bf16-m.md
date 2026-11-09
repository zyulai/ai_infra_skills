# PR #44045: [Perf] [AMD] batch-invariant: Tune the bf16 matmul for gfx950

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @aarnetalman |
| 合并 | 2026-10-28 |
| 分类 | Model-specific Fast Path(major) |
| Label | amd, run-ci, deterministic |
| Diff | pack 未提供 diffstat(additions/deletions/files 均缺失),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/44045 |

## 一句话结论

为 gfx950 定制 batch-invariant bf16 matmul 的 tile 配置(128x64x128 + 16-wide MFMA),确定性推理吞吐 1k 输入 +6%,decode 侧 GEMM 最高约 2x。

## 动机

开启 `--enable-deterministic-inference` 时,每个未量化线性层都走持久化 batch-invariant Triton matmul(`_matmul_persistent_triton`)。该 kernel 此前在所有 GPU 上共用单一 bf16 配置(128x128x64 tile,3 stages,8 warps),没有任何 AMD 专属优化。在 gfx950(MI350X/MI355X)上,128x64x128 tile 配合 16-wide MFMA 指令在 decode batch size 区间最高可快约 2x,值得为其单列配置。

## 优化点

把 launch 配置从 `matmul_kernel_persistent` 内联字典抽成 `_matmul_persistent_configs()` helper,并仅在 gfx950(`is_gfx95_supported()`)时把 bf16 条目替换为 `BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_stages=3, num_warps=8, matrix_instr_nonkdim=16`。该配置约需 147 KB LDS,只有 gfx950 的 160 KB LDS 装得下,因此严格限定在 gfx950;其他 GPU、fp16/fp32 与非确定性路径完全不变。配置仍对所有 M 固定,每个输出元素的 reduction 顺序与 batch 无关,batch invariance 语义不变。配置选自 MI350X 上约 2,900 个候选配置的 sweep,每个都做了 batch invariance 的逐位验证。确定性模式下 ROCm 上所有未量化 bf16 线性层(aten::mm/addmm)都会走这个 kernel,因此对任意 bf16 模型生效;量化层、MoE experts(Triton MoE runner)与 attention 不使用它。

## 关键改动

```text
python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py:
  新增 _matmul_persistent_configs();gfx950 bf16 配置
  128x64x128 + matrix_instr_nonkdim=16(约 147KB LDS)
test/registered/unit/batch_invariant_ops/test_matmul_persistent_configs.py:
  新增 CPU 测试守护 preset(gfx950 必须为该配置且 LDS 合规,
  其他 GPU 保持原配置)
```

## 性能声明

E2E(确定性模式,`sglang.benchmark.serving`,random dataset,range-ratio 1.0,request-rate inf,3 次取中位):ISL/OSL 1024/1024、256 prompts、并发 256 时,输出吞吐 3,370 -> 3,585 tok/s(1.064x,即约 +6%),ITL p50 -7.2%,请求延迟 p50 -6.0%,TTFT p50 +4.8%;ISL/OSL 8192/1024、并发 128 时为 1.012x/1.017x(两组 GPU 互换交叉验证,约 +1.5%),因 8k 输入以 prefill 为主,而新 tile 在 GEMM 层 prefill 形状约慢 5%。Kernel 级(相对原配置,`do_bench`,权重按 F.linear 转置传入):decode 批(M<=64)对 Qwen3-8B/Qwen3.8-27B 线性层形状最高 2.13x,geomean 1.59x(M<=64)、0.99x(M>=1024);如 4096x4096 在 M=1 为 1.98x、M=4096 为 0.96x。

## 验证与回归风险

确定性验证:MI350X(rocm/sgl-dev:v0.5.21-rocm724-mi35x),Qwen3-8B BF16,TP1,triton attention + deterministic inference,SGLANG_USE_AITER=0;`test_deterministic` prefix 与 radix_cache 模式(50 trials)通过;1,275 条响应(batch 1-50)与 batch size 1 逐位对比,token IDs 与 log-probabilities 1275/1275 完全一致;`test_batch_invariant_ops.py` 改前改后均 6 passed/54 subtests;新增 `test_matmul_persistent_configs.py` 2 passed(在无 helper 的旧代码上会 2 failed)。风险:配置对大 M(prefill 形状)在部分 (N,K) 上慢约 4-5%,PR 提出未来可用按 (N,K) 的配置找回,但配置绝不能依赖 M;preset 由 CPU 测试守护,防止无意改动。影响面仅 ROCm gfx950 确定性推理的 bf16 线性层。

## 关联

无(正文未引用其他 PR;确定性/batch-invariant 相关的其他 PR 编号在素材中未给出)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
