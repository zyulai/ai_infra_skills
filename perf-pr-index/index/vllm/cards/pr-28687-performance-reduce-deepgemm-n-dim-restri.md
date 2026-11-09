# PR #28687: [Performance] Reduce DeepGEMM N dim restriction from 128 to 64 multiplier

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @alexm-redhat |
| 合并 | 2025-11-19 |
| 分类 | Attention / Kernel (tier: major) |
| Label | documentation, performance, ready, ci/build, deepseek |
| Diff | diffstat 缺失(pack 未提供 +None/-None across None);可见 diff 含 3 个文件(vllm/utils/deep_gemm.py、test_block_fp8.py、test-pipeline.yaml) |
| 链接 | https://github.com/vllm-project/vllm/pull/28687 |

## 一句话结论

把 DeepGEMM 的 N 维整除要求从 128 放宽到 64,让 fused_qkv_a_proj 改走 DeepGEMM kernel,该算子约快 1.83x,e2e TPOT 提升 2.2%。

## 动机

W8A8BlockFp8LinearOp 用 should_use_deepgemm_for_fp8_linear() 决定 FP8 linear 走 DeepGEMM 的 fp8_gemm_nt kernel 还是 cutlass。原判断要求 N 维能被 128 整除,导致 DeepSeek 等模型的 fused_qkv_a_proj(输出维度小于 128 的倍数)只能落到较慢的 cutlass 路径。作者查阅 DeepGEMM 自身的测试后确认 N 为 64 的倍数即可被 kernel 支持。

## 优化点

单点改动:should_use_deepgemm_for_fp8_linear() 中 weight.shape[0](N 维)的判定从 `% 128 == 0` 放宽为 `% N_MULTIPLE(64) == 0`,K 维仍要求 128 的倍数;判定与 deep_gemm 测试的 skip 条件同步(测试改为调用同一函数决定是否 skip,避免两处逻辑漂移)。由此 N 为 64 倍数、非 128 倍数的 FP8 linear(典型即 fused_qkv_a_proj)自动切到 DeepGEMM。同时在 BuildKite 新增独立的 "Kernels DeepGEMM Test (H100)" CI step,覆盖 block fp8、MoE、batched DeepGEMM 与 DeepGEMM attention 测试。

## 关键改动

```text
vllm/utils/deep_gemm.py: should_use_deepgemm_for_fp8_linear 的 N 维检查 128 -> 64(N_MULTIPLE=64, K_MULTIPLE=128)
tests/kernels/quantization/test_block_fp8.py: 对齐尺寸的 skip 条件改用 should_use_deepgemm_for_fp8_linear 判定
.buildkite/test-pipeline.yaml: 新增 Kernels DeepGEMM Test (H100) 可选 CI step
```

## 性能声明

fused_qkv_a_proj 从 cutlass 切到 deepgemm 后约 1.83X faster(来自对 profile 的检视,对比图在截图中,硬件/模型未在正文文字中注明)。e2e TPOT 从 28.32ms 降到 27.72ms,约 2.2%(TPOT 测试的模型/硬件/workload 条件 PR 未注明)。lm_eval 正确性验证显示无精度损失。

## 验证与回归风险

test_w8a8_block_fp8_deep_gemm_matmul 去掉硬编码的 M%4/K%128/N%64 skip,改由 should_use_deepgemm_for_fp8_linear 决定,参数化矩阵(M、N、K、block_size)下验证;新增 H100 DeepGEMM CI step(可选)。回归面:所有走 W8A8 block-FP8 linear 的模型(DeepSeek 系及其他 FP8 block 量化模型),N∈[64,128) 倍数的 shape 会改变 GEMM 后端选择;M 未做约束变化,准确性由 lm_eval 与 kernel 测试兜底。

## 关联

同区域:DeepGEMM FP8 GEMM 路径的其他优化(正文与 diff 未列出具体 PR 编号)。
