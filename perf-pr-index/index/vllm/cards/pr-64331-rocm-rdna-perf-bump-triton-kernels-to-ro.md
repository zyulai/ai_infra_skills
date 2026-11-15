# PR #64331: [ROCm][RDNA][Perf] Bump triton_kernels to ROCm 3.8.x with the RDNA MXFP4 configs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @umarkovi-amd |
| 合并 | 2026-11-12 |
| 分类 | Quantization(tier: major) |
| Label | rocm, ready, ci/build, verified |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/64331 |

## 一句话结论

把 ROCm triton_kernels pin 从 `669b31acc1` 升到 `3ef5021860`(ROCm/triton#952,3.8.x backport of triton-lang/triton#10810),为 RDNA3/4 带来 bf16 × MXFP4 `matmul`(gpt-oss MoE GEMMs)的 tile config:gpt-oss-20b 在 Radeon AI PRO R9700(gfx1201)上吞吐 +11.6%(10.59 → 11.82 req/s),MoE GEMM 时间降 18.7%。

## 动机

triton-lang/triton#10810(2026-07-12 上 main)在 3.8 branch cut(2026-06-23)之后,任何 3.8 release branch 都没有它;vLLM 的 ROCm build 把 `release/internal/3.8.x` pin 到与镜像中 Triton compiler 匹配的版本,因此从未带上这个 RDNA MXFP4 matmul config。ROCm/triton#952 将其 backport 到 `release/internal/3.8.x`(3.9 branches 已包含)。

## 优化点

- 升级 triton_kernels pin 到 `3ef5021860`,新 commit 为 ROCm/triton#952:为 RDNA3/4 增加 bf16 × MXFP4 `matmul`(即 gpt-oss 的 MoE GEMMs)的 tile configs,新 block 由 `get_rdna_version() in (3, 4)` 门控;CDNA4 bf16 × MXFP4 block 仅在共享 dtype check 下重缩进。
- 仅 triton_kernels 变更:Triton compiler pin(`Dockerfile.rocm_base`,3.8.0+git669b31ac)不动;两 pin 之间 triton_kernels 唯一实质差异文件是 `matmul_details/opt_flags.py`(+30/−13),其余差异为 compiler/setup 文件,vLLM 的 triton_kernels fetch 不使用。
- pin 更新三处:`cmake/external_projects/triton_kernels.cmake`(非 Docker build)、`docker/Dockerfile.rocm` 的 `ROCM_TRITON_KERNELS_COMMIT`、`docker/Dockerfile.rocm_72` 的同名 ARG(`Dockerfile.rocm_72` 在 ARG 与 cmake tag 不匹配时 build 失败)。

## 关键改动

```text
cmake/external_projects/triton_kernels.cmake: ROCm TRITON_KERNELS_TAG 669b31acc1 → 3ef5021860(含 RDNA MXFP4 matmul config,ROCm/triton#952)
docker/Dockerfile.rocm: ROCM_TRITON_KERNELS_COMMIT ARG 同步升级
docker/Dockerfile.rocm_72: 同 ARG 升级(pack diff 未展示该文件)
```

```diff
-    # Pinned from release/internal/3.8.x
-    set(TRITON_KERNELS_TAG "669b31acc1dd1b3fd93286afd5db67f65d9f7557")
+    # Pinned from release/internal/3.8.x (includes the RDNA MXFP4 matmul config, ROCm/triton#952)
+    set(TRITON_KERNELS_TAG "3ef5021860f07f76820cce2e4c8e92d5e834bf39")
```

## 性能声明

- **Throughput**(`vllm bench throughput --model openai/gpt-oss-20b --max_model_len 4096 --dataset-name sharegpt --num_prompts 1000 --tensor_parallel_size 1 --gpu-memory-utilization 0.8`,单张 R9700,同 build 同 session,每臂 1 cold + 2 warm,取 warm-best):main 10.59 req/s(10.29, 10.59)→ 本 PR 11.82(11.50, 11.82),**+11.6%**。
- **rocprofv3 kernel time**(同 benchmark 一轮 warm run,s):MoE GEMM tiles——small steps `32x256x128 → 16x64x512`:16.51 → 9.68;mid+large steps `64x256x64 + 128x256x128 → 64x128x128`:36.33+7.12=43.45 → 39.10;MoE GEMMs 总计 59.96 → 48.77(−18.7%);GPU kernel 总时间 103.02 → 91.83。main 上 192 个 large-step 调用用 `128x256x128`(GEMM1 平均 24.7 ms、GEMM2 12.3 ms),本 PR 改跑 `64x128x128`。main 的 2-warp `32x256x128` 变体(≤4 token steps)以 256 VGPRs 运行、每次约 1,200 spills;新 tiles 无 spill(151–180 VGPRs)。
- **GSM8K**(`lm_eval --model vllm --tasks gsm8k --num_fewshot 5`,1,319 题,R9700;新 tiles 改变 fp32 累加顺序,MoE 输出与 main 非 bitwise):main flexible/strict 0.3647/0.2540、0.3700/0.2722(均值 0.3673/0.2631);本 PR 0.3639/0.2494、0.3723/0.2691(均值 0.3681/0.2593);均值差(+0.0008/−0.0038)在 per-run stderr(0.013)内;两版本间 386–397 题翻转,同版本重跑间为 389–399 题。

## 验证与回归风险

- `pytest tests/kernels/moe/test_gpt_oss_triton_kernels.py` 在 gfx1201 上两种 pin 结果相同:19 passed、4 个 `test_equiv` 失败——该失败在 main 上同样存在(ROCm 报 capability 12.0,测试套用 NVIDIA SM12x `is_persistent` 约束 `InapplicableConstraint`)。
- 栈:ROCm 10.2.0a20261020、torch 2.14、Triton 3.8.0+git669b31ac、main c0a41721c7。
- 风险:RDNA3(gfx1100)被新 block 覆盖但本 PR 未在 gfx1100 上测量;MoE 输出与 main 非 bitwise(fp32 累加顺序变化),GSM8K 差异在噪声内。

## 关联

- ROCm/triton#952(backport)、triton-lang/triton#10810(上游);#64055(同作者的 RDNA OAI MoE combine 优化,同为 gpt-oss/RDNA 系列)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
