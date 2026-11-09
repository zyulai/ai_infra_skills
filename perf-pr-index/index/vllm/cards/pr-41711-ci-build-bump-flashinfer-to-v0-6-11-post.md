# PR #41711: [CI/Build] Bump flashinfer to v0.6.11.post2

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @arpera |
| 合并 | 2026-05-16 |
| 分类 | Attention / Kernel \| tier: major |
| Label | ready, ci/build, nvidia, ready-run-all-tests |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/41711 |

## 一句话结论

把 FlashInfer 从 v0.6.8.post1 升到 v0.6.11.post2，同步更新 Docker/requirements pin 与 mxfp4 权重交错。

## 动机

PR 正文 Purpose 说明本 PR 把 FlashInfer 从 `v0.6.8.post1` 升级到 `v0.6.11.post2`；原计划为 cu13 用户改用 `flashinfer-python[cu13]` extra 一项被划掉，正文注明 "Upd. Not needed anymore. PR #42438 fixed that issue"；另修复 `tests/kernels/moe/test_cutedsl_moe.py` 中的 tensor shapes。正文未提供更多背景或性能瓶颈描述。

## 优化点

版本升级落在 `docker/Dockerfile` 的 `ARG FLASHINFER_VERSION`、`docker/Dockerfile.nightly_torch` 的源码构建分支（`git clone --branch`）与 release 注释、`docker/versions.json` 的 `FLASHINFER_VERSION.default`、`requirements/cuda.txt` 的 `flashinfer-python` 与 `flashinfer-cubin` 四处 pin，全部由 0.6.8.post1 改为 0.6.11.post2。`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` 删除本地 `_interleave_mxfp4_cutlass_sm90`，改为调用 FlashInfer 的 `interleave_moe_weights_for_sm90_mixed_gemm` 与 `interleave_moe_scales_for_sm90_mixed_gemm` 处理 w13/w2 权重与 scale，并把交错后的权重返回给调用方。`tests/kernels/moe/test_cutedsl_moe.py` 把 `input_global_scale` 取 `[:1].contiguous()` 作为 `a_global` 传给 `fp4_quantize` 与 `dequantize_nvfp4_to_dtype`，因为这两个非 grouped API 期望 `[1]` 或 `[num_tokens]` 而非 per-expert `[num_experts]`。

## 关键改动

```text
docker/Dockerfile、docker/Dockerfile.nightly_torch、docker/versions.json: 把 FLASHINFER_VERSION 从 v0.6.8.post1 升到 v0.6.11.post2
requirements/cuda.txt: flashinfer-python / flashinfer-cubin 同步升到 0.6.11.post2
vllm/model_executor/layers/fused_moe/oracle/mxfp4.py: 改用 FlashInfer 的 interleave_moe_*_for_sm90_mixed_gemm API 做权重/scale 交错
tests/kernels/moe/test_cutedsl_moe.py: 修正 fp4 量化参考实现里 input_global_scale 的 shape
```

```diff
 # FlashInfer should be updated together with the Dockerfile
-flashinfer-python==0.6.8.post1
-flashinfer-cubin==0.6.8.post1
+flashinfer-python==0.6.11.post2
+flashinfer-cubin==0.6.11.post2
 apache-tvm-ffi==0.1.9
 tilelang==0.1.9
```

## 性能声明

PR 正文未给出定量性能数据（纯依赖版本升级与形状修复），也未给出测试条件，模型、硬件、精度与 workload 均 PR 未注明测试条件。

## 验证与回归风险

PR 正文的 Essential Elements of an Effective PR Description Checklist 四项全部未勾选，未注明测试命令或结果。回归风险：FlashInfer 版本升级影响所有依赖 FlashInfer 的 attention 与 MoE 路径；`mxfp4.py` 的权重/scale 交错改由 FlashInfer API 完成，影响 SM90 mixed-gemm MXFP4 MoE 路径；`test_cutedsl_moe.py` 的 shape 修复影响 cuTeDSL MoE 参考实现；正文相关说明提到旧版 FlashInfer 会 fail capability validation 而非静默走非原生 tuning path，PR 未注明更多受影响路径。

## 关联

正文明确：PR #42438 修复了 cu13 用户改用 `flashinfer-python[cu13]` extra 的问题，故本 PR 不再需要该改动。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
