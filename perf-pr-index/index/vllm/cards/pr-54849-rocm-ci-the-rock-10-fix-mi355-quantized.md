# PR #54849: [ROCm][CI][The Rock 10] Fix (MI355) Quantized Models failure on The Rock 10 with Triton 3.8.x

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @rasmith |
| 合并 | 2026-09-17 |
| 分类 | Quantization(tier: major) |
| Label | rocm, ready, quantization |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数) |
| 链接 | https://github.com/vllm-project/vllm/pull/54849 |

## 一句话结论

修复 Triton 3.8 在 gfx950(MI355)默认启用 async copy 后 MXFP8 kernel 因 shared memory 放不下 num_stages=3 而启动失败的问题,该配置下改用 num_stages=2 且 micro-bench 反而更快。

## 动机

The Rock 10(Python 3.12 / Ubuntu 22.04 / Triton 3.8.x)上,`(MI355) Quantized Models` 测试组的 `test_mxfp8.py` 四个用例(logprobs/generation × dense/moe)全部因 "Engine core initialization failed" 失败。根因:MI355 上 `TRITON_HIP_USE_ASYNC_COPY` 默认开启(其他 MI 平台默认关闭),async copy 的额外 LDS buffer 使 `BLOCK_K=256` 时放不下 `num_stages=3`;用 `TRITON_HIP_USE_ASYNC_COPY=0` 复跑即可通过,证实了这一点。

## 优化点

改动仅限 `vllm/model_executor/kernels/linear/mxfp8/rocm_native.py` 的 `_select_cfg` 调度:新增 `_BK256_STAGES` 常量,默认仍为 3;仅当 `triton.__version__` 以 "3.8" 开头且平台为 ROCm gfx950(`on_gfx950()`)时降为 2。`_select_cfg` 中两处返回 `BLOCK_K=256, num_stages=3` 的分支(大 M、qkv-class/deep-K/very-large-M)改用该常量。作用域刻意收窄到 "gfx950 + Triton 3.8" 这一组合——async copy 仅影响 3.8.0 的 gfx950——确保 Triton 3.7.1 与其他平台完全不受影响,无回归面。附带 TODO:待 #50605 合并后移除 3.8 版本判断。

## 关键改动

```text
vllm/model_executor/kernels/linear/mxfp8/rocm_native.py: _BK256_STAGES 按 gfx950+Triton3.8 选 2,两处 BLOCK_K=256 分支引用
```

## 性能声明

作者用 `_mxfp8_linear_kernel` 做的小型 micro-bench(Triton 3.7.1 num_stages=3 vs Triton 3.8.x num_stages=2,MI355):

| shape (M,N,K) | 3.7.1 ns3 (µs) | 3.8.x ns2 (µs) | Δ |
|---|---:|---:|---:|
| 8192,4096,1024 | 74.3 | 62.8 | -15.5% |
| 8192,1024,1024 | 23.4 | 20.8 | -11.1% |
| 8192,6144,1024 | 122.8 | 105.0 | -14.5% |
| 4096,2048,6144 | 91.8 | 75.0 | -18.3% |

即 gfx950 + Triton 3.8 下 num_stages=2 不仅修复启动失败,较 3.7.1 的 num_stages=3 还快约 11%~18%。PR 未给出 e2e 吞吐/延迟数字;未注明测试硬件细节(仅 MI355/gfx950 语境)与迭代次数。

## 验证与回归风险

Test plan:`pytest -sv models/quantization/test_mxfp8.py::test_mxfp8_logprobs`,结果 `2 passed, 26 warnings in 144.95s`。作者在 MI300 与 MI350 上分别用 Triton 3.7.1 与 3.8.0 验证了各组合选取正确的 `num_stages` 值。风险面很小:改动是编译期常量选择,只影响 gfx950+Triton3.8 的两个 `_select_cfg` 分支;非 gfx950 ROCm、NVIDIA、Triton 3.7.x 行为逐位不变。潜在遗留:版本号字符串前缀匹配("3.8")在 3.8.x 之外的 3.80+ 版本号场景未讨论,且依赖 #50605 合并后清理。

## 关联

同区域:ROCm mxfp8 原生 kernel 与 gfx950 量化卡(pr-40033 triton mxfp4、pr-51274 gfx942 requant);待办依赖 #50605。PR 正文未列出其他关联编号。
