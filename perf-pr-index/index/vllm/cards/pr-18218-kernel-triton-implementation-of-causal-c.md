# PR #18218: [Kernel] Triton implementation of causal-conv1d for Mamba-based models

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @thoangtrvn |
| 合并 | 2025-07-09 |
| 分类 | Attention / Kernel (tier: major) |
| Label | ready, ci/build, v1 |
| Diff | diffstat 缺失(pack 未提供 +None/-None across None);可见 diff 显示 csrc/mamba/causal_conv1d 下 CUDA 实现被删除,另有 11 个文件因超长未包含 |
| 链接 | https://github.com/vllm-project/vllm/pull/18218 |

## 一句话结论

为 Mamba 系模型新增全 Triton causal-conv1d 内核,混合 prefill/decode batch 微基准最高快 27x,摆脱 CUDA-split 两路处理。

## 动机

PR 之前 vLLM 的 Mamba 系模型依赖 CUDA causal-conv1d 内核,处理混合 batch 时必须把请求拆成 prefill-only 与 decode-only 两路分别送不同内核(CUDA-split,即 PR #17146 的方案)。这与 vLLM v1 设计不兼容,开销也大。作者目标之一是让 Mamba 模型在 vLLM 中成为 fully Triton-only backend,更贴近 v1 架构。

## 优化点

实现两个 Triton 内核:causal_conv1d_update_triton(面向 decode-only 请求)与 causal_conv1d_fn_triton(面向混合 prefill/decode)。机制上与 CUDA 内核有三点不同:并行维度从 2D(feature-dim、batch)扩展为 3D(额外加 sequence-dim);不改输入数据的 layout(保持沿 feature-dim 连续);要求 conv-state cache 沿 feature-dim 连续,而原 CUDA 实现要求沿 kernel-width(序列)轴连续——正是这一 layout 差异使 CUDA 内核在 decode-only 或混合 batch 下无法高效处理。另外引入可跨层复用的 metadata 来压缩 Triton launch overhead。通过 `VLLM_USE_TRITON_CONV1D=1` 启用,合并时默认路径仍是 CUDA。

## 关键改动

```text
csrc/mamba/causal_conv1d/causal_conv1d.cu: 原 CUDA causal-conv1d fwd/update 内核(可见 diff 中已删除)
csrc/mamba/causal_conv1d/causal_conv1d.h / static_switch.h: CUDA 实现配套头文件(可见 diff 中已删除)
csrc/ops.h: 移除 causal_conv1d_fwd/causal_conv1d_update 的 C++ 声明
(新增的 Triton 内核文件在 pack 注明的 11 个被裁剪文件中,不可见)
```

## 性能声明

微基准(正文文字,硬件未注明):同批混合 prefill/decode 请求下 causal_conv1d_fn_triton 比 CUDA 内核快 27x;causal_conv1d_update_triton 在 decode-only batch 上优于 CUDA 内核(具体数字仅在截图);整体优于 CUDA-split 路径(PR #17146)。E2E:ShareGPT 数据集(输入 prompt 短,几百 token)、ibm-ai-platform/Bamba-9B:生成 256 token 时 CUDA 后端仍更快,Triton 慢 10%(total token throughput),output token throughput 与 TTFT 仅差 2%;生成 1024 token 时 Triton 后端 token throughput 快 5%、TTFT 快 11%。

## 验证与回归风险

正文声明同时新增了测试代码,但细节在截图中,素材未见具体测试清单。回归面:Mamba 系模型的 conv1d 执行路径在 CUDA 与 Triton 双实现间切换(conv-state layout 语义不同),短生成长度场景 Triton launch overhead 可能使 E2E 略有回退;可见 diff 中 CUDA csrc 文件被删除,与正文"默认仍走 CUDA 路径"的表述存在出入,以最终合并代码为准。

## 关联

前序:PR #17146(CUDA-split pathway)。
