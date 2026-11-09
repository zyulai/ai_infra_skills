# PR #25948: [cpu][perf] Accelerate unquantized-linear for AArch64 through oneDNN/ACL and weight prepack

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fadara01 |
| 合并 | 2025-10-04 |
| 分类 | Quantization |
| Label | ready, ci/build |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 内容本身可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/25948 |

## 一句话结论

AArch64 非量化 linear 走 oneDNN/ACL + weight prepack,16 核 Neoverse-V2 吞吐约 +78%。

## 动机

PR #24150 已为 CPU linear 引入 weight prepack 与直连 oneDNN 的 matmul 路径,但该路径仅对 x86 生效:AArch64 上 linear 仍由 PyTorch 执行,每次 `torch.linear` 都要重新 pack weight,产生重复开销。同时不能无条件切换——oneDNN 在未启用 ACL backend 时反而比 PyTorch 路径慢,因此需要"编译期带 ACL 才分发"的条件路径。

## 优化点

本 PR 让 AArch64 在 oneDNN 以 Arm Compute Library(ACL)为 backend 编译时,同样启用 weight prepack 并把非量化 linear 分发到 oneDNN。为此对 oneDNN matmul 路径做了四处 ACL 兼容改造:(1) ACL 的 matmul 不支持 runtime 维度,创建 primitive descriptor 时固定 M=128、input stride=K;(2) ACL 不支持直接传 bias,改为 `c=bias; c+=matmul(a,b)`,通过给 matmul 附加 fused sum post-op 实现;(3) ACL 要求 source tensor 连续,执行前对 `a` 做 `contiguous()`;(4) ACL API 允许权重格式随输入维度变化,execute 时对比 weights desc 并按需重新 prepack(作者注明实际中 ACL 权重格式基本不变,非性能热点)。另外新增 `VLLM_USE_ACL` 编译宏与 `is_onednn_acl_supported()` 运行时探测,并修复 setup.py 丢弃外部 `CMAKE_ARGS`(含 `VLLM_BUILD_ACL`)的问题。

## 关键改动

```text
csrc/cpu/dnnl_helper.cpp: ACL 分支的 primitive desc 创建(M=128、fused sum post-op)、execute 时 repack 检查、is_onednn_acl_supported
csrc/cpu/dnnl_kernels.cpp: onednn_mm 中 source contiguous 化与 bias 的 c=bias;c+=matmul 处理
vllm/model_executor/layers/utils.py: dispatch_cpu_unquantized_gemm 放开 AArch64(oneDNN+ACL)分发条件
cmake/cpu_extension.cmake + setup.py: 定义 VLLM_USE_ACL、透传 CMAKE_ARGS
```

关键片段(utils.py 的分发条件):

```python
elif ops._supports_onednn and (current_platform.get_cpu_architecture()
                               == CpuArchEnum.X86
                               or ops.is_onednn_acl_supported()):
```

## 性能声明

16 Neoverse-V2 核心(taskset -c 0-15),meta-llama/Llama-3.1-8B-Instruct,`vllm bench throughput`:64 prompts、ShareGPT 数据集、max-model-len 4096、VLLM_CPU_KVCACHE_SPACE=32,LD_PRELOAD tcmalloc/libgomp。oneDNN(ACL backend)相对当前默认路径吞吐约高 78%。PR 未注明具体 tok/s 数值与权重精度(bf16/fp32)信息。

## 验证与回归风险

测试:`tests/kernels/test_onednn.py` 覆盖 linear 的 oneDNN 路径;作者报告在 Neoverse-V2 上 oneDNN 带/不带 ACL 编译时全部通过。回归风险:ACL fast path 仅在 oneDNN 编译了 ACL backend 时激活,无 ACL 时行为完全不变(仍走 PyTorch);x86 路径逻辑不变;`a.contiguous()` 在非连续输入时会引入一次拷贝;新增 repack 检查理论上在输入维度变化时触发额外 prepack,作者称实际不会成为性能问题。

## 关联

前序:PR #24150(weight prepack + oneDNN linear 直连路径,x86)。作者提到后续 PR 将考虑在合适平台上默认以 ACL backend 构建 oneDNN。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
