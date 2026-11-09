# PR #39058: [Kernel] Implement CUDA kernel for ReLUSquaredActivation (relu^2)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tanish-malekar |
| 合并 | 2026-07-13 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, rocm, ready, nvidia |
| Diff | diffstat 缺失(素材未提供 additions/deletions/文件数);diff 全文含 7 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/39058 |

## 一句话结论

为 ReLUSquaredActivation (relu^2) 实现专用 CUDA kernel,替代双 kernel 的 eager 路径,A100 上比 torch.compile 融合版再快约 20%。

## 动机

`ReLUSquaredActivation.forward_cuda` 一直是 `# TODO: implement cuda kernels` 占位,直接回落 `forward_native`(即 `torch.square(F.relu(x))`,eager 下展开为两个独立 kernel launch)。在已有的自定义 CUDA 激活 kernel(gelu_new、gelu_fast、gelu_quick)中,`relu_squared` 是唯一未实现的一个。relu^2 激活出自 arXiv 2109.08668v2。作者经 PR 搜索确认无重复工作。

## 优化点

新增 `relu_squared_kernel<T>` device 函数(单精度浮点内计算 `relu(x)^2`),经既有 `LAUNCH_ACTIVATION_KERNEL` 宏与共享 `activation_kernel` 发射,完全复用 gelu 系 kernel 的模式,单次 launch 完成 relu + 平方融合。分发行为保持谨慎:`torch.compile`/Inductor 激活时 custom_ops 默认 `none`,仍走 Inductor 融合的 `forward_native`(行为不变);custom ops 开启(eager 模式)时走新 kernel 替代两个未融合 op;CPU/XPU 经基类回落 forward_native(本 PR 仅做 CUDA kernel)。`ReLUSquaredActivation.__init__` 在 `is_cuda_alike()` 时把 `self.op` 绑到 `torch.ops._C.relu_squared`,`forward_cuda` 写入预分配 out tensor。

## 关键改动

```text
csrc/libtorch_stable/activation_kernels.cu: relu_squared_kernel<T> 与 relu_squared host 入口(LAUNCH_ACTIVATION_KERNEL)
vllm/model_executor/layers/activation.py: ReLUSquaredActivation.forward_cuda 改调 torch.ops._C.relu_squared
csrc/libtorch_stable/{ops.h,torch_bindings.cpp} 与 csrc/ops.h: op 注册与绑定
benchmarks/kernels/benchmark_relu_squared.py: custom / native(eager)/ native(torch.compile)三方微基准
tests/kernels/core/test_activation.py: ReLUSquaredActivation 加入对照测试参数化
```

## 性能声明

微基准 `benchmarks/kernels/benchmark_relu_squared.py`,A100-SXM,bf16,`triton.testing.do_bench_cudagraph`(中位数,排除 per-launch overhead):现实内存受限尺寸下,自定义 kernel 比 torch.compile 融合的 native 快约 20%(例:128x1024x12288 为 3.84 ms vs 4.72 ms);比 eager native(未融合的 relu→square 双 kernel)快约 2x;≤64 tokens 的极小形状被噪声主导、大致持平。方法学注记:`do_bench_cudagraph` 不计 launch 开销,是 torch.compile 对比的公平基准;eager 模式无 CUDA graph,实际收益至少不低于所示数值(另省一次 launch)。结果图以图片形式给出,未附完整数值表。PR 未注明端到端模型级吞吐数据。

## 验证与回归风险

测试:`tests/kernels/core/test_activation.py -k relu` 18 passed / 180 deselected(ReLUSquaredActivation 加入 custom-op 对照参数化,与 `torch.ops._C.relu_squared` 逐 token 比对);`tests/model_executor/test_enabled_custom_ops.py` 21 passed / 6 skipped(ROCm-only)/ 0 failed。基准脚本对每个 config 先断言 kernel 输出与 `torch.square(F.relu(x))` 一致再计时。实测 RTX 3090 (sm_86),基准在 A100-SXM (sm_80)。风险面:共享 `activation_kernel` 的 per-token 偏移 `blockIdx.x * d` 用 32 位计算,>2^32 元素的张量会溢出——这是 gelu_* 系 kernel 的既有问题(基准 sweep 因此限制在 2^31 元素内),作者留待单独修复;AI 辅助开发(Claude),所有改动行经人工 review 与测试。

## 关联

同区域:激活 kernel 族 gelu_new / gelu_fast / gelu_quick(共用 LAUNCH_ACTIVATION_KERNEL 与 activation_kernel,32 位偏移溢出问题同源);relu^2 使用方为采用 ReLUSquaredActivation 的模型。pack 素材未列出明确的前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
