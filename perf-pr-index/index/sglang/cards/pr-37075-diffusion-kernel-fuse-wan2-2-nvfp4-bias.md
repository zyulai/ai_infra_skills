# PR #37075: [Diffusion][Kernel] Fuse Wan2.2 NVFP4 bias + GELU on Blackwell

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-08-30 |
| 分类 | Quantization (tier: major) |
| Label | blackwell, run-ci, diffusion, jit-kernel, run-ci-extra, mergeable |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/37075 |

## 一句话结论

把 Wan2.2 NVFP4 MLP 的 fc_in bias 加法与 tanh GELU 融合为一个 JIT CUDA kernel,GB300 e2e 快约 2.9%。

## 动机

Wan2.2 ModelOpt NVFP4 MLP 的 `fc_in` 热路径此前执行三步:`NVFP4 GEMM → BF16 bias add → tanh GELU`。在 GB300 上,bias add 与 GELU 是 memory-bound,占去可观的一部分去噪时间。本 PR 把后两步融合为一个 kernel,同时保留 eager 半精度的舍入边界(bias 相加结果先舍回输入 dtype 再进 GELU)。

## 优化点

新增轻量 JIT CUDA kernel `bias_gelu_tanh`:支持 FP16/BF16;Blackwell 前用 16-byte 向量、Blackwell 上用 32-byte 向量;occupancy 封顶的 persistent launch 几何;支持 PDL wait/trigger。集成方式是 request-scoped 融合站点:`_WanGELUMLP` 仅当 `fc_in` 的 quant_method 为 `ModelOptFp4LinearMethod` 时标记站点;`ColumnParallelLinear` 在融合挂载时置 `skip_bias_add`,NVFP4 GEMM 返回输出与延迟 bias,再由 fused kernel 一次完成 bias+GELU。挂载/卸载接入既有 diffusion `quality=high` 生命周期(`DenoisingStage._maybe_toggle_quality_fusions`),默认 `quality=lossless` 路径与被 checkpoint 排除的 linear 保持不变。

## 关键改动

```text
python/sglang/kernels/jit/csrc/elementwise/bias_gelu.cuh: JIT CUDA bias+approx-GELU kernel(向量化/PDL/persistent grid)
python/sglang/kernels/ops/elementwise/bias_gelu.py: JIT 加载与 register_custom_op 封装 bias_gelu_tanh
python/sglang/kernels/ops/diffusion/sites/nvfp4_bias_gelu_site.py: quality 门控的 mount/unmount 站点与资格校验
python/sglang/multimodal_gen/runtime/layers/mlp.py: _apply_activation 增加 fused bias+GELU 分支
python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py: _WanGELUMLP 标记站点并消费延迟 bias
```

```cpp
const float x_f32 = device::cast<fp32_t>(x[i]);
const float bias_f32 = device::cast<fp32_t>(b[i]);
const T biased = device::cast<T>(x_f32 + bias_f32);   // 半精度舍入边界
result[i] = device::cast<T>(gelu_tanh(device::cast<fp32_t>(biased)));
```

## 性能声明

E2E(GB300 单卡,eager native SGLang,`nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4`,832x480、81 帧、40 步、guidance 4、seed 42,基线 `89816a21`):denoise 均值 71.737731 s → 69.596439 s(**2.984889% faster**);端到端 72.733052 s → 70.591010 s(2.945073% faster);峰值 reserved 显存不变(55.580 GiB)。5 个去噪 timestep 剖析:add+GELU 链 1,771.077 ms → 1,382.591 ms(节省 388.486 ms),BF16 add 1,553.063→1,138.976 ms,原生 GELU 218.014→36.028 ms,fused kernel 207.587 ms,CUDA events 22,688→22,282。

微基准(GB300,rebase 到 `032fe91bf1`):32760x5120 JIT 171.9725 µs vs PyTorch 498.9286 µs(2.90x);32760x13824 457.6064 µs vs 1338.7175 µs(2.93x)。

## 验证与回归风险

注册测试 `test_bias_gelu.py`(FP16/BF16 多形状与 `F.gelu(input+bias, approximate="tanh")` 逐位一致,含生产 shape `(1, 4096, 13824)`;拒绝不整除宽度与不支持 dtype)、`test_wan_gelu_mlp.py` 的 mount/unmount 与 excluded-linear 回归(需 Blackwell);注册 benchmark `bench_bias_gelu.py`。GB300 上 11 个聚焦 GPU 测试通过;`quality=high` 输出 MP4 与 main 的 SHA256 一致,`quality=lossless` 与未改基线逐字节一致。风险:融合只在 `quality=high` + ModelOpt NVFP4 的合格站点生效,未合格站点走原 eager 路径;kernel 要求 hidden_dim 按向量宽度整除。

## 关联

与既有的 `fused_linear_gelu`(cublasLt GELU epilogue)站点共用 quality 门控机制;正文未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
