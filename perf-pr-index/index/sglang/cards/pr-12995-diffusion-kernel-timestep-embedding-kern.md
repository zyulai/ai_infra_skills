# PR #12995: [Diffusion] kernel: timestep embedding kernel implementation

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @66RING |
| 合并 | 2025-12-19 |
| 分类 | Attention / Kernel |
| Label | sgl-kernel, run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 + / - 数字);diff 含 7 个文件,其中 2 个新文件(kernel .cu 与测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/12995 |

## 一句话结论

为 diffusion 的 `timestep_embedding` 写融合 CUDA kernel(log/exp/cos/sin 单 kernel + float4 128bit 存储),微基准平均 4.5x,大 shape 达 H200 峰值带宽 88.75%。

## 动机

multimodal_gen 的 timestep embedding(`visual_embedding.py` 引用的 diffusers 实现)在 Python/PyTorch 侧由多次独立算子完成:`log`/`exp` 计算频率、`cos`/`sin` 计算嵌入、多次 elementwise 与拼接,小张量(batch × dim 通常很小)下 kernel 启动与中间张量开销占主导;它处在每个 denoise step 的 TimestepPreparationStage,所有 diffusion 模型(Wan、HunyuanVideo、FLUX 等)都要执行(关联 issue #12992、#12799)。PR 正文未附逐算子 profile,以"融合为单 kernel + 向量化存储"为方案。

## 优化点

新增 sgl-kernel CUDA 算子 `timestep_embedding`(csrc/sgl_diffusion/elementwise/timestep_embedding.cu):(1) 融合——频率 `scale * t * exp(neg_log_max_period * idx)`、`cosf`/`sinf` 与输出写回在单 kernel 内完成,`neg_log_max_period` 在 host 预计算;(2) 按 4 对齐使用 `float4` 128-bit 向量化存储(STG.128),host 校验 `dim % 8 == 0`、输出必须 float32 且 contiguous、t 为 1D;(3) `flip_sin_to_cos` 编译期模板分叉,输出 buffer 由 caller 传入(out 变体);线程组织为每行 `min(1024, half_dim/4)` 线程、CTA 纵向叠多行。Python 侧在 `visual_embedding.py` 定义 `Timesteps(_Timesteps)` 子类把 diffusers 前向替换为 sgl_kernel 调用,`CombinedTimestepGuidanceTextProjEmbeddings`/`CombinedTimestepTextProjEmbeddings` 重写 `__init__` 让 `time_proj` 用 sgld op、其余仍用 diffusers op;AT_DISPATCH 支持 int32/int64/bf16/fp16 输入。

## 关键改动

```text
sgl-kernel/csrc/sgl_diffusion/elementwise/timestep_embedding.cu: 新文件,融合 kernel + host launcher
sgl-kernel/csrc/common_extension.cc / include/sgl_kernel_ops.h / CMakeLists.txt: 算子注册
sgl-kernel/python/sgl_kernel/elementwise.py: timestep_embedding Python wrapper(out float32)
python/sglang/multimodal_gen/runtime/layers/visual_embedding.py: Timesteps 等子类切换到 CUDA 实现
sgl-kernel/tests/sgl_diffusion/test_timestep_embedding.py: 新文件,正确性 + 性能测试
```

关键机制(单 kernel 融合 + STG.128):

```cpp
vals.x = scale * t_val * expf(neg_log_max_period * __int2float_rn(thread_offset * 4 + 0));
...
sin_vals.x = cosf(vals.x); ...  *top_half = sin_vals;    // STG.128
cos_vals.x = sinf(vals.x); ...  *bottom_half = cos_vals; // STG.128
```

## 性能声明

PR 正文数据(H200):

- 微基准(20 次取均值):相对 torch native 平均 **4.5x**(Average Speedup 4.5196);例:bs=1/dim=256 0.112749→0.024861 ms,bs=8/dim=64 4.71x,bs=8/dim=128 0.116290→0.024661 ms。
- 带宽:dim=1536、batch=32k 档实测 4.26 TB/s,约 **88.75%** 的 H200 峰值带宽(4.8 TB/s)。
- E2E(`sglang generate`,单 prompt raccoon):Wan2.1-T2V-1.3B E2E 78604.33 → 75233.14 ms(fast math,-4.3%)/ 75500.19 ms(no fast math,-3.9%);FLUX.1-dev 10654.84 → 8947.77 ms(**-16.0%**);FLUX.2-dev 23582.89 → 22759.08 ms(-3.5%);HunyuanVideo -0.9% / +0.2%;Qwen-Image +1.2%(略慢)。

## 验证与回归风险

正确性测试覆盖 batch × dim × dtype 全组合(fast math 容差 1e-3,无 fast math 1e-4),另与 diffusers `get_timestep_embedding` 在 flip_sin_to_cos/downscale_freq_shift/scale 组合下对比;E2E 输出图对比五个模型。风险点:kernel 断言 dim 对齐、输出 float32、t 1D contiguous——非典型调用会 TORCH_CHECK 失败而非回退;`use_fast_math` 与否是编译期选择,精度容差依赖测试;E2E 各 stage 中 TextEncoding 等波动较大(-42.9% 等),与本 kernel 无直接因果,解读需谨慎;仅覆盖 `num_channels=256` 的默认 Timesteps 配置路径,自定义 num_channels 调用点未切换。

## 关联

前序 issue:#12992、#12799。同区域(diffusion kernel):#27590(fused FP8 GEMM)、#35981(Wan VAE RMSNorm row-ad)、#41830(Wan VAE decoder kernels)、#12995 的后续 #46043(diffusion registry)。同分类 kernel 类:#16397、#7884。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
