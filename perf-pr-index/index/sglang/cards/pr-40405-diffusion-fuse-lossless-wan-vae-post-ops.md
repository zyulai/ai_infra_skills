# PR #40405: [Diffusion] Fuse lossless Wan VAE post-ops for LongLive 2 I2V

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-09-24 |
| 分类 | Model-specific Fast Path (major) |
| Label | documentation, run-ci, diffusion, jit-kernel, mergeable |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 含 5 个文件:JIT CUDA 头文件、Python op 封装、VAE wrapper、测试与文档 |
| 链接 | https://github.com/sgl-project/sglang/pull/40405 |

## 一句话结论

把 LongLive 2 Wan VAE 解码器 lossless 档的 5 个 FP32 后处理算子融合为一个 JIT CUDA kernel,I2V worker E2E 提速约 5.5-5.9%。

## 动机

LongLive 2 的 lossless Wan VAE 解码器在归一化之后要执行五个独立的 FP32 pointwise pass(divide、scale、affine 乘法、bias 加法、SiLU)。原生 I2V profile 下,464 个 norm/SiLU scope 分布在 3248 个 kernel 上,累计耗时 690.82 ms。已有的 full RMSNorm+SiLU 融合(#33546、#35981)走的是非 bit-exact 路径、被 quality gate 挡住,lossless 默认档只能跑原生逐算子链,存在明显的 kernel launch 与 DRAM 往返浪费。

## 优化点

保留原生 autocast 感知的 L2 归一化 reduction 与 epsilon clamp 不动,只把其后的 FP32 divide/scale/affine/bias/SiLU 融合进一个 JIT CUDA kernel;输入与 gamma 可为 BF16,归一化中间量与输出保持 FP32。kernel 显式使用 FP32 round-to-nearest 指令(`__fdiv_rn`/`__fmul_rn`/`__fadd_rn`)与 `expf`,编译时关闭 FMA contraction 与 fast math,保证与 aten 逐位一致。同时支持 dense NCDHW 与 channels-last-3D 两种布局。接入侧通过 `BitExactFusionGate` 做 per-signature 整数位验证:首次调用与参考实现逐 bit 比对,不匹配则永久回退原生路径;不支持 dtype/layout、梯度开启、torch.compile 与 CUDA graph capture 调用均保留原生路径。不支持的 post-op 输入会复用已计算好的分母,避免重复归一化。

## 关键改动

```text
python/sglang/kernels/jit/csrc/elementwise/wan_norm_silu_post.cuh: 新增 JIT CUDA kernel,按 NCDHW/channels-last 两个模板分支做 128bit 向量化后处理融合
python/sglang/kernels/ops/diffusion/norm/wan_norm_silu_post.py: Python 侧 JIT 加载、can_use 守卫与 custom op 注册
python/sglang/multimodal_gen/runtime/models/vaes/wan_vae_cuda_opt.py: FusedWanRMSNormSiLU 增加 lossless post-op 分支与 BitExactFusionGate 验证回退
test/registered/kernels/ops/diffusion/test_wan_norm_silu_post.py: autocast/layout/affine、数值边界与 graph replay 测试
docs/docs/sglang-diffusion/fused_kernels.mdx: 融合覆盖文档更新
```

核心计算片段(每向量 4 元素,FP32 逐位精确):

```cuda
float y = __fdiv_rn(device::cast<float>(x[i]), denom);
y = __fmul_rn(y, scale);
y = __fmul_rn(y, weight);
y = __fadd_rn(y, offset);
result[i] = __fdiv_rn(y, __fadd_rn(1.f, expf(-y)));
```

## 性能声明

- E2E(1×H200/TP1,`Rabinovich/LongLive-2.0-5B-Diffusers`,BF16 DiT 未量化,native VAE/autocast 精度,4 步、CFG1、61 帧@24fps,ABBA 顺序):I2V worker E2E 四组全部降低 5.53-5.88%(如 5.6111→5.2703 s,6.07%);代表性 I2V 客户端 E2E(含视频保存)5.980→5.665 s(5.27%)。Denoise 阶段基本不变(-0.20%~0.47%),符合优化位于 VAE 解码的预期。T2V worker E2E 提升 5.01-6.10%,但客户端出现未定位的保存后延迟,PR 明确不做 T2V 客户端 E2E 提速声明。
- Profile:I2V 归因 norm/SiLU kernel 从 3248 个/690.82 ms 降到 1834 个/275.26 ms(含首次参考校验与回退调用)。
- NCU 单独测量五算子链:五 kernel 合计 1599.136 µs vs 单融合 kernel 315.104 µs,即 5.07× 快;DRAM 读写总量减少 84.33%(读 1,842,186,240B→206,129,664B,写 1,930,885,632B→385,019,392B)。
- Marker 基准(中位数):如 N=1,C=1024,T=2,H=60,W=104 channels-last-3D 下 208.448→52.800 µs。
- 输出无损:42 个 lossless 视频逐字节一致,61/61 帧解码后 SSIM=1、PSNR=∞。
- 峰值 reserved 内存 59.6426→61.0195 GiB(+1.377 GiB)。

## 验证与回归风险

新增 `test_wan_norm_silu_post.py` 覆盖全部 65,280 个有限 BF16 位型、FP32 SiLU/subnormal 边界、BF16/FP32 输入与 affine 组合、两种布局、非整数 sqrt(C)、零范数、输入/affine 变更后的 CUDA graph replay、首见不匹配回退、dtype/layout 守卫、分母复用、compile guard、梯度与 import 隔离;H200 上 9 个 pytest case 加 16 个 subtest 通过。PR 还提供了输出对比、质量矩阵与 CI 跟进(合并 main 后 CPU fixture 49 tests + 57 subtests 通过)。风险面:共享 wrapper 同时服务其他 Wan/Qwen VAE,其他模型家族只受首见精确性 gate 保护、未被 benchmark 覆盖;BF16 输入依赖 CUDA autocast 行为;HIP 与梯度路径显式排除。CI 状态中 AMD ROCm 10 为失败,workflow 因无关 CPU fixture 缺 `split_index` 报红。

## 关联

前序/同区域:#33546、#35981(Wan full-norm 融合,quality gate 路径)、#38650(conv-bias/full-norm tiling,开放中)。本卡走 lossless post-only 路线,与上述 full 融合互补。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
