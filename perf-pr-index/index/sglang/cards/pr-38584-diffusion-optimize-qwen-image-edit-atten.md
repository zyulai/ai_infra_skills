# PR #38584: [Diffusion] Optimize Qwen-Image-Edit attention on Hopper

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-09-13 |
| 分类 | Attention / Kernel |
| Label | documentation, run-ci, diffusion, jit-kernel, run-ci-extra, mergeable |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 可见改 qwen_qkv_epilogue.cuh/.py、sdpa.py、qwen_image.py、registry 与测试 |
| 链接 | https://github.com/sgl-project/sglang/pull/38584 |

## 一句话结论

把 Qwen QKV epilogue 从 SM100 下放到 Hopper(SM90)并用 PyTorch 公开 varlen attention 一次处理 96 个 vision 窗口,H200 lossless E2E 32.8171s → 32.3141s(1.533%)。

## 动机

Qwen-Image-Edit 的 Hopper profile 显示:DiT 中重复的 QK norm/RoPE 与 joint-QKV 拼接,以及 vision encoder 每个 window-attention 层多达 96 次独立 SDPA 窗口调用(5,400 tokens / 96 ragged windows、16 heads、head dim 80)。DiT 实际形状 8,152 图像 tokens + 1,365/1,358 文本 tokens、24 heads、head dim 128。已有的 QKV epilogue kernel 只支持 SM100+,Hopper 走不了;SDPA 逐窗口调用则产生大量短 kernel。

## 优化点

两条线:(1) QKV epilogue——把 `try_fused_qwen_qkv_epilogue` 的能力门槛从 SM>=10 放宽到 SM>=9,支持未量化 BF16、batch=1、head dim 128 的 eligible Qwen attention;新增独立 image/text token stride(原来要求 img 与 txt 的 stride 一致);RoPE cache 载入向量化(每 lane 一次 `float2` 读两个相邻 cache 条目,消除 NCU 定位的 4 处 stride-2 标量载入造成的 7,309,056 个多余 L2 sectors),配对 V-head 拷贝(一个 warp 一次拷两个相邻 head,内存指令减半),Q/K/V 分派移入 grid.y(3 维 grid),显式 `__fmaf_rn`/`__fmul_rn` 保住原 RoPE 收缩与中间舍入(bit-exact);新增 mask/shard/对齐/16B-stride 回退与 `SGLANG_ENABLE_FUSED_QKNORM_ROPE` 环境开关,masked attention(attn_mask 非空)继续走普通归一化。(2) SDPA——`SDPAImpl.forward_varlen` 在严格谓词(SM90、Flash SDPA 推理、bf16/fp16、head_dim≤256 且为 8 倍数、cu_seqlens int32 contiguous、dropout=0、非编译/训练、API 存在)下改用公开的 `torch.nn.attention.varlen.varlen_attn`,一次调用消费全部 packed 窗口(含 ragged 与空窗口),无 token 重排或 padding;其余平台/子类保留原路径。共享 QKV kernel 同时服务既有 SM100 路径。

## 关键改动

```text
python/sglang/kernels/jit/csrc/diffusion/qwen_qkv_epilogue.cuh: 双 stride、float2 向量 RoPE 载入、V-head 配对拷贝、grid.y 分派
python/sglang/kernels/ops/diffusion/rope/qwen_qkv_epilogue_jit.py: SM>=9 门槛与对齐谓词
python/sglang/multimodal_gen/runtime/layers/attention/backends/sdpa.py: SM90 packed varlen_attn 快路径
python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py: 未量化 SM90 开启 fused epilogue
python/sglang/kernels/ops/diffusion/__init__.py: registry 能力 SM100+ → SM90+
```

RoPE 向量载入与保精度收缩:

```cpp
const auto cos_pair = __ldg(reinterpret_cast<const float2*>(cos_ptr) + lane);
const auto sin_pair = __ldg(reinterpret_cast<const float2*>(sin_ptr) + lane);
elems[i]     = __fmaf_rn(x, cos, -__fmul_rn(y, sin));
elems[i + 1] = __fmaf_rn(y, cos,  __fmul_rn(x, sin));
```

## 性能声明

条件:单卡 H200,BF16,1024x1024、50 步、guidance 4、embedded guidance 6、seed 42,`Qwen/Qwen-Image-Edit@ac7f9318`,eager 无 torch.compile,PyTorch 2.13.0+cu130,A-B-B-A 同卡:lossless engine E2E 32.8171 → 32.3141 s(**1.533%**);quality=high 32.2855 → 31.6036 s(2.112%,已有质量权衡,非新优化)。Torch 2.11 lossless 1.485%(低于 1.5% 验收阈值,未向上取整或与 2.13 合并)。去噪阶段 1.053%-1.639%。内核/trace 证据(独立小 trace 累计 GPU 时间):QK norm+RoPE+joint QKV(8152+1365 tokens)7 kernels/179.617µs → 1 kernel/109.023µs;packed vision SDPA(5400 tokens/96 窗口)96 kernels/517.984µs → 2 kernels/38.240µs;两步请求 image encoding 11,042 → 5,778 kernels(190.449 → 164.337 ms),denoising 7,113 → 5,673 kernels。NCU 诊断:vector 载入单独不改善时长(121.248 → 121.760µs),grid.y 提升后 112.544µs、occupancy 93.204%。

## 验证与回归风险

输出验证:16 张正式 PNG(4 次 A-B-B-A x 2 quality x 2 Torch 版本)在每种模式内 A/A、B/B、A/B 均字节一致,SSIM 1、最大像素误差 0;high 与 lossless 之间 SSIM 0.830311/PSNR 17.38 dB 对 baseline 与 candidate 完全一致,证明候选不引入额外质量变化。测试:35 个 H200 测试在 Torch 2.11.0+cu130 与 2.13.0+cu130 各通过(QKV 精确性、分离/打包 stride、奇数 head、mask、ragged/空窗口、dropout/训练回退等);28 个 CPU registry 测试;B200 CI 14 个 QKV epilogue 测试通过(修复了既有 SM100 路径的 B200 regression)。风险点:varlen_attn 谓词非常严格,依赖 PyTorch 版本存在该 API(缺失自动回退);packed SDPA 仅 SM90;epilogue 新增对齐谓词(16B token stride、8B cache 指针)会把原先能跑的某些布局改为回退;`SGLANG_ENABLE_FUSED_QKNORM_ROPE` 默认开启,对未量化 SM90 Qwen attention 全局生效;作者声明只测了一个 input/seed 每臂两次,不构成跨 Qwen 模型/输入的普适结论;AMD ROCm 7.2 CI 失败标记(x)。

## 关联

同区域优化:既有 `try_fused_qwen_qkv_epilogue`(SM100+,本 PR 的直接前序)、`try_fused_flux2_qkv_epilogue`(FLUX.2)、`fused_inplace_qknorm_rope`;registry/diffusion kernel 体系同批 PR #36592 与 #38044 均出自同作者。PR 正文未给出其他关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
