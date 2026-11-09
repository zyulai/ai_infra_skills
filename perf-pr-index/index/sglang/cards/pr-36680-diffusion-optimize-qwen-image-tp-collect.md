# PR #36680: [Diffusion] Optimize Qwen-Image TP collectives and attention

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-09-01 |
| 分类 | Attention / Kernel(major) |
| Label | documentation, run-ci, diffusion, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/36680 |

## 一句话结论

Qwen-Image TP 路径三处优化(CARv2 大 collective、segmented QKV 打包、FA3 静态调度):24 MiB 集合通信快 21.5%,Qwen-Image-2512 TP2 去噪快 8.58%。

## 动机

profiler 显示 Qwen-Image-2512 TP2 路径相对 vLLM-Omni 有三个独立缺口:(1) diffusion 直接实例化 legacy custom all-reduce,24 MiB 的 row-parallel 输出超过默认 16 MiB workspace,回退到 NCCL;(2) 每个 attention block 都先物化 text/image 拼接的 joint Q/K/V 再打包,采样窗口内有 1,212 次 `CatArray` launch、耗时 10.768 ms;(3) batch-one 的 dense 序列被选到 FA3 `VarlenDynamic`(159.403 µs/call),而 vLLM 用 `StaticPersistent`(147.402 µs/call)。

## 优化点

三项对应改动:(1) CUDA diffusion TP collective 改走 SRT custom-all-reduce dispatcher,`CustomAllReduceV2` 的 max_size 提到 32 MiB(`_DIFFUSION_CUSTOM_AR_MAX_SIZE`),使 24 MiB 的 Qwen-Image row-parallel 输出留在调优 kernel 而非 NCCL;all_reduce 分支简化为直接取 `custom_all_reduce` 的返回值。(2) 打包链路:`fused_pack_segmented_qkv` 分段 Triton kernel 直接从"虚拟 [prefix, main] 序列"按 indices gather Q/K/V,与"先三路 dense 拼接再 pack"bitwise 等价但省掉三次拼接;非量化 Qwen 的 added-text Q/K/V 权重保持内存 packed,新 `qwen_image_added_qkv_site` 按 request quality 挂载——`quality=lossless`(默认)把 packed 权重切片当三个独立 GEMM 跑以保持参考 reduction 顺序(bit-exact),`quality=high` 才挂载单 GEMM(改变 reduction 结合序);checkpoint 加载的 quant 推断用独立 `quant_param_names_mapping` 避免 ModelOpt checkpoint 误套融合映射。(3) attention:batch-one 或全有效的 packed 序列用空 `cu_seqlens` 走 FA3 静态 persistent 调度,必要时 `fused_scatter_to_padded` 散回 padded/BCG bucket。所有新路径保留对不支持形状/设备/量化/非 CUDA 平台的回退。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/distributed/group_coordinator.py: CARv2 dispatcher + 32 MiB diffusion workspace
python/sglang/kernels/ops/diffusion/layout/varlen_pack_pad_triton.py: 新增 fused_pack_segmented_qkv(免物化拼接)
python/sglang/kernels/ops/diffusion/sites/qwen_image_added_qkv_site.py: 新增;request-scoped quality=high 融合挂载
python/sglang/multimodal_gen/runtime/layers/attention/layer.py: segmented prefix 输入 + FA3 静态持久调度分支
python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py: added-QKV packed 权重 + lossless 三 GEMM 拆分
```

```python
# varlen_pack_pad_triton.py: 直接从虚拟 [prefix, main] 序列 gather,免物化拼接
from_prefix = row_in_batch < PREFIX_ROWS
q_val = tl.load(Q_prefix_ptr + prefix_offset, mask=prefix_mask, other=0.0)
q_val += tl.load(Q_main_ptr + main_offset, mask=main_mask, other=0.0)
...
tl.store(Q_unpad_ptr + dst_offset, q_val, mask=col_mask)
```

## 性能声明

微基准(H200):24 MiB collective NCCL 104.196 µs → V2 graph one-shot pull 81.754 µs(21.5% faster);segmented pack 47.7293 → 21.0982 µs(batch 1)、17.8346 → 7.7642 µs(batch 2);FA3 `VarlenDynamic` 159.403 → `StaticPersistent` 147.063 µs;`quality=high` 融合 added-QKV 去除 1,212 个小 split-K GEMM,每去噪步省约 0.43 ms。

跨模型 H200 ABBA 审计(base `a7e3f590c` vs PR `8e056cfbf`,同一 H200 对,ABBA 顺序,两轮中位数,eager):Qwen-Image-2512 TP2+BCG 去噪 8.5406 → 7.8657 s(+8.58%,SSIM 0.9713,非 bit-exact);Qwen-Image 1GPU+BCG +2.39%(bit-exact);Z-Image-Turbo TP2 +5.05%;ERNIE-Image +1.73%;其余 Qwen-Image-Edit 系列、FireRed、Cosmos3、Z-Image 等中性(±0.7% 内,多数 bit-exact)。

vLLM-Omni 对比(独立 lane,Qwen-Image-2512 1024x1024/50 steps/guidance 1/TP2/BCG/no CFG):SGLang 组合候选 median client latency 4.101 s、denoise step 76.9 ms,比最快 vLLM-Omni(4.442 s)快 7.7%。

## 验证与回归风险

新增 fused-QKV/BCG 捕获测试(H200,12 passed + 3 subtests)、quality gate 切换测试(GB300,14 passed)、segmented pack/layout 测试(16 passed);11 条受影响 lane Base vs PR bit-exact,Qwen-Image-2512(SSIM 0.9713/PSNR 30.15 dB)与 Layered(min SSIM 0.9891)非 bit-exact 但各自稳定;`quality=high` 对未融合基线 PSNR 40.55 dB / MAE 1.393/255。风险:BF16 added-QKV 融合改变 reduction 结合序(已隔离到 quality=high);CARv2/32 MiB workspace 影响所有 TP>1 的 CUDA diffusion preset(FLUX.1/2、Qwen 系、Z-Image-Turbo、Cosmos3、MiniMax-H3 等);FLUX/Krea/Ideogram 因 gated repo 403 未在本轮审计执行;BCG Qwen-Image-2512 配方文档化(BCG 捕获每 GPU 多约 5 GB 峰值显存,SSIM 0.984/PSNR 39.7 dB 非 bit-exact)。

## 关联

Supersedes #36693 与 #36694(本 PR 整合两者);evidence branch `pr-artifacts/36680-h200-all-models`。pack 未列出其他同分类 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
