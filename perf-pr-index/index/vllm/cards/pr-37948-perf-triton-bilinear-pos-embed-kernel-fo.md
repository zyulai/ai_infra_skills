# PR #37948: [Perf] triton bilinear_pos_embed kernel for ViT

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zhandaz |
| 合并 | 2026-04-01 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, ready, multi-modality, qwen, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files),实际改动为 1 个模型文件 + 新增 benchmark 与测试文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/37948 |

## 一句话结论

为 Qwen3-VL ViT 的双线性位置编码插值新增融合 Triton kernel,单次启动替代约 25 个 eager kernel,encoder 前向快 28%。

## 动机

`Qwen3_VisionTransformer` 的 `fast_pos_embed_interpolate` 原实现用 eager PyTorch 组合 linspace、meshgrid、索引、乘加、reshape、permute 等约 25 个小 CUDA kernel,每个 grid 都要重复启动一遍,CPU launch 开销主导了这一小算子的延迟。该工作来自 NVIDIA MLPerf Inference V6.0 提交,VLM 任务中 ViT 侧这段开销占比可观。

## 优化点

新增单个 Triton kernel 融合整段计算:每个 program 负责一个输出 token,在 index math 里直接完成双线性插值(预计算 `h_scale`/`w_scale` 代替 `linspace`,四个邻居 embedding 的 weight 转到输出 dtype 后做 multiply-accumulate,精度路径与 native 一致),并把 spatial-merge 重排(`spatial_merge_size` 分块重排)烘焙进索引计算,输出无需再 permute 即可直接加到 patch embedding 上。时间维 t 通过外层重复处理。`HAS_TRITON` 不可用时回退到保留原实现的 `pos_embed_interpolate_native`,行为不变。所有共享 `Qwen3_VisionTransformer` 的四个 Qwen3 VL 变体(`qwen3_vl`、`qwen3_5`、`qwen3_vl_moe`、`qwen3_5_moe`)自动受益。

## 关键改动

```text
vllm/model_executor/models/qwen3_vl.py: 新增 _bilinear_pos_embed_kernel(Triton)、triton_pos_embed_interpolate、pos_embed_interpolate_native;fast_pos_embed_interpolate 按 HAS_TRITON 分发
benchmarks/kernels/benchmark_vit_bilinear_pos_embed.py: 新增 kernel 基准(triton.testing.do_bench,7 种 grid 配置)
tests/kernels/core/test_vit_bilinear_pos_embed.py: 新增精度测试(8 种 grid x 2 dtype + temporal repeat)
```

关键 kernel 索引逻辑(spatial-merge 烘焙进 index math):

```python
block_idx = spatial_idx // (M_SIZE * M_SIZE)
local_idx = spatial_idx % (M_SIZE * M_SIZE)
br, bc = block_idx // num_blocks_w, block_idx % num_blocks_w
lr, lc = local_idx // M_SIZE, local_idx % M_SIZE
row, col = br * M_SIZE + lr, bc * M_SIZE + lc
```

## 性能声明

硬件 NVIDIA GB200,bfloat16:

- Kernel 级(单 grid,`triton.testing.do_bench`):grid (1,48,48) 419 → 8.2 us(51.3x);(1,64,64) 408 → 10.3 us(39.6x);(1,128,128) 427 → 19.2 us(22.3x)
- E2E(`vllm bench mm-processor`,Qwen/Qwen3-VL-8B-Instruct,20 prompts,random-input-len 512/random-output-len 128,每 prompt 限 3 图,bucket (256,256,1)):encoder_forward 均值 4.85 → 3.49 ms(28% faster);E2E 延迟均值 763.70 → 728.72 ms(4.6% faster)

精度:18/18 测试通过,float32 max diff < 5e-5,bfloat16 max diff < 1e-2(与真实 Qwen3-VL-8B-Instruct pos_embed 权重比对);MLPerf Inference VLM 提交期间在 Shopify 数据集上未观察到 `nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4` 精度损失。

## 验证与回归风险

新增 `tests/kernels/core/test_vit_bilinear_pos_embed.py`:4 方形 + 4 非方形 grid、fp32/bf16 双 dtype 与 native 实现逐元素比对(容差覆盖 Triton 预计算 scale 与 `torch.linspace` 的单 ULP 差异),temporal repeat 测试断言 t>1 输出与 t=1 repeat 逐位一致(atol=rtol=0)。风险面:仅触及 Qwen3-VL 系 ViT 位置编码路径;Triton 不可用平台走 native fallback;h/w 需被 spatial_merge_size 整除(原有约束,assert 保留)。

## 关联

同区域:Qwen3-VL 系多模态优化见 pr-22742(Qwen2.5-VL DP for ViT)、pr-61458(batch PaddleOCR VL vision encoder);Qwen3-VL 模型支持本身见相关 model PR。同分类(Attention / Kernel)其他 PR 见索引。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
