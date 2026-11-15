# PR #67235: [Kernel][GLM-5.3-Flash] Fused in-place q/k RMSNorm + RoPE and residual add inside RMSNorm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gcanlin |
| 合并 | 2026-11-13 |
| 分类 | Attention / Kernel(tier: major) |
| Label | ready, glm |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/67235 |

## 一句话结论

GLM-5.3(`Glm5NextForConditionalGeneration`)vision blocks 用单次 `ops.fused_qk_norm_rope` 在 qkv 投影输出上原位完成 q/k RMSNorm + 2D RoPE,并把 residual add 折进下一个 fused add-RMSNorm:vision tower wall time −12% 到 −41%,E2E output throughput +4.3% / +2.3%、TTFT −10% / −13%。

## 动机

原 q/k 路径慢:Triton `fused_q_kv_rmsnorm` 是为 MLA 的 512–1,536 行宽写的,vision tower 的 64-element heads 下它每 head row 启动一个 program(4,096 patches 时 2 x 16 x tokens = 131K programs)。它是最大的 non-attention 开销:1,024-token 图像 10.4 ms GPU 时间中占 3.3 ms,超过全部 attention(2.5 ms)或全部 GEMM(2.6 ms)。原路径还包含两次 reshape copy、`torch.cat` 与独立 rotary kernel;residual add 也是独立 kernel。

## 优化点

- Fused q/k norm + RoPE:`ops.fused_qk_norm_rope` 已有 CUDA 与 ROCm 实现(head dim 64,NeoX style),在 qkv 投影输出上原位调用;2D RoPE 以每 forward 一次构建的 per-token `cos|sin` 表(由既有 `rotary_pos_emb_cos/sin` 元数据 `torch.cat` 而来,按 `arange` positions 索引)进入;encoder CUDA-graph buffers 不变。q/k 保持 qkv 的 view(v 本来就是);只有 FlashInfer 因其 `cu_seqlens` 布局需要 contiguous copy。`ApplyRotaryEmb` 与 `fused_q_kv_rmsnorm` 在该文件不再使用。
- Residual in RMSNorm:block 之间传 `(hidden, residual)`;下一 block 的 `norm1` 及最后一个 block 之后的 `post_layernorm` 做加法;第一个 block 以 `residual = None` 开始。
- 数值:kernel 把归一化后的 q/k 保持 fp32 进入 RoPE、只取整一次;此前中间会 round 到 bf16。逐 block 误差到 block 12 与 main 一致;tower 在 blocks 13–18 本身对 bf16 敏感(纯 bf16 PyTorch 参考在那些层距 fp32 达 27–31%),整体比 main 更接近 fp32 forward。

## 关键改动

```text
vllm/models/glm5next/common/multimodal.py: forward 改为 ops.fused_qk_norm_rope 原位调用 + cos|sin 拼接表 + residual 传入 norm1/post_layernorm
```

```python
x, _ = self.qkv(x)
num_heads = self.num_attention_heads_per_partition
# q/k RMSNorm and RoPE in place in qkv, in fp32 with one rounding.
ops.fused_qk_norm_rope(
    x.view(-1, x.shape[-1]),
    num_heads,
    num_heads,
    num_heads,
    self.head_dim,
    self.q_norm.variance_epsilon,
    self.q_norm.weight,
    self.k_norm.weight,
    rotary_pos_emb_cos_sin,
    is_neox=True,
    position_ids=rotary_positions,
)
```

## 性能声明

- Vision tower wall time −12% 到 −41%(on top of #67194,1x B200;最大收益在 256–1,024 tokens 的单图请求)。逐输入(两轮交错,各 20 迭代均值;#67194 wall/GPU → +fused q/k norm+RoPE → +residual in RMSNorm,wall vs #67194):1 图 256 tokens 18.52/3.46 → 12.51/2.48 → 11.71 ms(−37%);1 图 1,024 tokens 18.08/10.35 → 11.49/6.84 → 10.73 ms(−41%);1 图 4,096 tokens 59.06/58.82 → 48.00/47.77 → 47.87 ms(−19%);1 图 7,920 tokens(max)169.6/169.1 → 150.0/149.8 → 149.8 ms(−12%);16 图 x 256 tokens 31.75/31.48 → 19.73/20.75 → 19.54 ms(−38%);16 张 MMStar 真图 29.48/29.19 → 18.47/18.79 → 18.41 ms(−38%)。
- 端到端(#67194 + 本 PR vs main,glm-5.3-bf16 DEP8,8x B200 TP1/DP8/EP,无 MTP,prefix caching off,每请求一张随机图,256 text in / 128 out,c 128):1024x1024 图 1,024 prompts:output tok/s 3,357 → 3,501(+4.3%),total 46,342 → 48,331(+4.3%),TTFT mean/p90 1,435/2,093 → 1,292/1,908 ms(−10%/−9%);512x512 图 2,048 prompts:output 4,868 → 4,982(+2.3%),total 28,862 → 29,541(+2.4%),TTFT 726/966 → 635/804 ms(−13%/−17%)。同臂 repeat 间 TTFT 波动可达 20%。
- 数值(16 张 MMStar 图,对照同权重的 fp32 forward):main rel. L2 8.84e-2 / cosine 0.99609;+fused 8.27e-2 / 0.99658;+residual 8.13e-2 / 0.99670。MMStar(1,500 图像必答题,temp 0):main 0.778 → #67194+本 PR 0.787(n=1,500 标准误约 1.1 点,噪声内)。
- Residual 改动对 GPU 时间影响至多 2%,其收益是小单图的 launch overhead。大图上 FA4 虽未改变却慢 5–8%:短 memory-bound kernel 消失后 GPU 撞 1,000 W power cap,SM clock 从 1,938 降到 1,812 MHz(7,920-token 输入 400 迭代 200 ms 采样),总时间仍降 12%。Encoder CUDA graph 在 fused q/k 之上无益(TTFT 1,418/619 ms vs eager 1,340/584 ms,run-to-run spread 内)。

## 验证与回归风险

- Vision tower 单测环境:glm-5.3-bf16 权重(vision tower 同 GLM-5.3-Flash 的 BF16 tower),1x B200,eager(默认),FA4。
- `fused_qk_norm_rope` 有自己的 kernel 测试(`tests/kernels/core/test_fused_qk_norm_rope.py`);pre-commit 通过。
- CI:该模型只注册在 `tests/models/registry.py`,无 CI 测试在真实权重上跑这个 vision tower;`cudagraph_mm_encoder` 与 ROCm、FlashInfer ViT backend 未跑。
- 风险:数字在本 PR 叠加 #67194 时测得(含本改动的 arm 均含 #67194);rebase 到 `9e6815acab` 后未重跑,去掉 #67194 的组合 eager 路径与 main 相同但未在 GPU 上跑过;独立于 #67194 时两 PR 改 `Glm5NextVisionTransformer.forward` 相同行,后合并者需 rebase。

## 关联

- #67194(独立 PR 但编辑相同行;性能数字叠加其上测得)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
