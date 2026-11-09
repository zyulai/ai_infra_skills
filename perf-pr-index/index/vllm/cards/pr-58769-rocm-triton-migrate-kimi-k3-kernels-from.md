# PR #58769: [ROCm][Triton] Migrate Kimi-K3 kernels from make_block_ptr to tensor descriptors

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @JadenMathias |
| 合并 | 2026-10-01 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, k3, kimi, ready, rocm |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files;diff 全文可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/58769 |

## 一句话结论

Kimi-K3 的 KDA Triton kernel 从已废弃的 `tl.make_block_ptr` 迁移到 `tl.make_tensor_descriptor`,MI355 上 decode TPOT/吞吐小幅改善约 0.7%。

## 动机

Kimi-K3 的 AMD 路径(`vllm/models/kimi_k3/amd/ops/third_party/kda/`)大量使用 Triton 的 `tl.make_block_ptr` API 描述分块加载/存储,该 API 已被官方标记为废弃,后续 Triton 版本(本测试环境为 3.8.0)推荐使用硬件感知的 tensor descriptor(TMA 风格描述符)。维持旧 API 不仅面临兼容性风险,也让 kernel 无法受益于描述符的地址计算优化。本 PR 做 API 迁移并验证精度与性能不掉。

## 优化点

纯 kernel 访存表达层迁移,不改算法:三个 KDA 文件(`chunk.py`、`chunk_intra.py`、`chunk_intra_token_parallel.py`)中所有 `tl.make_block_ptr` + `tl.load/tl.store(boundary_check=...)` 组合改写为 `make_tensor_descriptor(ptr, shape, strides, block_shape)` 创建描述符后用 `desc.load([off0, off1])` / `desc.store([off0, off1], val)` 访存,boundary 语义由描述符自动处理。涉及 `recompute_w_u_fwd_kernel`、`chunk_gla_fwd_kernel_o`、`kda_gate_chunk_cumsum_vector_kernel`、`kda_gate_fwd_kernel`、`chunk_kda_fwd_kernel_inter_solve_fused`(off-diagonal 三层嵌套的 q/k/g 块统一复用 3 个描述符,替代原先每迭代重建 block_ptr)、`chunk_kda_fwd_kernel_intra_sub_chunk` 等。`make_tensor_descriptor` 由 `vllm.third_party.flash_linear_attention.ops.op` 提供。一维小张量(beta 向量)则放弃 block_ptr 改为直接带 mask 的指针加载。纯机械等价改写,无量化、无 tile 尺寸变化。

## 关键改动

```text
vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py: recompute_w_u/chunk_gla_o/kda_gate 系列 kernel 迁移
vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra.py: intra/inter_solve_fused kernel 迁移
vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra_token_parallel.py: token-parallel kernel 迁移
```

迁移模式示例(节选):

```python
# before
p_A = tl.make_block_ptr(A + (bos*H + i_h)*BT, (T, BT), (H*BT, 1),
                        (i_t*BT, 0), (BT, BT), (1, 0))
b_A = tl.load(p_A, boundary_check=(0, 1))
# after
desc_A = make_tensor_descriptor(A + (bos*H + i_h)*BT, [T, BT], [H*BT, 1], [BT, BT])
b_A = desc_A.load([i_t * BT, 0])
```

## 性能声明

PR 正文数据(Kimi-K3,TP=8,MI355 gfx950;Triton 3.8.0+git669b31ac、aiter 0.1.23、torch 2.13.0+git733fca1;`VLLM_ROCM_USE_AITER=1`、cudagraph FULL_DECODE_ONLY、max-num-batched-tokens 4096):

- `vllm bench serve` 1k/1k 输入输出、并发 1、10 prompts(run 1 预热,取 2-3 平均):Mean TTFT 209.0 → 211.0 ms(+0.9%),Mean TPOT 15.55 → 15.44 ms(−0.7%),Median ITL 15.54 → 15.43 ms(−0.7%),P99 ITL 15.84 → 15.74 ms(−0.6%),Mean E2EL 16109 → 16001 ms(−0.7%),输出吞吐 63.57 → 64.00 tok/s(+0.7%)。
- 叠加 PR #59287 的新 Triton:1k/1k c=64 时 Mean TPOT 35.72 → 35.43 ms(−0.8%)、P99 ITL 38.79 → 37.41 ms(−3.6%)、吞吐 1748.1 → 1762.4 tok/s(+0.8%);8k/8k c=64 时 Mean TPOT 42.93 → 42.84 ms(−0.2%)、吞吐 1477.4 → 1480.5 tok/s(+0.2%),TTFT 持平。
- 精度:GSM8K 5-shot lm_eval 0.4.13,64 并发 1319 题 4m56s,flexible/strict exact_match 均 0.9560 ± 0.0056。

幅度均在 1% 量级,定位为等价迁移 + 轻微改善而非大幅提速。

## 验证与回归风险

验证为端到端基准 + 精度评测(上文数字),pack 内未见新增单元测试。回归风险:仅影响 Kimi-K3 AMD/ROCm 路径,且依赖 `make_tensor_descriptor` 在目标 Triton(3.8.x)的可用性;描述符的边界处理与旧 `boundary_check` 语义需在非整除 T 上等价(代码以 mask 化指针加载处理 beta 向量的尾部);CUDA 路径与其他模型不受影响。Triton 版本耦合是主要升级风险——旧 Triton 是否提供该 op,素材未注明。

## 关联

依赖: PR #59287(测试所用新 Triton 的引入 PR,性能表中单独列出叠加效果)。同区域:Kimi-K3/K3 相关 ROCm kernel PR(标签 k3/kimi),本批次内无其他同分类卡。
