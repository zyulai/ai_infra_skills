# PR #27705: Fuse the DSA (V3.2, GLM-5.x) indexer Q/K paths into single kernels

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @b8zhong |
| 合并 | 2026-06-27 |
| 分类 | Attention / Kernel(major) |
| Label | high priority, lora, deepseek, run-ci, jit-kernel, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/27705 |

## 一句话结论

把 DSA indexer 的 Q/K 多步算子链(投影/归一化/RoPE/Hadamard/fp8 量化/cache 写)融合为单 kernel 并双流并发,B300 decode BS=1 提升约 10%。

## 动机

DeepSeek-V3.2/GLM-5.x 的 DSA 稀疏注意力 indexer 在每次 forward 中要跑一条多算子链:key 侧两个 GEMM(wk 投影 + weights_proj head-gate)加 LayerNorm、RoPE、(可选)Hadamard、fp8 量化、index cache 写入各为独立 op;query 侧 rope/Hadamard/量化/head-gate 缩放也是四个独立 op,cache store 还排在 key 工作的关键路径之后。启动开销与小 batch 下的 launch 间隙成为瓶颈,需要按算子融合与并发执行压缩。

## 优化点

五项改动:(1) key 投影与 head-gate 投影融合为一个 matmul——`wk_weights_proj`(输出 head_dim + n_heads,bf16)替代两个 ReplicatedLinear,前向后 split;(2) query 路径融合:把 `main_norm_rope.cuh` 的融合 Q kernel 模板泛化出 `kRopeFirst`(V3.2 是 [rope|nope] 布局,V4 是 [nope|rope])与 `kHadamard` 两个编译期开关,新增 `fused_q_indexer_rope_first_quant`(RoPE 前置 + fp8 量化 + head-gate 缩放,一个 kernel);(3) key 路径融合:新增 `deepseek_v32/indexer_k.cuh` 的 `fused_k_indexer_norm_rope(_store)` kernel——单头 LayerNorm(非 RMS)+ 前置 64 维 RoPE + fp8 激活量化 + paged index-k cache 写(132*page_size 字节/页:128 fp8 key + 4 fp32 scale)一次 launch 完成;不可用场景(非 CUDA/NPU、HIP,`_is_fp8_fnuz` 等)保留分离 kernel + store 的回退;(4) V3.2 去掉 Hadamard 旋转——H 为正交阵且同时作用于 q/k,对 indexer 得分是 logit-preserving,只影响 fp8 量化精度,故 V3.2 直接量化未旋转激活(开关保留,HIP 等路径仍可旋转);(5) key 与 query 双流并发:两阶段重叠(wq_b GEMM 与 wk_weights_proj GEMM 分别在 alt/current stream,随后融合 Q kernel 与"融合 K kernel + cache 写"并行),wait_stream 按 issue 顺序只等各自消费的 GEMM。可用 `SGLANG_DISABLE_DSA_INDEXER_FUSION=1` 完全回退。权重加载:融合 matmul 是 bf16,FP8 checkpoint 的 key 投影在加载时 dequant 到 bf16,NVFP4 checkpoint 本就是 bf16 直接拷贝。

## 关键改动

```text
python/sglang/jit_kernel/csrc/deepseek_v32/indexer_k.cuh: 新增;fused K(LayerNorm+RoPE+fp8 quant+paged store)CUDA kernel
python/sglang/jit_kernel/dsv32/elementwise.py: 新增;JIT 包装 fused_k_indexer_norm_rope(_store)
python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh: Q kernel 模板加 kRopeFirst/kHadamard,导出 V3.2 变体
python/sglang/srt/layers/attention/dsa/dsa_indexer.py: wk_weights_proj 融合线性、双流两阶段调度、融合 K 写 cache 分支
python/sglang/srt/environ.py: 新增 SGLANG_DISABLE_DSA_INDEXER_FUSION
```

```python
# dsa_indexer.py: 两阶段重叠,GEMM 与融合 kernel 交替占双流
current_stream = torch.cuda.current_stream()
self.alt_stream.wait_stream(current_stream)
with torch.cuda.stream(self.alt_stream):
    q = self.wq_b(q_lora)[0].view(-1, self.n_heads, self.head_dim)
kw, _ = self.wk_weights_proj(x)
key, weights_raw = kw.split([self.head_dim, self.n_heads], dim=-1)
current_stream.wait_stream(self.alt_stream)   # 各侧只等自己消费的 GEMM
```

## 性能声明

- B300,BS=1、0 context:decode 97.72 → 107.77 TPS(约 10%),基于正文给出的 server 命令(GLM FP8 checkpoint)。
- B300 + GLM FP8 checkpoint,BS=128:decode 约 3212.05 → 3418.63 TPS(约 +6.4%)。
- 正文另有 FP4/FP8 的 profiler 前后截图(文字数字未在正文中给出);未注明 TP/EP 配置与 prompt 长度等细节。

## 验证与回归风险

GPQA 精度(198 例 x repeat 8,temperature 1.0,top_p 0.95,thinking-mode glm-45):FP4 路径(nvidia/GLM-5.1-NVFP4,TP4/EP4)前后 mean 0.854 → 0.859;FP8 路径(zai-org/GLM-5.2-FP8,TP8/EP8)0.856 → 0.856,均无退化。风险:去掉 Hadamard 只在 V3.2/GLM-5.x 语义下成立(依赖 q/k 同时不旋转),其他模型走 `_maybe_rotate` 保持原行为;融合 K 写 cache 要求页布局与 `fused_store_index_cache.cuh` 一致且页大小为 2 的幂,`can_use_dsa_fused_store` 不满足时回退分离 kernel;FP8 checkpoint 加载时需把 key 投影 dequant 到 bf16(正文确认与 FP8 dequant 精确一致);HIP/NPU 走原 split 路径不受影响。

## 关联

vLLM 采用了类似方法(vllm-project/vllm#38928,正文注明);方法上复用 DSV4 已有的融合 norm/rope/quant kernel 家族并泛化出 V3.2 变体。pack 未列出其他前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
