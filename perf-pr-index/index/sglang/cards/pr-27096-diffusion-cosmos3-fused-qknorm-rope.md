# PR #27096: [diffusion] Cosmos3 fused qknorm rope

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @qimcis |
| 合并 | 2026-06-06 |
| 分类 | Attention / Kernel (tier: major) |
| Label | run-ci, diffusion, jit-kernel |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,约 5 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/27096 |

## 一句话结论

Cosmos3 复用已有 fused QK-norm+RoPE kernel,该段 microbenchmark 提速 4x,E2E 去噪阶段缩短 3.46%。

## 动机

Cosmos3 之前在 UND causal-attention 路径和 GEN cross-attention 路径上,都把 Q/K RMSNorm 与 Qwen3-style mRoPE 作为独立算子执行:先 `apply_qk_norm` 再 `qwen3_apply_rotary_pos_emb`(基于预计算 cos/sin 张量)。视频生成的序列很长(batch=1 时 seq_len 达 28520),这段两步操作的额外访存和 kernel launch 开销占比可观。仓库里已有针对 Qwen3 half-split RoPE 约定的 fused QK-norm+RoPE kernel,Cosmos3 没有复用。

## 优化点

把 RoPE 的输入从 "(cos, sin) 张量对" 改为 "cos_sin_cache 表 + positions 索引",从而能直接接入已有 fused kernel:(1) `Qwen3VLTextRotaryEmbedding` 拆出 `_compute_interleaved_freqs`,新增 `build_rope_cache_inputs`,把 mRoPE 的 cos/sin 拼成 `[rows, head_dim]` 的 cos_sin_cache,positions 为恒等行号;(2) `apply_flashinfer_rope_qk_inplace` 放宽为支持 q/k head 数不同(q_heads != k_heads)、rope_dim 为 head_dim 前缀(partial rope)的场景,Triton fallback 同步支持;(3) Cosmos3 的 UND 与 GEN 路径改用 `apply_qk_norm_rope`(fused,受 `SGLANG_ENABLE_FUSED_QKNORM_ROPE` 门控,默认开)或 split fallback `_apply_qwen3_qk_norm_rope_split`;模型缓存的从 cos/sin 张量变为 position_ids,SP 分片时 pad/切分也改在 position_ids 上做;(4) GEN 路径用 `use_fused_qk_norm_rope = T > 1` 选择 fused 或 split。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py: UND/GEN 两路径接入 fused QK-norm+RoPE,缓存 position_ids 取代 cos/sin。
python/sglang/multimodal_gen/runtime/layers/rotary_embedding/mrope.py: 新增 build_rope_cache_inputs 生成 cos_sin_cache+positions。
python/sglang/multimodal_gen/runtime/layers/rotary_embedding/utils.py: FlashInfer/Triton RoPE 支持 q/k 头数不同与 partial rope_dim。
python/sglang/multimodal_gen/runtime/layers/layernorm.py: apply_qk_norm_rope 放宽形状校验(q/k 头数可不同)并补 device/dtype 检查。
python/sglang/multimodal_gen/runtime/loader/component_loaders/vocoder_loader.py: vocoder _class_name 缺失时回退 pipeline 架构名(顺带小修)。
```

## 性能声明

kernel microbenchmark(1x H200,ncu,main 分支 commit `83bc7766`;batch=1, seq_len=28520, q_heads=32, kv_heads=8, head_dim=128, bf16):Main split QK-norm + Qwen3 RoPE 平均 1923.93 us,Patched fused 480.88 us,即该段 4x speedup。E2E(`nvidia/Cosmos3-Nano`,1280x720,121 帧 @24FPS,35 步,seed 42,`--performance-mode manual`,dit/vae cpu offload 关闭,warmup 开启):warmup-excluded total 102.53s -> 99.29s(-3.27%),denoising stage 96.56s -> 93.33s(-3.46%),decode stage 5.59s -> 5.58s(持平)。

## 验证与回归风险

精度验证为视觉对比:同 prompt("A curious raccoon walking through a quiet forest")、同 seed 的生成视频前后链接,未提供数值指标;单测 checklist 未勾选。PR CI:Base Test 通过,Extra Test 失败(pack 中 CI 状态标注)。回归风险:(1) `use_fused_qk_norm_rope = T > 1`,单帧(T==1)走 split 路径,两路径行为需各自验证;(2) `apply_flashinfer_rope_qk_inplace` 放宽约束后,FlashInfer 路径仍要求 q_heads == k_heads(否则走 Triton fallback),Cosmos3 的 q=32/kv=8 头配置实际命中 fallback/新校验分支;(3) layernorm.py 校验收紧(rope_dim 必须为偶数且 <= head_dim)可能影响其他 multimodal 模型的既有调用;(4) RoPE 缓存从 cos/sin 改为 position_ids,`reset_cache` 与 `cached_gen_rope_inputs` 的失效逻辑需与 text/visual cache key 一致。

## 关联

同区域 fused QK-norm+RoPE:PR #29708(LTX2 QKNorm split-RoPE CUDA fast path,本批次内);diffusion kernel 优化如 PR #12995、PR #20395。本卡无直接前序 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
