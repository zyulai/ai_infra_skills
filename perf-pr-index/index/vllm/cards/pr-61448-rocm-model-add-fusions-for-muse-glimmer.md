# PR #61448: [ROCm][Model] Add fusions for Muse Glimmer for AMD backend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @vgokhale |
| 合并 | 2026-11-06 |
| 分类 | Attention / Kernel (tier: major) |
| Label | rocm, ready, torch.compile |
| Diff | diffstat 缺失(pack 未提供 +None/-None across None);可见 diff 含 2 个新增 fusion pass、pass_manager/config 修改与 2 个测试文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/61448 |

## 一句话结论

为 Muse Glimmer 在 ROCm/AITER 上新增 sandwich-norm 与 QK norm+RoPE+KV-cache 两个 fusion pass,TPOT 与总吞吐约提升 20%,TTFT 约 5%。

## 动机

Muse 模型有 rmsnorm+add+rmsnorm 的 sandwich norm 模式,此前由 inductor 生成的单一 kernel 处理,但其融合 kernel 性能不佳;attention 走的是未充分优化的 vLLM ROCm FA backend。decode 中三个 pre-attention kernel 各约 4us、rmsnorm 6us、attention 高达 144us,kernel launch latency 在 decode 中被反复放大,值得做 compile-pass 级融合并切换到 AITER FA。

## 优化点

新增两个 torch.compile(Inductor post-grad)fusion pass:其一 MuseSandwichNormFusionPass 把 Gemma 风格 RMSNorm + residual add + 第二个 RMSNorm 匹配后替换为 AITER 的 fused_rmsnorm_add_rmsnorm 自定义 op;其二 MuseGlimmerQkRopeCacheFusionPass 用 pattern matcher 匹配 QK norm(无 weight 的 RMSNorm)+ q_scale 缩放 + RoPE + unified_kv_cache_update 的组合,替换为 AITER 的 fused_qkv_split_qk_norm_rope_cache 单 kernel。两者注册为新 custom op(fused_muse_qk_rope_cache)并仅在 ROCm + AITER 开启时启用(sandwich norm 还要求 gfx950、bf16、MuseGlimmer 架构)。同时新增 MuseGlimmerAttentionConfig:AITER MHA 可用时把 attention backend 默认设为 ROCM_AITER_FA,并打开 use_inductor_graph_partition 与 +rotary_embedding 以保证 QK norm/RoPE/KV 写入留在同一 Inductor graph 中。

## 关键改动

```text
vllm/compilation/passes/fusion/muse_glimmer_sandwich_norm_fusion.py: 新增,rmsnorm+add+rmsnorm -> aiter fused_rmsnorm_add_rmsnorm
vllm/compilation/passes/fusion/muse_glimmer_qk_rope_cache_fusion.py: 新增,qk_norm+rope+kv_cache -> aiter fused_qkv_split_qk_norm_rope_cache
vllm/compilation/passes/pass_manager.py: 按 rocm+AITER(+gfx950/bf16/MuseGlimmer)条件注册两个 pass
vllm/model_executor/models/config.py: MuseGlimmer 架构默认 ROCM_AITER_FA backend,强制 inductor graph partition
tests/compile/passes/test_muse_glimmer_sandwich_norm_fusion.py: 新增 pass 单测(gfx950)
```

## 性能声明

kernel 微基准(正文):fused sandwich-norm kernel decode 约 2us(较 inductor 提升 40%)、prefill 提升 30-40us/50%;融合后 pre-attention 4us(原 3x4us=12us)、rmsnorm 4us(vs 6,prefill 快约 30%)、attention 27us(vs 144)。E2E(A=基线,B=本 PR,列 concurrency/TTFT ms/TPOT ms/total tok/s):concurrency=1:TTFT 60.4→57.2,TPOT 13.5→11.0,147.5→181.9 tok/s;concurrency=128:TTFT 2125.1→1975.2,TPOT 24.9→20.5,9462.5→11361.2 tok/s。作者总结:TPOT 与 total throughput(decode-heavy 场景)约 20% 提升,TTFT 约 5% 提升。硬件型号正文未注明(测试路径限定 gfx950)。

## 验证与回归风险

新增 test_muse_glimmer_sandwich_norm_fusion.py(非 ROCm/未开 AITER/gfx950 之外 skip,验证 matched_count==3 与 aiter op 调用),扩展 test_rocm_aiter_qk_norm_rope_kvcache_fusion.py(5/512 token、KV cache 一致性)。改动被 gate 在 rocm+AITER、MuseGlimmerForCausalLM/ForConditionalGeneration、head_size=128、kv_cache_dtype auto 等条件下,理论上不影响其他后端;但 config.py 会为 MuseGlimmer 全局改 attention backend 与 custom_ops,非 gfx950 ROCm 卡上的 MuseGlimmer 行为变化需关注。

## 关联

同区域:既有的 QkNormRopeKvCacheFusionPass、RocmAiterSiluMulFp8GroupQuantFusionPass、MLADualRMSNormFusionPass(同一 pass_manager 注册体系)。
