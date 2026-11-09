# PR #24696: [Gemma4] Optimize Gemm4 with fused Q/K/V RMSNorm + per-expert FP8 ckpt loader

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-05-10 |
| 分类 | MoE (tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/24696 |

## 一句话结论

Gemma4 attention 的 Q/K/V 三次 RMSNorm 合并为一次 Triton launch(microbench 最高 3.1x),并修复 per-expert FP8 checkpoint 静默加载失败。

## 动机

Gemma4 FP8-MoE 路径上两个不相关问题:(1) attention RMSNorm 是 launch-bound——每层发三次独立 sgl-kernel `rmsnorm`(q/k/v),小 token 数下每 launch 的 host 开销占主导,三次连续 launch 约 60 微秒,与 M 无关,而实际计算量很小;(2) compressed-tensors per-expert FP8 checkpoint(如 `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`)按 `experts.<id>.{gate,up,down}_proj.{weight,weight_scale}` 存储,现有 loader 只识别 stacked `experts.gate_up_proj`/`experts.down_proj` 布局,所有 routed expert 被静默跳过,零初始化导致服务能启动但生成全是 `<pad>`、GSM8K 为 0.0。

## 优化点

(1) 新 Triton kernel `gemma_qkv_rmsnorm`:单次 launch 对全部 q/k/v head 就地计算 `x * rsqrt(mean(x^2) + eps) * w`,grid 为 `(M,)`、`BLOCK = next_pow2(head_dim)`、`tl.static_range` 遍历 head;直接作用于 `qkv.split(...)` 的 strided view(用 `stride_*_m`),无需 `.contiguous()` 拷贝;V 路径跳过权重乘(`with_scale=False` 时权重恒为 1);KV-shared 层只归一化 Q。`Gemma4Attention.forward` 的 fused 路径有保守前置条件:CUDA、`q_norm/k_norm.scale_shift == 0`、`not v_norm.with_scale`,非标准配置与 KV-shared 层其余情况保留原三 norm 路径。(2) `load_weights` 新增正则分支,把 per-expert key 映射到 fused FusedMoE 参数(`w13_weight` 的 w1/w3 shard、`w2_weight` 的 w2 shard)与 expert id 后经标准 `param.weight_loader` 分发;regex 命中但 fused 参数缺失时发 warning 而非静默跳过。

## 关键改动

```text
python/sglang/srt/layers/gemma4_fused_ops.py: 新增 _gemma_qkv_rmsnorm_kernel 与 gemma_qkv_rmsnorm 入口
python/sglang/srt/models/gemma4_causal.py: attention forward 接入 fused Q/K/V norm,含 fallback 条件
python/sglang/srt/models/gemma4_mm.py: per-expert FP8 checkpoint key 的正则映射与加载
test/registered/models/test_gemma4_fp8_per_expert_loading.py: 4 卡端到端回归(新文件)
```

## 性能声明

- Microbench(单 H100,bf16,Q=8 头、KV=2 头、head_dim=128):M=1 44.4→19.8 µs(2.25x)、M=256 59.7→20.1 µs(2.96x)、M=4096 58.4→18.6 µs(3.14x)、M=8192 59.3→32.3 µs(1.83x)、M=16384 108.0→75.0 µs(1.44x);M=1/8/128 与 unfused bit-exact(max abs diff = 0)。
- E2E(4 x H100 TP=4,RedHatAI FP8,batch=1、in=2048、out=64):prefill 2048 tok 47.0→46.0 ms(-2.1%);decode median 5.47→5.45 ms/tok(-0.4%);total 0.393→0.390 s(-0.8%)。PR 说明 decode 收益被 CUDA Graph 摊薄 launch 开销,无 graph 的 prefill 才体现完整约 2% 收益;测试 M 范围内不劣化。
- 正确性:per-expert FP8 loader 修复前 GSM8K-20 = 0.00,修复后约 0.95(测试门限 >= 0.80)。MMLU 500 例:BF16 TP=4 为 0.878,FP8-Dynamic overall 0.885。

## 验证与回归风险

新增 CI 测试 `test_gemma4_fp8_per_expert_loading.py`(suite `stage-c-test-4-gpu-h100`,预估 120 s)以 TP=4 启动 FP8 模型并断言 health、非 `<pad>` 输出、GSM8K-20 >= 0.80,三项在修复前均失败。microbench 断言 bit-exact。风险面:fused 路径仅在标准 Gemma4 norm 配置下启用,其余走原路径;loader 正则只匹配 per-expert FP8 布局,fused 参数缺失时降级为 warning;PR 自述"Existing Gemma4 tests still pass (need full CI run)"在合并时未勾选,完整 CI 结果待确认。

## 关联

同区域:`gemma4_fused_ops.py` 中既有的 `gemma_dual_rmsnorm_residual_scalar` 等融合算子(本 PR 同文件扩展);pack 内未给出其他具体 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
