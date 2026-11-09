# PR #61855: [ROCm][Perf] Triton port of the fused GDN MTP decode kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @iseeyuan |
| 合并 | 2026-10-23 |
| 分类 | Speculative Decoding |
| Label | rocm, ready, qwen |
| Diff | diffstat 缺失(素材未记录 +/- 行数与文件数);diff 可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/61855 |

## 一句话结论

把 CUDA-only 的 fused GDN MTP decode kernel 移植为 Triton 并在 ROCm 启用:Qwen3.5 系 spec decode 每步中位 ITL 最多 -12%,单请求 decode -13%。

## 动机

flat qkvz layout 的 GDN 层(Qwen3.5 家族)在 ROCm 上投机解码一直回退 unfused 路径,因为 CUDA kernel `fused_gdn_decode_post_conv_mtp`(#51674)只为 CUDA 13 构建。于是每层 verify step 要跑约 8 个小 kernel:`a`/`b` 的 contiguous 拷贝、三次拷贝加一个 `cat` 解包 q/k/v、zero-fill、输出拷贝与 gated RMSNorm,launch 开销淹没 recurrent kernel。

## 优化点

新增 `fused_gdn_decode_post_conv_mtp_triton`,签名与 CUDA op 相同,把 post-conv 工作压到两个 launch。update kernel 每 program 处理一个 (request, value head, V tile):就地从 projection 读 q/k/v 与 `a`/`b`,按 `fused_sigmoid_gating_delta_rule_update` 的同款状态语义(从最后 accepted token 的状态出发、逐 token 存状态、slot ≤0 无状态)跑 gated delta-rule update,每个 tile 写自己的输出行与 sum of squares。epilogue kernel 每 (request, value head, token) 一个 program,聚合 partial sum 并做 gated RMSNorm,写 out_proj 输入。Tile 规则:(request, value head) 对数少于 CU 数用 32 行、4×CU 内用 64、否则每头一 tile;扫描显示该规则处处在最优 8% 以内,固定 32 在 MI300X 16–32 requests 时最多慢 36%。接入上,ROCm 经 `fused_mtp_decode_impl()` 选 Triton port,eligibility 与 CUDA op 相同(flat layout、K=V=128、head ratio 1/2/3/4/8、BF16、每 verify ≤8 token);warmup 预编译所有可达 tile size;可用现有 `VLLM_GDN_DECODE_KERNEL=triton` 关闭。

## 关键改动

```text
vllm/third_party/flash_linear_attention/ops/fused_gdn_decode_post_conv_mtp.py: 新文件,update + gated RMSNorm 两 kernel 与 tile 选择
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py: supports_fused_mtp_decode / fused_mtp_decode_impl 平台路由
vllm/model_executor/warmup/qwen_triton_warmup.py: MTP decode kernel 按 tile size warmup
tests/kernels/fla/test_fused_gdn_post_conv.py: 按 impl=cuda/triton 参数化
```

```python
# Tile V only while the GPU needs more programs: 32 rows per program while
# there are fewer (request, value head) pairs than CUs, 64 up to 4x the CUs,
# else one tile per value head (tuned on MI300X and MI350X).
heads, num_cus = num_requests * HV, num_compute_units(out.device.index)
BV = 32 if heads < num_cus else 64 if heads <= 4 * num_cus else V
```

## 性能声明

- Qwen3.5-35B-A3B,MTP k=3,1x MI350X,单请求:decode 2.39 → 2.08 ms/token(-13%),E2E 544 → 488 ms(-10%)。
- Per decode step(torch profiler):copy/fill/cat 1.56 → 0.31 ms,kernel launches 1008 → 768,总 GPU kernel time 7.74 → 6.71 ms。
- `vllm bench serve`(random 2K/256,`--ignore-eos`):median ITL -12% / -6% / -6%(并发 1/8/32),output tok/s 314.0→359.2 / 971.7→1126.8 / 2150.1→2354.9。
- Long context(64K in/1K out):median ITL -12% / -6% / -3%,prefill 不变。
- Kernel per call(MI300X,gfx942):比 unfused 快 49–60%(1 请求)、25–44%(32 请求);MI350X FP32 单请求 30.0 → 11.5 µs。
- 精度:GSM8K base 0.854–0.861 vs PR 0.848–0.862(阈值 0.84);GPQA Diamond 0.821 vs 0.826;NIAH 4K–64K 双方 100%;MTP acceptance 2.85–2.87 不变。

## 验证与回归风险

kernel test 按 impl=cuda/triton 参数化,Triton port 在任意 GPU(含 NVIDIA CI)上验证数值:MI350X 与 MI300X 各 22 passed / 22 skipped(CUDA op 用例 skip);状态按 tile 尺寸 bit-identical;另有 warmup 覆盖可达 tile size 的 ROCm-only 测试。风险:E2E 仅在 Qwen3.5-35B-A3B 上跑过,表中其余模型(Qwen3.5 各尺寸、Qwen3.8-27B、Qwen3.8-Flash-Next)只有 kernel test 覆盖;MI350X BF16 state 32 请求时比 kernel 初版慢 8–9%(仍远快于 unfused);依赖 aiter main——aiter ≤v0.1.25 会把 verify batch 送进未拆分的 gfx950 attention kernel 而掩盖收益;Qwen3-Next(interleaved layout)不适用,CUDA 路径不变。

## 关联

前序:#51674(CUDA fused GDN MTP decode kernel)。相关:#61675(shape-specialized ROCm kernel,连 conv 一起融合,opt-in,仅 16/48 头 TP=1)、#61434(unfused kernel 内延迟状态写入)、#59366(RecoverSSM CUDA replay op)、#60833/#61325(Qwen3.8 mono decode MI355X)。索引同区域另有 pr-65874(fused GDN MTP 支持 DSpark)、pr-52816(dflash2 local convolution cache)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
