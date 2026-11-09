# PR #50578: [ROCm][MLA] Use asm decode for non-divisor small head counts

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @vanshbhatia-amd |
| 合并 | 2026-08-06 |
| 分类 | Attention / Kernel (tier: major) |
| Label | rocm, ready, quantization |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files),实际改动为 backend helper/env + 两个测试文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/50578 |

## 一句话结论

ROCM_AITER_MLA 上把非 16 约数的少 query head 数(如 Kimi-K3 TP8 的 12 头/rank)tile-pad 到 16 走 asm persistent decode,长上下文 decode 最快 2.3x。

## 动机

`ROCM_AITER_MLA` 后端中,每 rank 少于 16 个 query head 的 MLA decode 此前跑 Gluon decode kernel:它只在 head 维度并行,单个 workgroup 要顺序扫完整个 KV cache,per-token 延迟随上下文长度线性增长。更快的 asm persistent decode 要求 `num_heads >= 16`;小于 16 时虽可 pad,但 `get_mla_padded_q`/`get_mla_unpadded_o` 只支持 16 的约数(`repeat_interleave`),非约数(如 Kimi-K3:96 头 / kv_lora_rank 512,TP8 下 12 头/rank、TP16 下 6 头/rank)只能留在 Gluon,长上下文 decode 严重劣化。

## 优化点

机制:非约数 head 数的 padding 改为"tile + 截取"——`get_mla_padded_q` 用 `q.repeat(1, ceil(16/num_heads), 1)[:, :16, :].contiguous()`,把 query head 循环平铺再截到恰好 16 个(单个 append 在 `num_heads < 16-num_heads` 时会 pad 不足;截取 tile 会产生非连续视图,而 asm persistent decode 按紧凑的 `[tokens, 16, head_dim]` 读 q,故 materialize contiguous 拷贝)。padding 的精确性论证:MLA attention 对共享 KV 逐 head 独立,追加的重复 head 无法影响真实 head,`get_mla_unpadded_o` 直接切回前 `num_heads` 个。kernel 选择上,新增 `_gluon_mla_decode_supported()`(Gluon kernel 仅有 gfx950/CDNA4 构建,其 tiling 需约 160 KiB LDS 超过 CDNA3 的 64 KiB,gfx942 无构建)与新环境变量 `VLLM_ROCM_AITER_MLA_ASM_PADDING`(auto/gluon/asm,默认 auto):auto 下 16 的约数在 gfx950 保留 Gluon,非约数与全部 gfx942 计数走 asm;`max_qo_len > 1` 的 small-head verify 路径同样加 arch/env 门控。

## 关键改动

```text
vllm/v1/attention/backends/mla/rocm_aiter_mla.py: get_mla_padded_q/unpadded_o 支持非约数(tile+slice),use_gluon_decode 按 arch+env 选择,_gluon_mla_decode_supported 探测
vllm/envs.py: 新增 VLLM_ROCM_AITER_MLA_ASM_PADDING(auto/gluon/asm,默认 auto)
tests/kernels/attention/test_rocm_aiter_mla_head_padding.py: 新增,pad/unpad 往返、12/6 头 tile 语义、kernel 选择矩阵、12 头 asm decode 对参考实现
tests/kernels/attention/test_rocm_aiter_mla_causal_verify_mask.py: verify flatten 测试改为 spy + arch probe patch
```

关键 padding 代码:

```python
reps = -(-m // num_heads)  # ceil(m / num_heads)
return q.repeat(1, reps, 1)[:, :m, :].contiguous()
```

## 性能声明

测试平台:Kimi-K3(mxfp4),8x MI355X(gfx950),TP8,`VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_USE_AITER_MLA=1` + `--moe-backend aiter`;decode 延迟 vs 上下文长度(greedy,单流,`vllm bench serve`,max-concurrency 1),Mean TPOT(ms/token):1K 输入/1K 输出 22.9 → 22.8(1.00x);1K/32K 33.4 → 23.3(1.43x);32K/1K 43.8 → 23.6(1.86x);32K/32K 54.1 → 23.6(2.29x,正文表述 up to 2.3x faster)。decode 被拉平到约 23.5 ms/步且与上下文长度无关,短上下文无回退;greedy 输出不变。rank-0 trace 确认 asm 路径激活(decode reduce kernel 为 `kn_mla_reduce_v1_...Li512ELi16E...`,16 个 padded head),`_mla_gluon` 不再出现。

## 验证与回归风险

新增 `test_rocm_aiter_mla_head_padding.py`:全部 1-15 head 数的 pad→unpad 精确往返、12 头(TP8)与 6 头(TP16)的 tile 语义断言(旧 append-only pad 在 6 头时产生 12 个 head 会打崩 asm kernel,由测试钉住)、`use_gluon_decode` 在 auto/gluon/asm 三种 env 与 gfx950/gfx942 组合下的选择矩阵、12 头 asm decode 与 `scaled_dot_product_attention` 参考在 atol/rtol=1e-2 内一致。verify mask 测试改为 spy 替代 Gluon kernel 并强制 arch probe,使 metadata 回归在所有 ROCm AITER CI runner 上可测。风险面:tile-pad 会为非约数计数引入一次 q 的 contiguous 拷贝(与 Gluon 全量扫 KV 相比可忽略);divisor 计数与 `max_qo_len>1` verify 的默认路径不变;gfx942 上强制 "gluon" 会告警并回退 asm。

## 关联

同区域:PR #39616(ROCm AITER MLA attention 启用,本 PR 的直接前置)、pr-59478(AITER FlyDSL INT4 quick reduce)、pr-61837(Kimi-K3 Step5 preview on ROCm);Kimi-K3 KDA 优化见 pr-53396、pr-53614。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
