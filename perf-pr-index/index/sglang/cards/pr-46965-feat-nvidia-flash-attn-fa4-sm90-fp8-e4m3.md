# PR #46965: [Feat] [NVIDIA] flash-attn: FA4 SM90 FP8 e4m3 KV cache for head_dim 256 and 512 (Gemma-4)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @cognaiger9 |
| 合并 | 2026-11-11 |
| 分类 | Quantization(tier: major) |
| Label | run-ci, jit-kernel |
| Diff | 素材未提供 diffstat(超长 diff 被截断:保留 flash_fwd.py/flash_fwd_sm90.py,另 7 个文件略去) |
| 链接 | https://github.com/sgl-project/sglang/pull/46965 |

## 一句话结论

FA4 SM90 前向支持 FP8 e4m3 KV cache(head_dim 256/512),Gemma-4 可整体开 FP8 KV,容量翻倍、吞吐 +10-25%。

## 动机

#44399 之后,Gemma-4 的 10 个 full-attention 层(head_dim 512)经 `fa3` 后端跑在 FA4 上,但 FA4 在 SM90 没有 FP8 KV 路径,`--kv-cache-dtype fp8_e4m3` 会在 init 被拒;FA3 却已在 50 个 sliding-window 层(head_dim 256)读 FP8 KV。而且 batch 一旦带 image tokens(含 VLM 启动 warm-up),sliding-window 层也会落到 FA4,所以 FA4 必须在两个 head_dim 都支持 FP8 KV。

## 优化点

核心是把反量化搬进 kernel:K/V tile 以 fp8 载入新增的 smem staging slot,由 MMA warpgroups 在 shared memory cast 成 fp16 再进 MMA,K/V descale 折进 softmax scale 与 output scale。hdim 512 时两个 warpgroup 沿 N 各 cast 一半 V;hdim 256 时全体 MMA 线程过一个 barrier 后 cast 整个 V;FP8 KV 下 pipeline 退化为 1 stage,单个 fp8 staging slot 按 K/V 交替喂 consumer。接口上 bf16/fp16 Q + fp8 e4m3 K/V 在 SM90 选中该路径,Q/K/V dtype 进 compile key;不支持 e5m2、其他 head_dim、non-paged KV、`qv`、`rel_bias`、block sparsity 等;bf16 Q 在 kernel 内 narrow 成 fp16(|q|>65504 溢出,QK-norm 使其远离)。后端上 `fa3` 升层到 FA4 时接受 fp8 cache:FA4 层 Q 不量化并接收 descales,FA3 层照旧量化 Q;`fa4` 后端对 MHA 非 256/512 head_dim 的 fp8 cache init 即拒,MLA 保持原路径。native FP8 MMA(WGMMA e4m3)留作后续 PR。

## 关键改动

```text
python/sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm90.py: fp8_kv_dequant 路径——fp8 staging slot、load_fp8_kv producer、MMA 内 smem cast 与 descale 折算
python/sglang/kernels/ops/attention/flash_attn/cute/flash_fwd.py: Q/O dtype 与 MMA dtype 解耦(bf16 Q/O、fp16 计算)
python/sglang/kernels/ops/attention/flash_attn/cute/interface.py: SM90 bf16/fp16 Q + fp8 e4m3 K/V 选中该路径,Q/K/V dtype 进 compile key
python/sglang/srt/layers/attention/flashattention_backend.py: fa3 接受 fp8_e4m3 cache 并升层 FA4;fa4 init 校验
```

```python
self.fp8_kv_dequant = fp8_kv_dequant
if fp8_kv_dequant:
    assert self.dtype == Float16, "FP8-KV dequant computes in fp16"
    assert self.tile_hdim == self.tile_hdimv and self.tile_hdim in (256, 512)
    # The cast writes only slot 0 of sK/sV; one staging slot feeds both.
    assert self.num_stages == 1
    assert mma_pv_is_rs and not has_bias and not self.Q_in_regs
```

## 性能声明

Gemma-4-31B-it FP8-block,1×H200,TP1,fa3 后端,`--mem-fraction-static 0.9`,memory-limited 场景(in-flight KV 超出 BF16 池、装进 FP8 池),`sglang.benchmark.serving`,随机数据集固定输入/输出长度。输出吞吐:1k 输入 c64/c128 +20%/+25%,4k +18%/+10%,8k +4%/+6%,16k +6%。median TTFT −7% 至 −61%(1k/c128:14,600→5,699 ms)。mean decode batch 至多 2×(4k/c64:43.6→64.0),retraction 1k/c64 16→0、4k/c64 11→0。KV 容量翻倍:full-attention 池 405,504→811,072 tokens,sliding-window 池 81,088→162,176 tokens。

## 验证与回归风险

test_flash_attention_4.py 新增 FP8 KV 子测试(H200 36 个:hdim 256/512、decode/MTP verify/chunked prefill/8k prefill、page size 1/64、split-KV、image mask、CUDA graph replay,对照 fp32 dequant 参考并带 per-(request, KV head) descales);test_flashattention_paged_mha.py 验证 FA4 层 Q 不量化、FA3 层量化。精度:GSM8K 0.9729→0.9719(Δ −0.0009,95% CI −0.0036..+0.0017);KL drift 在 ~4k/16k/48k context mean 9.5e-3/6.9e-3/7.4e-3(噪声底 ~2.3e-4),不随 context 增长。风险:bf16 Q narrow fp16 的溢出条件;e5m2/non-paged KV/block sparsity 等场景不支持;Q/K/V dtype 进 compile key 会影响缓存命中。

## 关联

#44399(前序:Gemma-4 full-attention 层升级到 FA4);同区域后续方向为 SM90 native FP8 MMA 路径(PR 未给编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
