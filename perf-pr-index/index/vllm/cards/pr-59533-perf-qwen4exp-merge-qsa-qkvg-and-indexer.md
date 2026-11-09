# PR #59533: [Perf][Qwen4Exp] Merge QSA QKVG and indexer QK projections

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gau-nernst |
| 合并 | 2026-10-05 |
| 分类 | Attention / Kernel (tier: major) |
| Label | ready, qwen |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾无 "file(s) dropped from diff" 说明,diff 完整保留 6 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/59533 |

## 一句话结论

把 Qwen4Exp QSA QKVG 与 indexer QK 投影合并成单个 GEMM,4×GB300 TEP4+MTP3 吞吐最高 +5.1%(并发 64)。

## 动机

Qwen4Exp QSA 的 QKVG 投影与被复制的 indexer Q/K 投影此前是两个独立 GEMM,且各有自己的 low-latency dispatch 计划。PR 把二者并成一次 GEMM,减少 launch 与权重读取,并借机清理过时的 merged-HC / separate-QSA 计划。

## 优化点

Projection fusion:把 QKVG 与 replicated indexer Q/K 投影合并成 **一个 GEMM**(merged weight `[4224,2560]` = `[3584,2560]` + `[640,2560]`),LoRA 仍保留分离投影与标准 linear dispatch。GEMM dispatch 为 merged GEMM 配 LL-GEMM 计划:SM103(GB300)TP4、SM100(B200)TP4、SM90(H200)TP4(M=1/2/4)、SM121(GB10)TP1,并 prune 过时的 merged-HC 与 separate-QSA plans。Prepare kernel 支持 strided merged outputs、把 row-offset 算术放宽到 i64(正文称 GB300 上无显著性能差异,重复测量 ≤1%)。`qsa.py` 去掉单独 `index_qk_proj` 调用、改由合并结果切分;`low_latency_gemm.py` 按 capability 维护 merged plans;`ops/qsa_prepare.py` 处理 strided/i64。

## 关键改动

```text
vllm/models/qwen4_exp/nvidia/qsa.py: 合并 QKVG 与 indexer QK 投影,去掉单独 index_qk_proj 调用
vllm/models/qwen4_exp/nvidia/low_latency_gemm.py: 按 SM103/SM100/SM90/SM121 capability 提供 merged GEMM 的 LL-GEMM 计划,prune 过时计划
vllm/models/qwen4_exp/nvidia/ops/qsa_prepare.py: prepare kernel 支持 strided merged outputs 与 i64 row-offset
vllm/models/qwen4_exp/nvidia/model.py: 模型侧接入合并投影
tests/kernels/test_bf16_skinny_gemm.py, tests/models/qwen4_exp/test_qsa_prepare.py: 校验 plan 路由与 prepare 正确性
```

关键代码(diff 中的 capability 计划路由断言调整):

```python
        ((10, 3), qwen4_exp_gemm.QWEN4_EXP_SM103_GEMM_PLANS),
        ((10, 0), qwen4_exp_gemm.QWEN4_EXP_SM100_GEMM_PLANS),
        ((9, 0), qwen4_exp_gemm.QWEN4_EXP_SM90_GEMM_PLANS),
```

## 性能声明

Microbenchmark:一块 GB300、BF16、TP=4,`X=[M,2560]`、separate weights `[3584,2560]` + `[640,2560]`、merged weight `[4224,2560]`;两臂均用生产 `qwen4_exp_low_latency_gemm`(LL-GEMM at M=1/4、M>=16 用 cuBLAS via standard linear),FlashInfer CUPTI、CUDA graphs、cold L2,3 轮 × 50 iterations 中位、交替臂序;对 FP32 累加校验(`rtol=0.02`、`atol=0.01`);Torch `2.13.0+cu130`、baseline `9e6550bd05`。Separate → Merged(µs)/Speedup:M=1 9.857 → 6.624(1.49x)、4 11.024 → 7.616(1.45x)、16 13.472 → 8.032(1.68x)、64 13.664 → 8.224(1.66x)、256 13.584 → 9.216(1.47x)、1024 21.920 → 15.584(1.41x)、4096 51.809 → 51.648(1.00x)、16384 191.424 → 185.233(1.03x);run 中位间最大差 5.5%,M=4096 基本持平。End-to-end:TEP4 + MTP3、4× GB300、Qwen4Exp BF16,baseline `aee8fe202a` → feature `401b1249c0`,SPEED-Bench `throughput_8k`(输入上限 8192、输出 1024 tokens、prefix caching off、5 warmups、每并发档 10 prompts)。Output tokens/s(并发 1/4/16/64):317.0 → 315.7(−0.4%)、843.0 → 855.6(+1.5%)、1883.4 → 1934.0(+2.7%)、3566.3 → 3747.6(+5.1%);每臂 850 measured requests、0 failed;并发 1 基本持平、并发 64 最高 +5.1%,并发 64 两臂 MTP acceptance 均 36.56%。End-to-end accuracy(feature only、TEP4+MTP3、prefix caching on):GSM8K 96.55% / 96.63%(flexible/strict,2638 responses)、MMMU-Pro 76.65% / 76.59%(1730 responses),0 request errors,与历史约 96.7% / 76.94% 接近,正文注明未做同期 baseline accuracy run。

## 验证与回归风险

本地正确性:16 个 prepare 测试在 rebase 前通过;rebase 后 focused LL-GEMM suite 7 passed、102 architecture-specific skips(`tests/models/qwen4_exp/test_qsa_prepare.py`、`tests/kernels/test_bf16_skinny_gemm.py -k qwen4_exp`);另覆盖 BF16/FP8 投影与 checkpoint 加载 TP=1/2/4/8、LoRA fallback 构造、FP8 per-tensor/block scales、跨 `2**31` 元素的 sparse inputs(验证 widened offsets)、36 项 pruned shape 的 standard-linear dispatch、capability-map 选择与 unsupported-capability 回退、M=1/2/4/16 的 CUDA 投影检查;贡献的 H200 merged-GEMM plans 通过本地 table/routing 检查,GPU 正确性测试在本机跳过。静态检查:pre-commit、mypy-3.12、`git diff --check`,rebase 到 main `0c16eee3f1` 后再次通过。风险面:改动 QSA 投影与 LL-GEMM dispatch 表(影响按 capability 选计划、LoRA 回退、FP8 scale 处理);正文注明一次 sweep、无重复运行 variability 估计,且未做同期 baseline accuracy run。

## 关联

PR 正文 Review notes 明确:#57097 融合后续 prepare operations、#59214 调优现有 SM100 shapes,二者与本 PR 的 projection fusion 不同;重复检查未发现其他 projection-fusion PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
