# PR #64467: [Perf][MoE] Use 32-row tiles in prepare_megamoe_inputs from 1024 tokens

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Juntian777 |
| 合并 | 2026-10-30 |
| 分类 | Speculative Decoding(classify.py 给出;实际为 DeepGEMM MegaMoE 输入准备内核的 tile 调优) |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +N/-N 与文件数;diff 内容可用,覆盖 2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/64467 |

## 一句话结论

DeepGEMM MegaMoE 的 MXFP8 量化/重排内核从 1024 token 起改用 32 行 tile,内核提速 1.3–1.6x 且逐位一致。

## 动机

`prepare_megamoe_inputs` 在每个 DeepGEMM MegaMoE 层前做 MXFP8 量化与 top-k 重排。此前策略是 64 token 以上用 8 行 tile、更小 batch 不分 tile——8 行 tile 是为小 batch 调的,在 prefill 级别的 1024+ token 下线程块过碎、每块负载过低,成为可测量的 GPU 时间开销(40 层合计超过 1.5ms/forward)。

## 优化点

一行 tile 策略改动:`block_m = 32 if num_tokens >= 1024 else 8 if num_tokens >= 64 else 1`——在保留原有两个档位(64 token 以下不分 tile、64–1023 token 用 8 行)的基础上,1024 token 及以上切换到 32 行 tile。Triton 网格按 `cdiv(num_tokens, block_m) x cdiv(hidden_size, block_k)` 缩减为 1/4 的块数,`block_topk` 与 padding/scale 逻辑不变。所有输出(x、scales、top-k、shared scales)在所有测试尺寸上与 main 逐位一致,属于纯 launch 形状优化。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/prepare_megamoe.py: block_m 按 token 数三档选择(1/8/32)
tests/kernels/moe/test_deep_gemm_mega_moe.py: bitwise 测试新增 1025-token 尾部用例
```

```python
# On GB200, 8-row tiles win from 64 tokens and 32-row tiles from 1024 (1.3-1.6x
# over 8 rows); smaller batches stay untiled.
block_m = 32 if num_tokens >= 1024 else 8 if num_tokens >= 64 else 1
```

## 性能声明

(DeepSeek-V4.1-Flash 形状:hidden 5120、top-6、fused shared expert `shared_block_m=128`;1x GB200,CUPTI + CUDA graph,3 次取中位;输出全部与 main bit-identical)

| tokens | main 8 行 (us) | 32 行 (us) | Speedup |
| ---: | ---: | ---: | ---: |
| 512 | 3.1 | 3.0 | (8 行,不变) |
| 1024 | 4.7 | 3.7 | 1.27x |
| 2048 | 7.4 | 5.1 | 1.45x |
| 4096 | 13.1 | 8.0 | 1.64x |
| 8192 | 29.8 | 22.5 | 1.32x |
| 16384 | 57.4 | 42.9 | 1.34x |
| 32768 | 113.6 | 84.3 | 1.35x |

- E2E(`vllm bench forward`,DS-V4.1-Flash dummy weights,DEP2 单 GB200 节点,FlashMLA MegaAttn + MXFP4 indexer,rank 0 kernel profile):`_prepare_megamoe_inputs`(x40 层)GPU 时间 16K 新 token 1517→948us;128K cached + 8K 新 761→492us;128K cached + 16K 新 1475→952us。即 8K–16K 新 token 每 forward 省 0.27–0.57ms(约 0.3%)。
- 精度(与 #64466 一并应用,均 bit-identical;DSpark K=5,GSM8K 5-shot greedy,并发 32):strict-match 0.9287、acceptance length 3.447,对照改动前 0.9265 / 3.451。

## 验证与回归风险

`pytest tests/kernels/moe/test_deep_gemm_mega_moe.py -k prepare_megamoe` 通过;`test_prepare_megamoe_inputs_is_bitwise_exact` 增加 1025-token 参数,覆盖 32 行 tile 的尾部(非整除)情形。风险低:改动是单行 tile 阈值,输出 bit 级不变;仅在 SM100/GB200 MegaMoE 路径生效,<1024 token 的 decode/small-batch 行为完全不变。

## 关联

#64466(同作者同日、与本 PR 一起做精度验证的姊妹改动);#57977(动 shared-scale staging 但不动 tile 大小,#64467 素材中引用以说明无重叠)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
