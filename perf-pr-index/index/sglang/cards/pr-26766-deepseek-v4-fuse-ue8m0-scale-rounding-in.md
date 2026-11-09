# PR #26766: [DeepSeek-V4] Fuse UE8M0 scale rounding into FP8 group quantization

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JoeLee314 |
| 合并 | 2026-06-18 |
| 分类 | Speculative Decoding(tier: major) |
| Label | high priority, quant, deepseek, run-ci, jit-kernel, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 未提供 +add/-del 行数与文件数;diff 内容可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/26766 |

## 一句话结论

DeepSeek-V4 `wo_a` FP8 路径把 UE8M0 scale 舍入融合进 per-token group quant kernel,省一次独立 `ceil_to_ue8m0`,输出吞吐 +7.8%。

## 动机

DeepSeek-V4 的 `wo_a` FP8 路径原先先做激活量化,再用单独的 kernel 把生成的 FP8 scale 舍入成 UE8M0/2 的幂:`sglang_per_token_group_quant_fp8 -> deep_gemm.ceil_to_ue8m0 -> deep_gemm.fp8_einsum`,多一次 kernel launch 与 scale 缓冲的读写往返。

## 优化点

让 per-token-group 量化 kernel 直接输出舍入后的 2 的幂 scale:`sglang_per_token_group_quant_fp8(scale_ue8m0=True) -> deep_gemm.fp8_einsum`。具体:

- JIT kernel `per_token_group_quant_8bit_v2.cuh` 的 scale 类型从 `SCALE_UE8M0` 改为按 `SCALE_UE8M0 && IS_COLUMN_MAJOR` 决定 packed uint32/uint8 还是 float32,row-major UE8M0 路径保留 float scale 缓冲(与 `deep_gemm.ceil_to_ue8m0`/`fp8_einsum` 的期望一致)。
- UE8M0 时统一走 `calculate_fp8_scales<SCALE_UE8M0=true>`:先算出舍入后的 2 的幂 scale,再用它量化——对齐官方 DeepSeek-V4 `kernel.py` 的 `act_quant(scale_fmt="ue8m0")`。
- `fp8_kernel.py` 为 row-major UE8M0 新增 float32 scale 输出布局;`deepseek_v4.py` 的 `wo_a` 调用删除 `deep_gemm.ceil_to_ue8m0(o_s)`,直接传 `scale_ue8m0=True`。
- Blackwell 精度修复(更新段落):初版 kernel 在 `SCALE_UE8M0 && !IS_COLUMN_MAJOR` 分支用精确 FP32 scale 量化、只对输出 scale 舍入,量化与反量化 scale 不匹配导致误差放大约 14x,EAGLE 投机解码 accept_length 在 B300/B200 从 2.76 降到 2.69;修复(感谢 @yhyang201)删除该特殊分支,所有 UE8M0 路径先舍入后量化。

## 关键改动

```text
python/sglang/jit_kernel/csrc/gemm/per_token_group_quant_8bit_v2.cuh: UE8M0 舍入融合进量化 kernel,先算 2 的幂 scale 再量化
python/sglang/srt/layers/quantization/fp8_kernel.py: row-major UE8M0 的 float32 scale 输出布局
python/sglang/srt/models/deepseek_v4.py: wo_a 路径去掉 ceil_to_ue8m0,改 scale_ue8m0=True
test/registered/jit/test_per_token_group_quant_8bit_v2.py: 新增 row-major UE8M0 精度测试
test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py: EAGLE accept_length 阈值 2.6→2.8
```

## 性能声明

测试条件:DeepSeek-V4-Flash,TP=4,32K input / 1K output,128 requests,max concurrency 4(PR 未注明硬件型号)。结果:Output token throughput 增加 7.8%(PR 附吞吐对比图)。无其他端到端数字。

精度:GSM8K 得分 0.928;Blackwell 精度修复后两个 B200 e2e 测试的 EAGLE accept_length 阈值从 2.6 提高到 2.8,反映 accept rate 恢复(修复前该指标曾从 2.76 降到 2.69)。

## 验证与回归风险

新增测试 `test_sglang_per_token_group_quant_fp8_row_major_ue8m0`(bfloat16/float16 × 1/33/128 tokens × 128/512/4096/7168 hidden):断言 scale 严格为 2 的幂(log2 取整相等)、dequant 平均相对误差 < 0.05,并要求 row-major UE8M0 的 fp8 codes 与 `ceil_to_ue8m0` 参考一致。风险:kernel 对 scale 一致性敏感——初版曾因量化/反量化 scale 不匹配拉低 EAGLE accept rate,说明同类改动需要 e2e accept-length 验证;row-major + UE8M0 组合受断言约束(column_major_scales 必须配 scale_tma_aligned);column-major UE8M0 路径继续用 packed uint32/uint8 格式,行为不变。

## 关联

无(pack 未给出同区域其他 PR 编号;性能影响最终体现在 EAGLE 投机解码的 accept_length 上)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
