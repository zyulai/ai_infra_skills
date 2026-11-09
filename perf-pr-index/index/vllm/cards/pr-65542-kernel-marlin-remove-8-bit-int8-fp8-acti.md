# PR #65542: [Kernel] Marlin: remove 8-bit (INT8/FP8) activation support, rely on Humming

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-11-04 |
| 分类 | Quantization(tier: major) |
| Label | documentation, performance, ci/build, nvidia, quantization |
| Diff | diffstat 缺失(素材未提供 +/− 统计;PR 正文称净 −2039 行);diff 素材截断(48 个文件未展示) |
| 链接 | https://github.com/vllm-project/vllm/pull/65542 |

## 一句话结论

Marlin 砍掉 W4A8(INT8/FP8 activation)路径净 -2039 行,8-bit activation 统一交给 Humming kernel。

## 动机

Marlin 的 8-bit activation 路径在 main 上已经很窄:linear 只有 GPTQ 对称 INT4 能走(AWQ+int8 加载即报错),FP8-weight+FP8-activation 只在 Python 层接线从未实例化;模板里 is_a_8bit 分支、sm89 专用 kernel 与双份 build target 维护成本高。#65560 规划把 8-bit activation 收敛到 Humming,本 PR 是其中一步。

## 优化点

删除 dense 与 MoE Marlin 的 W4A8-INT8(GPTQ/AWQ INT4)与 W4A8-FP8(GPTQ/AWQ INT4、MXFP4)路径:sm89 FP8-activation kernel 及其 SM89/SM12x build targets、INT8-activation 实例化、matmul_a8 及 a_scales 参数、marlin_int4_fp8_preprocess op、repack ops 的 is_a_8bit。模板改动是 is_a_8bit==false 的常量折叠,16-bit kernel 编译结果不变;mma helper 去掉 k=32 路径。Marlin 定位收窄为 16-bit-activation kernel:MarlinLinearKernel.can_implement 拒绝 1 字节 activation 后选择回落 HummingLinearKernel,ConchLinearKernel 也补上 16-bit 检查。VLLM_MARLIN_INPUT_DTYPE 移除,设置即启动报错并给出迁移提示(--quantization humming + VLLM_HUMMING_INPUT_QUANT_CONFIG)。FP8-weight(W8A16/MXFP8)与所有 16-bit activation 格式不受影响。

## 关键改动

```text
csrc/moe/marlin_moe_wna16/{kernel.h,marlin_template.h,ops.cu} 及 dense 对应文件: 删除 is_a_8bit 分支与 8-bit 实例化(48 个文件在素材 diff 中被截断)
vllm/envs(removed_environment_variables): VLLM_MARLIN_INPUT_DTYPE 设为硬错误
kernel 选择: MarlinLinearKernel/ConchLinearKernel 拒绝非 16-bit activation,回落 Humming
docs/features/quantization/README.md: Humming W4A8 迁移文档
```

## 性能声明

GSM8K,Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4,DGX Spark(GB10,SM121)、CUDA 13.0 源码构建。100 题 5-shot(main vs 本 PR):W4A16 默认 Marlin 0.62 / 3348 tok/s vs 0.60/0.62(两跑)/ 3364 tok/s;W4A8-INT8 0.68 / 3413 tok/s vs 0.63 / 4008 tok/s(Humming);W4A8-FP8 0.69 / 3554 tok/s vs 0.62 / 3842 tok/s(Humming)。全集 1319 题 accuracy:W4A16 0.622→0.622;W4A8-INT8 0.625→0.644;W4A8-FP8 0.591→0.603。作者称 100 题差距在噪声内。

## 验证与回归风险

kernel 测试:test_marlin_gemm.py 254 passed;test_moe.py -k marlin 189 passed/6 skipped;test_marlin_tile_padding.py 98 passed;test_reload.py -k marlin 6;test_marlin_vs_trtllm_mxint4.py 1(计数下降因删除 8-bit 用例);pre-commit 含 mypy 通过。风险:VLLM_MARLIN_INPUT_DTYPE 无弃用期直接移除(设置即 ValueError,防静默回落 W4A16);Humming W4A8 MoE 端到端未验证(Marlin MoE W4A8 此前只能经该 env var 到达);压缩张量 W4A8-FP8 在 SM89/SM12x 上从 Marlin 改选 HummingLinearKernel。

## 关联

属 #65560 的一部分;独立于 #65540(Marlin PTX-only build)。取代 8 个扩展 Marlin W4A8 的 open PR:#58326、#48926、#38066、#50096、#59895、#61405、#49039、#57091。同 kernel 区域卡:#45703(Marlin MoE thread-tile padding)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
