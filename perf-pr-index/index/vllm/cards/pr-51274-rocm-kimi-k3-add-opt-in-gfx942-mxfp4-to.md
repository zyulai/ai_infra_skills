# PR #51274: [ROCm][Kimi-K3] Add opt-in gfx942 MXFP4-to-int4 conversion

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @maeehart |
| 合并 | 2026-10-03 |
| 分类 | MoE(tier: major) |
| Label | rocm, ready, quantization, kimi, k3 |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数) |
| 链接 | https://github.com/vllm-project/vllm/pull/51274 |

## 一句话结论

在 gfx942 上为 Kimi-K3 MXFP4 专家提供 opt-in 的 load-time int4 重量化路径,绕开 gfx942 缺失 scaled FP4 转换指令导致 MXFP4 无法原生执行的问题。

## 动机

gfx942 硬件不支持 native MXFP4 路径所依赖的 scaled FP4 转换指令,MXFP4 专家权重在这代 MI300 上要么走模拟要么不可用。与其在推理时反复付出模拟开销,不如在模型加载阶段一次性把 MXFP4 专家重量化为 groupwise int4,之后走 AITER 的 BF16 x packed-int4(FlyDSL A16W4)替换 kernel。转换是有损的,因此设计为显式 opt-in(`--quantization-config.moe.weight int4_per_group_32`),默认 MXFP4 路径完全不变。

## 优化点

机制上分三层:注册 `int4_per_group_32` 为 MoE weight 量化 key,但故意不进 online MoE 表,让 `Mxfp4MoEMethod` 在 load 时拥有转换;`_setup_kernel_k3_situ_gfx942()` 把 MXFP4 权重经 `mxfp4_to_f32` 反量化、按 32-group 乘 scale 后转 BF16,再经 `per_1x32_i4_quant` 量化、shuffle 并 pack 成 packed-int4,替换 w13/w2 权重与 scale;用 AITER `moe_2stage_a16wmix` 的 `flydsl_a16w4_gemm1` 作为 kernel,并通过 inspect 签名 fail-closed——若该 AITER build 忽略 SiTUv2 激活会静默算成 SiLU,直接抛错。显存方面,`mxfp4_to_f32` 会把 packed 权重放大 8 倍、一次性全转峰值超 20 GiB/rank,故按 8 个 expert 一批转换并逐批释放源 tensor、预分配输出写 chunk,避免 `torch.cat` 的双份峰值。另外当 SiTUv2 可用时强制 `AITER_BF16_FP8_MOE_BOUND=0`,防止遗留 a8w4 config 让激活保持 bf16。

## 关键改动

```text
vllm/model_executor/layers/quantization/mxfp4.py: _use_k3_situ_int4_gfx942 判定 + _setup_kernel_k3_situ_gfx942 转换与 kernel 搭建
vllm/config/quantization.py: 注册 "int4_per_group_32" -> kInt4Static32 量化 key
vllm/model_executor/layers/quantization/online/base.py: 把 kInt4Static32 排除出 online MoE 表,留给 load-time 转换
tests/models/kimi_k3/test_gfx942_int4.py: opt-in 判定与 fail-closed 行为的单元测试
```

## 性能声明

PR 未给出吞吐/延迟加速比,数值均为精度与功能验证(8x MI325X,TP8 + expert parallelism,Triton RMSNorm/MLA,AITER fused MoE,full + piecewise graph capture;权重路径为 FlyDSL A16W4 SiTUv2 packed-int4,untuned fallback):

- 5-shot GSM8K(lm_eval,temperature=0,max_gen_toks=512,1,319 题):strict/flexible 均 0.9598(stderr 0.0054),与 A16W4 baseline 的 0.9560(stderr 0.0056)在噪声范围内。
- 1M NIAH 在 131072 / 524288 / 1047552 token 全部通过;DSpark k=7 在 262K 通过,DSpark draft token 接受率 104973/241563 = 43.45%。
- `vllm bench serve` 1K 输入/128 输出:并发 1 完成 10/10、并发 16 完成 160/160,无 server error。PR 自述这是功能检查而非调优后的峰值。

## 验证与回归风险

新增 `tests/models/kimi_k3/test_gfx942_int4.py`(override 判定、online 表排除、未知 weight 报错)与 `tests/quantization/test_quantization_config_args.py` 用例;生产 shape(model_dim=3584, intermediate_dim=3072, local_experts=112, topk=15)对 torch 参考做了严格对比。风险:转换有损且仅适用于 Kimi-K3 SiTU 专家 + gfx942 + 显式 override;依赖 AITER v0.1.21.post2(含 ROCm/aiter#4646,vLLM 侧经 #55968 引入);FlyDSL 0.3.2 `gemm2` 在 gfx942 上无法对 `v2bf16` 选 `BUFFER_ATOMIC_FADD`,当前实测走的是 `moe_2stage_a16wmix` kernel;设置全局环境变量 `AITER_BF16_FP8_MOE_BOUND` 可能影响同进程其他 AITER MoE 路径。

## 关联

依赖 #55968(vLLM 引入 AITER v0.1.21.post2);#56930 保证非 MoE draft(如 DSpark)在 `--enable-expert-parallel` 下关闭 expert parallel;后续可加 `Mxfp4MoeBackend.AITER_INT4_BF16`(#55684 方向,本 PR 明确不等待)。同区域另有 gfx950 上的 `_use_k3_situ_aiter` A16W4 路径。
