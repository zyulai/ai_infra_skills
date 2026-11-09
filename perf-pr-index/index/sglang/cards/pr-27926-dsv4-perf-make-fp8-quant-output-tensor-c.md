# PR #27926: [DSV4] perf: Make FP8 quant output tensor contiguous

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mattteochen |
| 合并 | 2026-07-08 |
| 分类 | Quantization(classify.py 给出,tier: major) |
| Label | quant, deepseek, sgl-kernel, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 显示 5 个文件(含 2 个新文件与新增测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/27926 |

## 一句话结论

为 DeepSeek-V4 wo_a 的 FP8 DeepGEMM einsum 新增专用 group-major 量化 kernel,避免非连续输入静默回退到慢路径,B300 上 1k1k 场景吞吐提升 2-4%。

## 动机

DeepGEMM 的 FP8 einsum 在输入张量非连续时会静默回退到未优化路径。此前 DSV4 的 `wo_a` 路径把 `o: [T, G, D]` 展平为 `[T*G, D]` 再把 scale 变形回 `[T, G, -1]`;在 SM100 FP8 einsum 路径上 scale 是按 group slab 消费的,跨 `T*G` 展平会让某个 token 的 FP8 码与另一个 group 的 scale 错配(隐式修复 issue #29038)。本 PR 是 PR #27680 的替代实现:让量化 kernel 直接输出 DeepGEMM 要求的布局,不再 flatten/contiguous 拷贝。

## 优化点

- 新增专用 JIT kernel `fp8_wo_a_group_major_quant.cuh`(比通用 `per_token_group_quant_8bit_v2` 更窄):输入为 hidden 维连续的 `[T, G, D]` strided 视图,输出连续 `[T, G, D]` FP8 码;scale 为 FP32 UE8M0 2 的幂,输出为逻辑 `[T, G, D/128]` 视图但底层是 group-major `[G, T, D/128]` 连续存储,保证 `scale[t, g]` 与 `o[t, g, :]` 的关联,且每个 group/head 的 `[T, S]` scale panel 对 DeepGEMM recipe=(1,1,128) 消费者是连续的。通用 kernel 无法在读 strided 视图的同时产出连续码 + group-major scale 而不额外做一次全量拷贝。
- kernel 细节:group_size 固定 128,8 线程/subwarp 处理一组,`int4` 32B 向量化读写,subwarp 内 `reduce_max` 求 absmax,UE8M0 scale 直接以位运算构造(`<<23`),支持 PDL;fast-math 编译与 AOT/JIT v2 路径一致保证 FP8 舍入 bit-identical。
- 注册为 custom op(`mutates_args=["output_q","output_s"]`),模型侧 `deepseek_v4.py` 的 wo_a 前向从 `sglang_per_token_group_quant_fp8(o.reshape(T*G,D).contiguous(), ...)` 换成 `sglang_per_token_group_quant_fp8_dsv4_wo_a(o)`,消除 flatten+contiguous 拷贝,直接喂 `deep_gemm.fp8_einsum`。

## 关键改动

```text
python/sglang/jit_kernel/csrc/deepseek_v4/fp8_wo_a_group_major_quant.cuh: 新增,group-major 输出布局的 wo_a 量化 kernel
python/sglang/jit_kernel/dsv4/fp8_wo_a.py: 新增,custom op 封装与 sglang_per_token_group_quant_fp8_dsv4_wo_a 入口
python/sglang/srt/models/deepseek_v4.py: wo_a 前向改用新量化入口,消除 flatten/contiguous
test/registered/jit/deepseek_v4/test_fp8_wo_a.py: 新增,bit-exact 与 einsum 消费契约测试
```

```python
def sglang_per_token_group_quant_fp8_dsv4_wo_a(x):
    """[T, G, D] -> fp8 [T, G, D] + scale 逻辑 [T, G, D/128](底层 [G, T, D/128] 连续)"""
    x_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    x_s_storage = torch.empty((num_groups, num_tokens, hidden_groups), ...)
    fp8_wo_a_group_major_quant_ue8m0(x, x_q, x_s_storage)
    return x_q, x_s_storage.transpose(0, 1)
```

## 性能声明

8×B300,DSV4-Pro,Pareto 1k1k(ISL/OSL=1024)、无 MTP,opt 相对 base:conc=4 TPS/user 81.90→85.37(1.04x)、TTFT 267.20→235.97 ms(1.13x);conc=8 总 TPS/GPU 135.75→141.26(1.04x);conc=16 1.03x;conc=32 1.03x;conc=64 1.02x。精度回归:GPQA 198 题×16 repeats pass@1(avg-of-16)88.04%±1.56%,pass@16 96.46%;GSM8K 1319 题 96.29%。PR 未注明服务并行配置(TP/EP)细节。

## 验证与回归风险

新增 registered 测试 `test_fp8_wo_a.py`(SM≥100、需 deep_gemm):与通用 flat UE8M0 量化 bit-exact(FP8 码与 scale 全等)、strided 输入、空 token 维、最大 30 万 token 维、以及 DeepGEMM fp8_einsum 消费契约(对 torch einsum 反量化参考 atol=1e-1/rtol=2e-2);CI 注册为 base-b-kernel-unit(4-gpu-b200)。CI 快照 Base/Extra 失败(以合并时为准)。风险:kernel 对输入有硬约束(hidden 维 128 对齐、16 字节对齐、stride 对齐),不满足时 RuntimeCheck 报错而非回退;group-major scale 布局仅适配 DeepGEMM recipe=(1,1,128) 的消费方式。

## 关联

替代实现:PR #27680;隐式修复 issue #29038。同区域:DeepSeek-V4 FP8/量化线(PR #26471 online compress、PR #32755 indexer quant kernel)、DeepGEMM 路径优化。PR 正文未列出其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
