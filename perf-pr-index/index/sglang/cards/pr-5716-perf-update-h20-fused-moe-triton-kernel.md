# PR #5716: perf: update H20 fused_moe_triton kernel config to get higher throughput during prefilling

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @saltyfish66 |
| 合并 | 2025-04-27 |
| 分类 | Attention / Kernel (major) |
| Label | 无 |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 为 1 个 tuned config JSON 的参数更新 |
| 链接 | https://github.com/sgl-project/sglang/pull/5716 |

## 一句话结论

为 H20 重新调优 DeepSeek 系 fp8_w8a8 fused_moe_triton 分块配置,prefill 吞吐提升约 4-5%,decode 基本不变。

## 动机

前序 PR #5641 为 H20 提供了 fused_moe_triton 调优配置,但使用后发现 prefill 阶段模型吞吐反而下降约 3%。分析发现该配置的性能强依赖 Triton 版本:在 triton==3.1.0 + torch==2.5.1 环境下能给 DeepSeek-R1-671B 带来约 3% 吞吐提升,而在 triton==3.2.0 + torch==2.6.0 的新环境下变成负收益。需要针对新版 Triton 重新调优,使配置适配新环境。

## 优化点

仅更新 `E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` 这一份调优配置(DeepSeek 系 272 专家、fp8 w8a8、128×128 block 量化)。按 batch size(M 值)重选 Triton tile 参数:大 M 的 prefill 档普遍把 `BLOCK_SIZE_N` 从 64 提到 128(如 1024/1536/2048/4096 档),`GROUP_SIZE_M` 从 64 调到 16 或 32;512 档从 BLOCK_SIZE_M=64/N=64/K=128 改为 16/256/64 并把 GROUP_SIZE_M 提到 64;16 档改为 N=256/K=64/num_stages=4;若干档位调整 `num_stages`(1 档 3→4,2 档 4→3,3072 档 4→3)与 `BLOCK_SIZE_K`(2 档 64→128)。机制上是为新 Triton 编译器重平衡 L2 分组与流水线级数,避免旧 tile 选择在 3.2.0 下的寄存器/缓存压力劣化。

## 关键改动

```text
python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json: 唯一改动,按 batch size 档位更新 BLOCK_SIZE_M/N/K、GROUP_SIZE_M、num_warps、num_stages
```

典型档位对比(512 档,prefill 大 batch 区间):

```json
"512": {
    "BLOCK_SIZE_M": 16,   // 旧: 64
    "BLOCK_SIZE_N": 256,  // 旧: 64
    "BLOCK_SIZE_K": 64,   // 旧: 128
    "GROUP_SIZE_M": 64,   // 旧: 32
    "num_warps": 4,
    "num_stages": 3
}
```

## 性能声明

测试条件:DeepSeek-R1,H20×16(2 节点 TP16),triton==3.2.0 + torch==2.6.0,`SGL_ENABLE_JIT_DEEPGEMM=0`,flashinfer attention backend,`--chunked-prefill-size 32768`,关闭 radix cache 与 overlap schedule。

- Prefill(batch 1,output 0):input 2048 时 (input+output) 吞吐 10845.82→11309.41 token/s;4096 时 12351.59→12812.78 token/s;8192 时 12497.38→12956.29 token/s;16384 时 11352.01→11845.40 token/s。TL;DR:约 4-5% 吞吐提升。
- Decode(batch 32/64/128,input 1,output 512):latency 与吞吐变化在 ±0.7% 内(如 batch 32:925.62→919.88 token/s),TL;DR:几乎无影响。
- 混合负载(bench_serving,50 请求、rate 10、random 4096 输入/512 输出):Total token throughput 4743.74→4761.30 tok/s,Mean TTFT 7731.00→7584.68 ms,Mean ITL 持平(76.28 ms)。

## 验证与回归风险

无新增测试;验证手段为上述 bench_one_batch_server 与 bench_serving 对比,以及 checklist 中勾选的吞吐/延迟 benchmark 项。风险面:配置按 `device_name=NVIDIA_H20` + dtype/block_shape 精确匹配,只影响 H20 上 DeepSeek 系(E=272,N=128)fp8_w8a8 模型的 fused MoE kernel 选择;其他卡型与其他模型不受影响,但配置对 Triton 版本敏感(这正是本 PR 的起因),Triton 再升级后可能需要重调。decode 档位数值有轻微回退(batch 32 约 -0.6%),PR 以 "almost no impact" 接受。

## 关联

前序:PR #5641(提供初始 H20 配置,本 PR 修正其在 triton 3.2.0 下的负收益)。同区域为各卡型 fused_moe_triton tuned config 系列(H20/H100 等),pack 未列出其他具体编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
