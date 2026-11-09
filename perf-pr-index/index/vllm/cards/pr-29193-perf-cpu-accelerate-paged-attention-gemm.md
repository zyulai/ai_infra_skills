# PR #29193: [perf][cpu] Accelerate paged attention GEMMs (QK, PV) on Arm CPUs with NEON

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fadara01 |
| 合并 | 2025-11-22 |
| 分类 | Attention / Kernel(major) |
| Label | performance, ready, v1 |
| Diff | diffstat 缺失(diff 全文已提供,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/29193 |

## 一句话结论

为 Arm CPU 的 paged attention(QK/PV 两个 GEMM)新增 NEON FMLA 微内核,`cpu_attention_with_kv_cache` prefill 提速 2x、端到端吞吐 +13%,从此在 Arm 上解锁 chunked prefill、prefix caching、attention sink 等特性。

## 动机

#27954 引入的 `cpu_attention_with_kv_cache` 支持 chunked prefill、prefix caching、SWA、alibi、softcap 与 attention sink(依赖 #23934 方向),但在 Arm CPU 上 prefill 阶段被禁用:较长 prefill 下其 QK/PV GEMM 比 `torch.sdpa` 慢,导致 Arm 只能退回 sdpa 路径,上述特性全部不可用(修复 #28981 的 Arm 部分)。

## 优化点

新增 `csrc/cpu/cpu_attn_neon.hpp`(仅 `__aarch64__` 编译):以 `AttentionImpl<ISA::NEON>` 实现 attention 框架中的 GEMM 钩子。核心是 `gemm_micro_neon_fmla_Mx8_Ku4` 微内核——M≤8 行 × N=8 列、K 按 4 展开,`vfmaq_laneq_f32` 双累加器 FMA,QK 阶段按 BLOCK_SIZE_ALIGNMENT=32 分块、PV 阶段按 HEAD_SIZE_ALIGNMENT=32 与 dynamic k 分块,全程 fp32 累计;B(K/V cache)按 fp32/fp16/bf16 三种 dtype 特化加载转 fp32(bf16 无硬件支持时用移位拼装模拟);`TileGemmNeonFMLA` 按 M 的 8/4/2/1 余数分派 micro kernel,N 保证 8 对齐无尾循环。另提供 `reshape_and_cache`(K 写成列主序 `[head_dim, block_size]`、V 行主序,`omp parallel for collapse(2)`)与 `copy_q_heads_tile`(Q 转 fp32 并乘 scale)。`ISA` 枚举增加 NEON,isa hint `"neon"` 接入 dispatch;Python 侧 `_get_attn_isa` 在 ARM 且 block_size%32==0 时返回 "neon",ARM 加入 `_CPU_ARCH_PREFER_MIXED_BATCH`,并从 V1 禁用 chunked prefill/prefix caching 的架构列表(POWER/s390x/RISCV)中移除 ARM。

## 关键改动

```text
csrc/cpu/cpu_attn_neon.hpp: 新增 NEON attention 实现(QK/PV FMLA 微内核 + reshape_and_cache)
csrc/cpu/cpu_attn.cpp: NEON_DISPATCH 分派与 "neon" ISA hint 解析
csrc/cpu/cpu_attn_impl.hpp: ISA 枚举加 NEON
vllm/v1/attention/backends/cpu_attn.py: ARM 选 neon;ARM 偏好 mixed batch
vllm/engine/arg_utils.py: ARM 不再禁用 chunked prefill/prefix caching
```

微内核核心操作:

```cuda
acc##i##_0 = vfmaq_laneq_f32(acc##i##_0, b0, aiv, L);
acc##i##_1 = vfmaq_laneq_f32(acc##i##_1, b1, aiv, L);
```

## 性能声明

- 内核对比(ISA::NEON vs ISA::VEC):batch size=64、query tokens=kv tokens=512、q heads=32、kv heads=8、head size=128、block size=128 时,prefill attention 加速 2x
- 端到端吞吐(Arm Neoverse-V2):Llama-3.1-8B-Instruct(load-format dummy),sharegpt 128 prompts、input-len 1024、output-len 128、max-model-len 2048、max-num-batched-tokens 8192,`VLLM_CPU_OMP_THREADS_BIND=0-63`,prefill/decode 均走 `cpu_attention_with_kv_cache`:NEON 比 VEC 吞吐高约 13%,与 prefill 用 torch.sdpa 的吞吐相当(具体 tok/s 数值 PR 未列,以文字描述)
- Future Work 注明:bfmmla/bfdot 与向量化 exp 预期再带来 2x

## 验证与回归风险

Test Plan 为 `.buildkite/scripts/hardware_ci/run-cpu-test-arm.sh`(含 `tests/kernels/attention/test_cpu_attn.py`),Test Result 称全部通过。风险面:NEON 路径要求 head_dim 与 block_size 按 32/16 对齐(`static_assert`),非对齐配置回退 vec16;bf16 无 `ARM_BF16_SUPPORT` 时走软件转换;X86/AMX 与 POWER 等路径不受影响,但 arg_utils 放开 ARM 的 chunked prefill 会改变 Arm 默认调度行为。

## 关联

Fixes #28981;Related:#23934(attention sink on Arm)、#27954(`cpu_attention_with_kv_cache` 引入)。同区域其他 Arm CPU kernel PR 编号素材未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
