# PR #40451: [Hardware][Power]Add Power VSX Attention Backend and fix l2 Cache Crash

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#40451](https://github.com/vllm-project/vllm/pull/40451) |
| 作者 | Akashcodes732 |
| 合并日期 | 2026-05-05 |
| labels | ready, v1, cpu |
| 热度 | 👍 1 · 💬 10 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
为 vLLM CPU 后端新增 ppc64le VSX 原生 attention 实现,并修复 `IndexError: unordered_map::at` 初始化崩溃。

## 问题现象
在 PowerPC(ppc64le)平台上:vLLM CPU 后端初始化阶段崩溃,报 `IndexError: unordered_map::at`;同时 CPU attention 后端不支持 Power VSX 指令集,没有原生实现。影响面为所有使用 ppc64le CPU 后端部署的场景。修复后用 `vllm serve ibm-granite/granite-3.3-8b-instruct --port 8000 --max-model-len 512 --max-num-batched-tokens 8192 --dtype bfloat16` 做了 100 并发 benchmark:100 请求全部成功,吞吐 1.15 req/s,Mean TTFT 46663.89 ms,Mean TPOT 1297.55 ms(正文贴出的完整 benchmark 结果)。

## 根因
正文披露:崩溃由 `IndexError: unordered_map::at` 引起,发生在初始化阶段。
(由 diff 推断)崩溃点在 `csrc/cpu/utils.hpp` 的 `get_available_l2_size()`:改动前该函数仅在 `defined(__s390x__)` 分支里用 `at::cpu::get_cpu_capabilities()` 查询 L2 cache size 并对 capability 表做下标查找,PowerPC 平台落入其他分支时,相关能力键在表中不存在,`unordered_map::at` 对缺失键抛出异常,表现为 Python 侧的 `IndexError`。本 PR 给该编译分支加上 `|| defined(__powerpc__)`,让 PowerPC 也走同一查询逻辑。

## 修复方案
- 新增 `csrc/cpu/cpu_attn_vsx.hpp`:实现 `AttentionImpl<ISA::VSX>`,包含基于 `altivec.h` 的 ppc64le GEMM micro-kernel(`gemm_micro_ppc64le_Mx8_Ku4`,M≤8 × N=8 × K 按每次 4 展开)、BF16→FP32 的 LE 字节序正确展开(`vec_mergeh/zeros_u16`),以及 `copy_q_heads_tile`、`reshape_and_cache`;注明 PowerPC 不支持 FP8 KV cache 与 FP16。
- `csrc/cpu/cpu_attn_impl.hpp`:`ISA` 枚举加入 `VSX`;metadata 打印补 `VSX` 分支。
- `csrc/cpu/generate_cpu_attn_dispatch.py`:ISA 编号表加入 `"VSX": 5`,`ISA_FOR_32` 加入 VSX,生成头文件在 `#ifdef __powerpc__` 下 include `cpu_attn_vsx.hpp`,并新增 `#elif defined(__powerpc__)` 分发宏块(`["VSX", "VEC", "VEC16"]`,fp8=False)。
- `csrc/cpu/utils.hpp`:`get_available_l2_size()` 的编译分支由 `defined(__s390x__)` 扩为 `defined(__s390x__) || defined(__powerpc__)`(修复崩溃的关键一处)。
- `vllm/v1/attention/backends/cpu_attn.py`:`_get_attn_isa` 在 `CpuArchEnum.POWERPC` 时返回 `"vsx"`;`_CPU_ARCH_PREFER_MIXED_BATCH` 加入 `CpuArchEnum.POWERPC`。
- `csrc/cpu/cpu_attn.cpp`:ISA hint 解析(`"vsx"`)与 ISA 字符串反查各加一个分支;`cpu_attn_vec.hpp` 的 `load_b_pair_vec` 返回值改为 `std::make_pair`。

## 影响与回归风险
- 回归面:CPU attention 后端的 ISA 选择与生成的分发宏。新增 `#elif defined(__powerpc__)` 分支插在 AMXBF16 与 `__AVX512F__` 之间,理论上会改变某些架构下宏匹配顺序,但各分支以平台宏互斥守护,X86/ARM/s390x 路径应不受影响;`get_available_l2_size()` 只对 `__powerpc__` 增加了处理;VSX 路径不支持 FP8 KV cache 与 FP16,相关请求会走不支持路径。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
