# PR #28953: [LoRA] BF16 support + EP cuda-graph crash fix for experimental_sgl_trtllm MoE-LoRA

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28953](https://github.com/sgl-project/sglang/pull/28953) |
| 作者 | jybsuper |
| 合并日期 | 2026-06-25 |
| labels | lora, run-ci, jit-kernel |
| 热度 | 👍 0 · 💬 3 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
`experimental_sgl_trtllm` MoE-LoRA 快路径:EP + CUDA graph 下 NVFP4 批量服务因路由结果 -1 索引越界而 IMA 崩溃,permute kernel 跳过未路由 slot;同时新增 BF16 支持与 `experts_shared_outer` LoRA。

## 问题现象
NVFP4(Kimi-K2.5)场景:`--moe-runner-backend experimental_sgl_trtllm`、two-stream overlap 开启、CUDA graph 开启、rank-16 adapter 时,批量的 `bench_one_batch_server` 在 prefill 期间以非法内存访问(TMA descriptor init)崩溃。测试硬件为 2 节点 × 8× GB300(TP8/EP8,MNNVL)、flashinfer 0.6.12。修复后 bs1(最小复现)至 bs128 均不再崩溃,贪心生成输出连贯。BF16 路径另有两条正确性修复:路由参数对齐(routing arg parity)与 CUDA-graph 分配安全(正文仅列出名称,未给出现象)。

## 根因
正文披露:该 IMA 崩溃发生在"EP + CUDA graph + two-stream overlap"组合下的批量服务中,但正文未展开机制。具体机制来自 diff 新增注释(由 diff 推断):在专家并行(EP)下,routing 会对"专家不在本 rank"的 (token, k) slot 产出 `permutedIdx == -1`;`permuteKernel` 中对该 slot 的写出此前没有防护,`params.outPtr[permutedIdx * params.hiddenDim + hiddenIdx]` 在 prefill 规模下以负索引越界写输出缓冲,触发 illegal memory access;`moe::dev::finalize` 本身已会跳过 `permutedIdx == -1`。

## 修复方案
1. **崩溃修复**(`jit_kernel/trtllm_lora_temp/data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu` 的 `permuteKernel`):对 `permutedIdx < 0` 的 (token, k) slot 直接 `continue` 跳过写出:

```diff
         int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
+        // Skip EP-unrouted (token, k) slots: under expert parallelism the routing emits
+        // permutedIdx == -1 for slots whose expert is not on this rank. ...
+        if (permutedIdx < 0) {
+          continue;
+        }
         params.outPtr[permutedIdx * params.hiddenDim + hiddenIdx] = data;
```

2. **BF16 支持**:新增 `trtllm_bf16_routed_moe_lora`(分解式 pipeline:routing → permute → gate_up GEMM → LoRA-aware activation → down GEMM → finalize,bf16 端到端无量化;`do_finalize=False` 时返回中间量供 python 侧 down-LoRA + 复用 finalize),对应 CUDA 侧 `Bf16LoraLauncher`/`sgl_trtllm_bf16_routed_moe_lora`(仅支持 gated SwiGLU、要求预计算 packed `expert_indices`、`intermediate_size` 为 128 倍数),python 侧新增 `fused_experts_none_to_experimental_sgl_trtllm_bf16_lora` 分发函数与 two-stream override 注册(无活跃 LoRA 的非 capture decode 回落普通 bf16 快路径);
3. 新增 `experts_shared_outer` LoRA 支持,以及 `SGLANG_OPT_LORA_PREFILL_ROUTING_REUSE` 环境 prefill 路由复用(A/B 阶段统一 routing BLOCK_SIZE_M,使对齐/排序每层只跑一次)。

**注:pack 的 diff 在 40000 字符处截断(two-stream BF16 实现 `moe_overlap.py`、`FlashInferTrtllmBf16MoeQuantInfo` 等"routing arg parity + CUDA-graph allocation safety"两条 BF16 正确性修复的具体代码未包含在素材中),以上第 2、3 点细节以 diff 已展示部分与正文描述为准。**

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:`jit_kernel/trtllm_lora_temp` 的 CUDA kernel(permuteKernel 行为变化影响所有走该 kernel 的 MoE 路径,含 fp8/fp4/bf16)、新增 bf16 LoRA 算子与 launcher、`lora_dispatch.py`/`trtllm_lora_temp/__init__.py`/`environ.py`。two-stream override 现覆盖 fp8/fp4/bf16 三个 MoE-LoRA 入口;permute kernel 新增的跳过分支只在存在 `permutedIdx == -1`(EP 未路由 slot)时生效,非 EP 场景行为不变(由 diff 推断)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
