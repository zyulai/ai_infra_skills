# PR #39391: fix: clamp NaN/Inf in topk_softmax to prevent duplicate expert IDs

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39391](https://github.com/vllm-project/vllm/pull/39391) |
| 作者 | jhaotingc |
| 合并日期 | 2026-04-21 |
| labels | ready, verified |
| 热度 | 👍 1 · 💬 12 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论

CUDA graph padding 产生 NaN 门控 logits,topk_softmax 输出重复专家 ID 触发 FlashInfer 排序野指针崩溃;在 argmax 前把 NaN/Inf 钳到 0。

## 问题现象

在 H200 上以 FlashInfer CUTLASS MoE + CUDA graphs、高并发服务 Qwen3.5-397B-A17B-FP8(`VLLM_USE_FLASHINFER_MOE_FP8=1`)时,出现 `CUDA error: an illegal memory access was encountered` 崩溃,发生在并发 8+ 请求、MoE batch 超过 256 token 时。端到端实测:8 并发 curl 修复前 5/8 成功、3/8 崩溃,修复后 8/8 HTTP 200;sglang 压测 conc 1-512(ISL=1600、OSL=600、REPEAT=5、4x H200)修复前 conc 16+ 即崩,修复后全过。

## 根因

正文给出完整根因链:CUDA graph replay 把 batch 补齐到最近的捕获尺寸(如 300 个真实 token 补到 512),padding token 携带过期/退化 hidden state,在 MoE router 产生 NaN 门控 logits。`topk_softmax` CUDA kernel 对 NaN 输出全 NaN 的 softmax——IEEE 754 规定 `NaN > NaN` 恒为假,argmax 循环永远停在 expert 0,且对胜者的 `-10000` 置零也失效(`-10000 > NaN` 为假)——于是每个 padding token 产出重复专家 ID `[0,0,0,0,0,0,0,0]`。重复 ID 进入 `cutlass_fused_moe` 后,当 num_tokens > 256 走 FlashInfer 三步排序路径时,`blockExpertPrefixSumKernel` 首个专家匹配后即 `break`,重复槽位使 `unpermuted_row_to_permuted_row` 留有未初始化条目,`finalizeMoeRoutingKernel` 把垃圾值当行索引解引用,越界崩溃。eager 模式无 padding token,两个触发条件(batch > 256 的三步排序路径 + NaN 重复 ID)只在 CUDA graph replay 高并发下同时成立。

## 修复方案

在 `csrc/moe/topk_softmax_kernels.cu` 的 `topkGating` 中,softmax/sigmoid 打分之后、argmax 选择循环之前,把 NaN/Inf 钳为 0:

```cpp
#pragma unroll
for (int ii = 0; ii < VPT; ++ii) {
    if (isnan(row_chunk[ii]) || isinf(row_chunk[ii])) {
        row_chunk[ii] = 0.f;
    }
}
```

全 0 分数下 argmax 按索引打破平局,选出唯一专家 `[0,1,2,...,k-1]`。diff 显示 softmax 与 sigmoid 两条打分路径的输出处也加了同样的钳制。新增回归测试 `tests/kernels/moe/test_fused_topk.py::test_fused_topk_nan_inf_clamp`(修复前 60 failed/12 passed,修复后 72 passed);核性能基准(H200,128/256 experts,batch 1-512,eager 与 graph replay 各 1000 次中位数)差异全部在噪声内(<2%)。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围;触发条件为 FlashInfer CUTLASS MoE + CUDA graphs + 高并发(batch > 256)。
- 回归面:改动位于 `topk_softmax` CUDA kernel 内部,所有经由该 kernel 的 MoE 路由(FlashInfer、Triton 等下游)均受其保护;对正常(非 NaN/Inf)输入钳制是 no-op,实测零开销;行为变化仅限退化输入下的专家选择结果(从重复 [0,...] 变为唯一 [0..k-1]),理论上改变了 NaN 输入时的路由分布,但这正是修复目的。

## 关联

- https://github.com/vllm-project/vllm/issues/39244(修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
