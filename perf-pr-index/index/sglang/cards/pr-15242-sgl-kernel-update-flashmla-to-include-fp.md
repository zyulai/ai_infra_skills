# PR #15242: [sgl-kernel] Update flashmla to include fp8 sparse_mla optimizations

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hlu1 |
| 合并 | 2025-12-16 |
| 分类 | Quantization |
| Label | sgl-kernel, run-ci |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 为单文件 flashmla.cmake 的 GIT_TAG bump |
| 链接 | https://github.com/sgl-project/sglang/pull/15242 |

## 一句话结论

升级 sgl-kernel 的 FlashMLA 依赖以纳入 fp8 sparse MLA 优化,内核时间 109µs → 85.21µs(1.28x),DeepSeek-V3.2 8k/1k TPOT 改善 1.6%。

## 动机

对应 issue sgl-project/sglang#15211:为 DeepSeek-V3.2 的 sparse MLA(DSA)路径引入 FlashMLA 上游的 fp8 优化。kernel 层面的改动在上游 FlashMLA 仓库完成,本 PR 在 SGLang 侧只做依赖版本提升。

## 优化点

`sgl-kernel/cmake/flashmla.cmake` 中把 FetchContent 的 FlashMLA GIT_TAG 从 `be055fb7df0090fde45f08e9cb5b8b4c0272da73` 提升到 `b1860dc4949e94241eb38503abd7b62eb62d36f6`,使 sgl-kernel 构建产物包含上游 fp8 sparse_mla 优化内核。其余构建配置不变(GIT_SHALLOW OFF 等)。kernel 机制细节在 pack 素材中没有展开,diff 不含源码改动。

## 关键改动

```text
sgl-kernel/cmake/flashmla.cmake: FlashMLA GIT_TAG be055fb7 → b1860dc(纳入 fp8 sparse_mla 优化)
```

```cmake
FetchContent_Declare(
    repo-flashmla
    GIT_REPOSITORY https://github.com/sgl-project/FlashMLA
    GIT_TAG b1860dc4949e94241eb38503abd7b62eb62d36f6
    GIT_SHALLOW OFF
)
```

## 性能声明

测试条件:DeepSeek-V3.2,TP8 + DP8 + DP-attention + disable-radix-cache;bench_serving random 8192 输入 / 1024 输出、range ratio 1、512 prompts。端到端:TPOT 111.98 → 110.22 ms(**1.6% 改善**),median TTFT 61877.25 → 60740.41 ms,request throughput 2.90 → 2.95 req/s,total token throughput 26748.72 → 27164.49 tok/s。内核:attention kernel time 109 µs → 85.21 µs(**1.28x faster**);正文注明注意力 kernel 占比很小(该 8k/1k 场景下 11% → 9%),所以内核的大幅加速在 E2E 上体现有限。精度:GSM8K 20-shot accuracy 0.955、invalid 0.000。

## 验证与回归风险

PR 未新增单元测试,验证依赖 GSM8K 精度评测与端到端 benchmark。风险点:依赖版本 bump 影响所有使用 FlashMLA sparse MLA 的路径(DeepSeek 系列的 DSA prefill/decode);fp8 sparse MLA 与 BF16 路径的选择逻辑在 sgl-kernel/上游侧,本 pack 未包含,回退与边界行为需依赖上游 FlashMLA;构建需重新拉取新 tag,离线环境需同步更新缓存。

## 关联

同区域优化:上游 FlashMLA 仓库(sgl-project/FlashMLA)的 fp8 sparse MLA 内核工作;issue #15211 为其直接前序。本批 PR #48437(AMD HIP split-K sparse MLA)与 #35770(AMD Kimi-K3 Triton MLA prefill)属同一 sparse MLA 领域但面向 AMD。PR 正文未列出其他关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
