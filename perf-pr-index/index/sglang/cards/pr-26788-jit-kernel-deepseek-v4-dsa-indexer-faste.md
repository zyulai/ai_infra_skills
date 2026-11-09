# PR #26788: [JIT Kernel] DeepSeek-V4 DSA indexer: faster top-k + page-table transform (runtime k <= 2048)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @DarkSharpness |
| 合并 | 2026-07-06 |
| 分类 | CUDA Graph / Compiler |
| Label | high priority, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight |
| Diff | diffstat 缺失;素材 diff 截去 9 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/26788 |

## 一句话结论

重写 DeepSeek-V4 DSA indexer 的 fused top-k + page-table transform:运行时 k≤2048,最多快约 1.5x。

## 动机

DeepSeek-V4 DSA indexer 为 sparse attention 每个 query 选 top-k KV 位置并经 page table 变换,该 fused top-k + page-table transform 位于 decode 关键路径,长上下文尤其明显。v1 把 k(512)作为编译期常量,每个 k 需单独编译一个 module;作者声明本 PR 后 v1 应弃用,v2 是 v1 的超集。

## 优化点

topk_v2.cuh 全面重写:k 变为运行时参数(≤2048),单一 JIT module 服务所有 k(不再按 `-DSGL_TOPK` 特化)。按 seq_len 分级 dispatch:Register2(≤8192)与 Register4(≤16384)为寄存器驻留单遍读取,Streaming 次之,Cluster<8> 处理超长项;主内核分 Level 0-3,超过 cluster_threshold 的长项交给 persistent cluster kernel,由 `topk_plan` 按 seq_lens 分布从候选 (threshold, cap) 表选择阈值并压缩出 work list,cap 取 B200 上 cluster-vs-streaming 的实测交叉点;batch ≤15 时 host 侧把 cluster_floor 降到 32768(实测 crossover 约 36-40K)。page-table transform 的 gather 与热 scatter 分离;PDL 串联;支持 ragged per-batch seq len、任意 2 的幂 page size、可选 raw(pre-transform)top-k 索引输出;metadata 从 (B+1,4) 缩为 (B+1,2),workspace 取消。

## 关键改动

```text
python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh: 重写 dispatch(Register2/4、Streaming、Cluster + plan kernel),k 运行时化
python/sglang/jit_kernel/dsv4/topk.py: module 不按 topk 特化;去掉 workspace 参数;新增 out_raw_indices
```

```cpp
 * - Level 0: max_seq_len <= 8192           -> trivial + register<2>
 * - Level 1: max_seq_len <= 16384          -> trivial + register<4>
 * - Level 2: max_seq_len <= cluster_floor  -> trivial + register<4> + streaming
 * - Level 3: max_seq_len > cluster_floor   -> + epilogue process of cluster path
RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");
```

## 性能声明

B200,与 flashinfer 的 `top_k_page_table_transform` 对比(apples-to-apples:两者都做 top-k 与 page-table transform),k=512:几乎所有 (batch, seq) 配置更快,大 batch/长上下文最多约 1.5x;batch ≥ 2048 且 seq ≥ 64K 时维持约 3.0-3.6 TB/s(接近 HBM 峰值)。PR 未注明端到端 serving 数字与模型/精度条件。

## 验证与回归风险

正确性对 `torch.topk` 校验,覆盖所有 dispatch 路径,共 172 个单测案例;CI Extra 通过。影响面:DeepSeek-V4 DSA indexer 的 top-k 路径;metadata 布局 (B+1,4)→(B+1,2)、workspace 参数删除属于调用 ABI 变化,调用方需同步;plan 的候选 (threshold, cap) 表仅按 B200 调优(代码注释注明 "Plan is tunable"),其他硬件行为未在正文给出。

## 关联

参考实现:vLLM persistent_topk.cuh 与 FlashInfer topk.cuh(文件头致谢);素材未列出其他同区域 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
