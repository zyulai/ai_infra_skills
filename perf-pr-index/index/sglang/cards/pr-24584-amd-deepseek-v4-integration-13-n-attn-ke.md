# PR #24584: amd/deepseek_v4 integration 13/N Attn kernel early exit with cuda-graph 0507

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @1am9trash |
| 合并 | 2026-05-07 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | (无) |
| Diff | pack 未提供 diffstat(diff 正文可用,仅收录 `nsa/tilelang_kernel.py` 一个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/24584 |

## 一句话结论

给 DeepSeek-V4 的 NSA/TileLang attention kernel 加按实际 topk 长度的 early-exit,消除 CUDA graph 捕获按 max_seqlen 计算的冗余,E2E 吞吐提升 10-40%。

## 动机

amd/deepseek_v4 集成分支的第 13/N 个 PR(与 #23600、#23608 并行推进,后两者与本 PR 冲突较大)。开启 CUDA graph 后,graph capture 以 `max_seqlen` 作为 seqlen 固化计算量,而 DeepSeek-V4 的 sparse attention 实际每个请求只需计算有效 topk 个 KV 块,大批量 graph 回放时大量 block 是无效计算。目标是在 kernel 内做 early-exit,只计算有意义的 token(< topk)。

## 优化点

改动集中在 `dpsk_v4_fp8_partial_kernel`(读 FP8 K cache、kernel 内反量化为 BF16、split-K online softmax 的 partial attention kernel,支持双 cache)与 `dpsk_v4_combine_kernel`:kernel 运行时从 `Topk_length_1/2` 张量读取每个 batch 行的实际 topk 长度,计算 `actual_n_groups_1 = ceildiv(tk_len_1, BI * inner_iter_1)`、`actual_n_groups_2` 同理,并与编译期最大 `n_groups` 比较——实际活跃的 group 才做 Q 加载、KV tile 流水线与 GEMM,并完成 finalize(归一化 partial O、写 log2-LSE);被跳过的 group 只把仍是 `-2^30` 的初始 `m_i` 写入 `Partial_LSE` 作为哨兵,由 combine kernel 按 LSE 识别为空 group。这样同一条 CUDA graph 既能按 max_seqlen 捕获,又能在回放时按每请求真实 topk 长度提前退出,且无设备-主机同步。同时把易混淆的参数改名(`topk1→topk_1`、`inner_iter→inner_iter_1` 等)以区分两个 cache 的迭代参数。

## 关键改动

```text
python/sglang/srt/layers/attention/nsa/tilelang_kernel.py:
  dpsk_v4_fp8_partial_kernel: 按 Topk_length 计算 actual_n_groups,跳过超界 group 的 Q/KV 计算与 Partial_O 写
  dpsk_v4_combine_kernel: n_groups 拆为 n_groups_1/n_groups_2,适配跳过组哨兵
```

## 性能声明

PR 正文声明:"This brings 10-40% TPUT improvement on e2e, more pronounced at higher conc."(并发越高越明显)。除此之外正文未给出基准数据表、硬件/模型/workload 条件,Accuracy Tests 与 Speed Tests 章节均为空模板,精度无数据。即:10-40% 为作者断言,pack 内无可核对的数字与测试条件。

## 验证与回归风险

PR 未提供任何测试(checklist 全部未勾选,正文 Accuracy/Speed 章节为空),也无 CI 状态记录。风险面:early-exit 的正确性依赖三个不变量——被跳过 group 的 LSE 哨兵值必须与 combine kernel 的判空约定一致;`Topk_length` 张量必须与 indices 的有效长度严格一致,否则会漏算或多算;TF32/TileLang 层面的 `T.Pipelined` 分支在 graph 捕获时需保持张量 shape 静态(本实现通过常量 grid + 运行时分支做到)。改动范围是 DeepSeek-V4 专属 kernel 路径,不触及通用 attention backend;但该路径属 AMD 集成分支系列,与上游 #23600/#23608 存在已知冲突,合入顺序需留意。

## 关联

系列 PR:amd/deepseek_v4 integration 13/N(前序集成片见 #23600、#23608,本 PR 用于并行整合);同区域:DSA/NSA sparse attention 相关 #39273(K-pool top-k)、#28451(线性注意力 decode)。pack 未列出其他具体 PR 编号。
