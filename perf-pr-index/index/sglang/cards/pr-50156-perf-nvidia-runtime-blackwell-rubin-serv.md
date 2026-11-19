# PR #50156: [Perf] [NVIDIA] runtime: Blackwell/Rubin serving optimizations for hybrid GDN + QSA + MoE models

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @YAMY1234 |
| 合并 | 2026-11-16 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, quant, speculative-decoding, hicache, blackwell, npu, run-ci, jit-kernel, unified-radix-cache, memory-pool, mooncake |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;正文称 Totals: 66 files changed, 5084 insertions(+), 199 deletions(-);diff 摘录含 4 个文件、其中 2 个新文件,另有 62 个文件因篇幅略去) |
| 链接 | https://github.com/sgl-project/sglang/pull/50156 |

## 一句话结论

Blackwell/Rubin 上混合 GDN + QSA + MoE 模型 MTP 投机解码服务的 20 项运行时优化(verify/post-verify、CUDA-graph 输入 staging、HiCache/TreeCore、host batch 准备、PLE host 表),全部 opt-in,由总开关 `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION` 加各子开关控制;每个 commit 独立给出微基准/剖析收益,默认关闭时行为不变。

## 动机

面向 agentic 多轮负载,优化点全部位于每步模型 forward 之外:投机解码的 verify 与 post-verify 工作(verify-tree 采样、draft top-1、post-verify 图与其池、verify commit 的状态回滚、compact verify 的 top-p bound)、CUDA-graph 输入 staging、HiCache 与 Rust TreeCore(KV 与 Mamba state 装载、备份、host 驱逐顺序、tracking slots、write-back demotion)、host 侧 batch 准备(text-only mrope rows)与 PLE host 表。in-forward 的模型 forward 优化不在此 PR——attention、GDN、QSA、MoE、HC、norms、GEMM routes、PLE、prefill/extend kernels 移至 sgl-nvidia-mega 的 NVIDIA megakernel;绑定是现有 megakernel 安装钩子上的小通用接口(#50157)。前一轮 revision 中效应落在噪声内或只服务 shortest-prefill-first 调度的 commit 已被丢弃。

## 优化点(20 个 commit,每个一项,收益行照抄正文)

- #2 多 CTA EAGLE verify-tree 采样:采样 kernel 1 请求 7 µs(原 39)、13 请求 14(原 74)(unit microbench,bitwise);关闭时静态 13 请求 TP1 decode round 长 1.1-1.2%。
- #3 draft top-1 复用 sampler argmax:1 行 2.0 µs(原 8.4)、128 行 12.5(原 46.8);端到端低于噪声。
- #4 host-only Mamba state 与 anchor 在途装载并行取:关闭时 cache-hit 负载 TTFT 长 55%。
- #5 copy plan 缓存的 graph input staging:关闭时每 decode round host 时间多 90-125 µs(静态 13 请求 TP1)。
- #6 post-verify graph key bound 可上调:bound 封顶 64 时 batch 65+ 的每 round host time 高 14%(静态 176 请求)。
- #7 EAGLE post-verify graph pool 按最大 batch 预留容量:否则 176 请求 ramp 1→128 时 OOM。
- #8/#9 HiCache CTA quota:KV load-back span 关闭时长 26-31%;Mamba load 切分后 span 长 65%。
- #10/#14 Mamba side-state stream-ordered 备份与 finished 请求 tracking slot 停放:各在 24 个后台 decode 用户的 cache-hit 负载上 decode round 长 1.9%/2.0%、revisit TTFT 长 14%。
- #11/#13 finished sub-agent session 优先驱逐(关闭时命中率 −0.2pp、匹配 interactivity 吞吐 −0.6/1.1%)与 Mamba host pool 固定占 `--hicache-size` 份额(关闭时 c176 AgentX 命中率 −1.9pp、吞吐 −13%)。
- #12 text-only extend 的 mrope rows 设备端构建:关闭时 8k prefill 长 0.7-1.2%。
- #15 PLE host 表用透明大页:16k-token prefill 的 PLE prefetch gather 3.03 ms(cudaHostAlloc)→ 0.41 ms(huge pages,nsys 两栈并跑);source 实测 5.29 → 1.14 ms;关闭时 c176 AgentX 吞吐 −0.8%。
- #16/#17/#19 tiled GDN ReplaySSM compact commit(bit-identical):post-verify state-commit graph 每 round bs1/2/3/4 约 62/89/117/145 µs → 58/74/91/105 µs;conv-window 与 PLE 回滚合并 launch(每 round kernel 24 → 5、12 → 5;58/74/91/105/122 → 40/54/67/83/99 µs);compact verify top-p bound 双 split-row launch:272 verify rows 68.3 → 29.6 µs(13,600 行 bit-identical)。
- #18 从 host flag 读运行中请求的空 tracking slot:host syncs 9.7/s → 0(c176、`--hybrid-cache-checkpoints-level 3`)。
- #20 write-back demotion 走 stream wait 而非 host wait:每 extend round 阻塞等待 1.51 → 0;extend-round GPU idle 7.22 → 3.04 ms(与 QSA row-map 改动合并计)。

## 性能声明

以上 Gain 全部照抄正文 commit 表(unit microbench / nsys / leave-one-out / Python sampling profile 等,条件见各项)。汇总级数字仅:"Totals: 66 files changed, 5084 insertions(+), 199 deletions(-), including tests."。端到端对比缺失:正文"Accuracy Tests"只给 top-p bound 的采样 GSM8K(temperature 0.6, top_p 0.95, top_k 20,1319 题)0.9780 / 0.9788 vs 无该改动 0.9757 / 0.9757,并注明"GSM8K for this PR vs `main` will be added";Speed Tests 一节为"Benchmark numbers vs `main` will be added."(pack 未提供 vs main 的 E2E 基准)。

## 验证与回归风险

每个 commit 携带单测;触及数值者按 bitwise 与默认路径对比(多 CTA 采样、draft top-1、tiled commit 均 bitwise);唯一非按构造 bitwise 的 top-p bound 在 13,600 随机行上 kept supports 与概率一致,另有上述 GSM8K 抽样对照。CI 状态:Base 与 AMD 通过,Extra 失败。回归风险:(1) 大 diff(66 文件)+ 20 个子开关,开关矩阵未全验;(2) `srt/layers` / `srt/models` 仅两处改动且不在 forward compute(`hybrid_linear_attn_backend.py` 的 verify commit state scatter、`qwen4_exp_ple_table.py` 的 PLE host 表分配),其余在 runtime;(3) 与 megakernel 安装钩子的绑定接口(#50157 / sgl-nvidia-mega#60)耦合;(4) tiled commit 有形状门控(FP32 state、HIP、K 非 2 幂、V 不整除 BV=32、HV 不整除 H 或 4 时回退 stock kernel;FP32 下 tiled fold 差 1 ULP 故不启用)。

## 关联

#50157(megakernel 安装钩子上的绑定接口)、sgl-project/sgl-nvidia-mega(模型 forward 所在,#60);标签含 blackwell、speculative-decoding、hicache、unified-radix-cache、memory-pool、mooncake。本批次内无直接依赖项。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
