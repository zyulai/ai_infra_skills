# RFC #58638: [RFC]: KV cache grouping for hybrid models with speculative drafters

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#58638](https://github.com/vllm-project/vllm/issues/58638) |
| 作者 | LucasWilkinson |
| 创建日期 | 2026-09-25 |
| 状态 | 🟢 open |
| labels | RFC, kimi |
| 热度 | 👍 3 · 💬 4 |
| 分类 | Speculative Decoding |

## 一句话结论
修复推测 drafter 在混合模型上破坏 KV cache 分组启发式的两类问题:组数爆炸与 full attention 填充层,方案为按最坏填充字节选组大小 + 优先 BLHNC 布局。

## 提案动机
混合模型上的推测 drafter 会新增一个小 KV cache bucket,从两方面打破现有分组启发式(按 bucket 数取 min,或 max < 1.5×min 时取 max):
- **组数过多(Qwen3.5/3.6 + DFlash)**:一层 drafter bucket 令 group size 变成 1,每层一组——Qwen3.5-27B 从 4 组变 70 组(#49730),Qwen3.6-35B-A3B 从 4 组变 46 组。每组每步重建 attention metadata(GDN 构建器很贵,#52297 实测一次 GDN build() 约 900 µs 且逐组重复),DFlash 增益大打折扣;而强行少分组在现有分组方式下要牺牲 KV 容量(#49730:强制 gs=16 恢复约 10% tok/s 但容量 216k → 141k tokens)。
- **full attention 填充(MiMo-V2.5 + DFlash)**:5 层 drafter bucket 决定 gs=5,除不尽 target 的 9 个 full attention 层,于是为整个上下文分配一个 full attention 填充槽(#58175)。单个 full attn 填充槽每请求可达 2.5 KiB/token × 256k tokens ≈ 640 MiB(1M 上下文约 2.5 GiB),远贵于有界的 SW 填充槽(~0.6 MiB)。Kimi-K3 + DSpark 同理:5 层 drafter bucket 填充 24 个 MLA 层,20 组。
已核查不受影响:GLM-5.3-Flash + DFlash2、MiniMax-M3 + DSpark、DeepSeek-V4-Flash + DSpark。

## 提出的方案
两部分互补(状态:Draft;PRs:#58207 组大小、#58380 布局,后者 stack 于前者):
- **Part 1(#58207):按最坏填充字节选组大小**。把启发式替换为在 `[下限, 最大 bucket]` 上搜索,最小化"填充槽数 × max_memory_usage_bytes"并兼看组数;下限不小于最小 bucket 且不小于 `--min-kv-cache-group-layers`(默认 3),从而单层 drafter bucket 无法再逼出逐层分组;按字节计价使 full attention 填充(随上下文增长)被有界的 SW/mamba 填充替代。
- **Part 2(#58380):混合 page size 时优先 `BLHNC` 布局**。block-outermost 布局让不同大小的 page 共享一个 block、免于 padding 到统一尺寸;仅当所有后端都能整块运行每个 attention block 时可用(B300 上 Triton 可、FlashInfer 16/32/64 不可、FA4 hd256 128 不可),且仅当 LBHNC 分组会填充 full attention 或 BLHNC 组数更少时才切换(成本检查)。
- 落地顺序建议:先合 #58207(对所有布局/后端都有益),再合 #58380;`VLLM_KV_CACHE_LAYOUT` 与 connector 偏好仍优先。

## 优化点与收益
- Qwen3.6-35B-A3B + DFlash(1× B300,GSM8K,c=1/c=32 tok/s):main 46 组、7.42M 容量、460/4654;#58207 17 组、3.05 GiB full attn 填充、5.19M、733/6495(即 +54% @c=1、+36% @c=32);#58380(Triton,BLHNC)5 组、58 MiB 填充、7.53M、746/7463。
- MiMo-V2.5 + DFlash(TP2):main 11 组、726 MiB 填充、5.76M;#58207 7 组、无 full attn 填充、469 MiB、5.96M;#58175 18 组、90 MiB、6.25M;BLHNC(未被 #58380 选中)12 组、266 MiB、6.10M。
- Kimi-K3 + DSpark(TP4):main 20 组 + 1 个 MLA 填充层;#58207 17 组、无 MLA 填充、KV 容量 +0.37%(4.76M → 4.78M @131k)。
- 测量注记:单次运行,噪声约 2-3%;Qwen 各运行接受长度 5.1-5.2;`--min-kv-cache-group-layers` 的权衡(默认 3 vs 4)是开放问题。

## 现状与落地
状态 open,正文标注 Draft;两个 PR(#58207、#58380)已在正文中列为实现载体,推荐先 #58207 后 #58380。开放问题还包括 BLHNC 在 FlashInfer/FlashAttention 下的重分页、FA drafter block size(#58630 单独修复)、以及超出 DFlash/MiMo 范围的精度与吞吐验证。

## 关联
- #58207(group size by worst-case padding bytes)
- #58380(prefer BLHNC)
- #49730、#51742、#52297、#58175、#42505、#46462(相关报告)
- #55219、#57169(sparse MLA 重分页参考)、#58630(FA drafter block size)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
