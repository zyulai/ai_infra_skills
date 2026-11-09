# PR #29551: sgl-kernel: bump sgl-attn for varlen num_splits OOM fix

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29551](https://github.com/sgl-project/sglang/pull/29551) |
| 作者 | JustinTong0323 |
| 合并日期 | 2026-07-02 |
| labels | sgl-kernel, run-ci, run-ci-extra |
| 热度 | 👍 0 · 💬 14 |
| 分类 | Attention / Kernel |
| bug 类型 | resource |

## 一句话结论
varlen 下大量短段(NSA 式 cu_seqlens_q=arange)使 get_num_splits 低估 total_mblocks、过度切分,fp32 out_accum 工作区爆掉 OOM;把 vendored sgl-attn 提到 f89bc23 引入启发式修复。

## 问题现象
varlen batch 由大量短段组成(NSA 风格的 `cu_seqlens_q = arange(N+1)`)且 K 很长时,先前 `get_num_splits` 的 "pretend batch=1" 上界低估 `total_mblocks`,导致过度切分(over-split),fp32 的 `out_accum` 工作区体积爆涨,发生 OOM。该 OOM 在 DeepSeek-V3.2 NSA prefill 中被观察到。正文未附具体 OOM 日志。

## 根因
正文披露:根因在 sgl-attn 的 varlen `get_num_splits` 启发式——对 varlen 输入沿用「假装 batch=1」的上界,在许多短段场景下低估 `total_mblocks`,从而选择过多的 split 数,`out_accum`(fp32 累加缓冲)按 split 数放大后 OOM。上游修复为 [sgl-project/sgl-attn#46]:varlen 时按真实 `total_q` 计算 heuristic 大小,非 varlen 路径保持 byte-identical。

## 修复方案
本 PR 是 vendored 依赖的 bump(把上游修复带进 sgl-kernel):

1. `sgl-kernel/CMakeLists.txt`:`repo-flash-attention` 的 pin 从 `65c54cc5`(65c54cc5a6d29fee56036484c749bc5b8e00fd66)升到 `f89bc23`(f89bc2306632d1ec5f97b014dded4254f5b4a907),同步更新 `URL_HASH`(SHA256=418b5681...)。相对当前 pin 的净变更恰好是 sgl-attn#46 这一个 commit。
2. `sgl-kernel/tests/test_flash_attention.py`:按 sgl-attn#46 review 的要求把回归测试移到这里——新增 `test_flash_attn_varlen_many_short_segments_num_splits`:64 个长度为 1 的短 Q 段、每段 K 长 2048(nheads=8, nheads_kv=1, d=128),`num_splits=0` 走 `get_num_splits` 启发式,输出与逐段 fp32 参考实现比对(atol/rtol=2e-2)。

关键 diff 摘录(CMakeLists pin):

```diff
-    URL      https://${GITHUB_ARTIFACTORY}/sgl-project/sgl-attn/archive/65c54cc5a6d29fee56036484c749bc5b8e00fd66.tar.gz
-    URL_HASH SHA256=53307eefba65d9ac26092433a8bc4d48182ce0bb27132e69ab0b8d6ff8ee47db
+    URL      https://${GITHUB_ARTIFACTORY}/sgl-project/sgl-attn/archive/f89bc2306632d1ec5f97b014dded4254f5b4a907.tar.gz
+    URL_HASH SHA256=418b5681584dc3efff496a1cab5ffd58d2728d89dcfe0ea16e6985d6ef35c68c
```

作者在 H100(sm90, torch 2.11+cu130)用新 pin 构建 sgl-kernel 并运行回归测试,结果 `1 passed`。

## 影响与回归风险
- 影响窗口:使用旧 pin(`65c54cc5`)构建的 sgl-kernel,在 varlen 大量短段 + 长 K 场景(如 DeepSeek-V3.2 NSA prefill)触发。
- 回归面:所有经 sgl-kernel 使用 sgl-attn flash-attention 的路径都会换到新 pin;上游修复保证非 varlen 路径 byte-identical,主要行为变化集中在 varlen 的 `num_splits` 启发式(split 数变化会改变数值累加顺序,输出可能有可容忍的浮点差异)。pin bump 也带入了上游该 commit 的其他改动(正文称净差恰为这一个 commit)。新增回归测试仅在 sm90/sm80 上运行。

## 关联
- https://github.com/sgl-project/sgl-attn/pull/46(上游启发式修复,正文给出)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
