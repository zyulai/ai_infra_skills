# PR #41572: [KDA] Fix ptx_kda prefill NaN without a gate lower bound and workspace growth

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#41572](https://github.com/sgl-project/sglang/pull/41572) |
| 作者 | yuan-luo |
| 合并日期 | 2026-09-30 |
| labels | run-ci, jit-kernel |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
ptx_kda prefill 两个问题:无 gate lower bound 的模型因 per-chunk decay 溢出 fp32 exp 产出全 NaN,无 bound 的批次回退 Triton;workspace 缓存按 batch 形状无限增长至 OOM,缓存超 4 条即清空。

## 问题现象
在 B200 上端到端验证 `--linear-attn-prefill-backend ptx_kda` 发现两个问题(均与 arch 无关,SM103 上即存在):
1. 无 gate lower bound 的模型输出 NaN:Kimi-Linear-48B-A3B-Instruct 上 ptx_kda GSM8K 得 0.000(Triton 为 0.902),6K-token prefill 的每条 prompt logprob 都是 NaN,且无任何日志。
2. workspace 缓存随每个 batch 形状增长直至进程 OOM:GLM-5.3-Flash TP4 服务器用 ptx_kda 在 GSM8K 开始约 20 秒即 CUDA OOM,同参数 Triton 可跑完整评测。

## 根因
正文披露:(1) PTX 内核只实现了"安全 gate"(设置了 `lower_bound`,如 Kimi-K3 与 GLM-5.3-Flash);没有 lower bound 时,diff 注释进一步说明 per-chunk decay 溢出 fp32 exp,kernel 返回 NaN。(2) `get_workspace` 的缓存按 (device, T, nc_tot, H, npieces, N) 键控且"entries live for the process"——服务场景几乎每个 batch 的 (T, N) 形状都不同,每个条目在 16K tokens、16 heads 时约 400 MiB,显存随见过的形状数线性增长直至 OOM。

## 修复方案
- `kda_ptx.py::extend`:eligibility 条件增加 `lower_bound is not None`——无 lower bound 的批次与 spec draft-extend、track-state snapshot、不支持形状一样走 wrapper 现有的 Triton fallback;每次进程只告警一次(`_no_lower_bound_logged`)。
- `kda_prefill.cu::get_workspace`:新增 `constexpr size_t kMaxWorkspaces = 4`,缓存达到 4 条即 `cache.clear()`;device 代码不变;注释同步更新(条目改为"freed when the cache is cleared",并记录 ~400 MiB/条与清理策略)。
- 新增回归用例 `test_ptx_without_lower_bound_matches_triton`(无 bound、`exp(A_log)≈200` 时输出必须有限且匹配 Triton)与 `test_ptx_workspace_memory_is_bounded`(16 个新 batch 形状不得各占一块 workspace),均在 SM103(GB300 CI)运行;`test_kda_ptx.py` 单测参数补 `lower_bound=-5.0`。

关键 diff(kda_prefill.cu):

```cuda
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  if (cache.size() >= kMaxWorkspaces) cache.clear();
```

## 影响与回归风险
- 影响窗口:正文未给版本/提交范围。
- 回归面:无 gate lower bound 的 KDA 模型(如 Kimi-Linear)prefill 一律走 Triton 路径(有告警),不再产出 NaN;有 safe gate 的模型继续走 PTX。workspace 缓存上限 4 条会使命中变 miss,实测 miss 代价:B200、16 heads、16K tokens,1 sequence fused 路径 895–939 µs(hit 764–798 µs),2–9 sequences 573 µs(hit 532 µs)。端到端:Kimi-Linear TP2 ptx_kda GSM8K 0.000 → 0.93;GLM-5.3-Flash TP4 OOM → 0.932(Triton 0.924)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
