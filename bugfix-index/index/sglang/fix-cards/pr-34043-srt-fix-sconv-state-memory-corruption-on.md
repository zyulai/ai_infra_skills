# PR #34043: [srt] Fix sconv state memory corruption on specdec

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34043](https://github.com/sgl-project/sglang/pull/34043) |
| 作者 | ekzhang |
| 合并日期 | 2026-08-09 |
| labels | run-ci, bypass-fastfail, run-ci-extra |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

DFlash/DSpark 投机解码下 mamba 索引跟踪被 `spec_algorithm.is_none()` 守卫关掉,radix cache 命中的请求用全零 sconv 状态求解,长上下文精度崩坏;放宽守卫并补 capture 安全与 padding 清零。

## 问题现象

DFlash/DSpark 投机解码池在 Inkling、Inkling-Small 等 hybrid-SWA 模型启用 radix cache 时,损坏 sconv(short-conv)状态,对长上下文影响尤其明显。实测:SWEBench Verified / Pro / Multilingual 的 70 题子集上,DFlash 把通过率从 55% 拉低到 35%——长上下文 Inkling rollout 因 sconv cache 损坏而卡住、「健忘」。

## 根因

正文披露:mamba 跟踪的启用条件被门在 prefill-graph capture 的判断上,其中包含 `spec_algorithm.is_none()`——开投机解码时跟踪被关闭,于是任何 radix cache 命中的请求会以全零 sconv cache 张量来求解。从 diff 看(由 diff 推断):`prefill_cuda_graph_runner._is_mamba_track_enabled` 的守卫在投机算法非 None 时直接返回 False,prefill graph 不跟踪 mamba 索引;sconv 后端在无跟踪元数据时 capture 期直接把 track scatter 从图中特化掉,重放时无法按需 scatter,读到的就是全零/残留状态。

## 修复方案

三处:

1. `prefill_cuda_graph_runner.py`:`_is_mamba_track_enabled` 去掉 `spec_algorithm.is_none()` 条件,投机解码下也启用 mamba 索引跟踪;
2. `inkling_sconv_backend.py`:capture 安全——新增 `_graph_track_inert_{mask,indices,seqlens}` 惰性缓冲,无跟踪元数据的 capture warmup batch 也必须启动 track scatter(全行 mask 掉),避免 Python 层 `if` 把 scatter 从捕获图中特化移除;
3. `runner_utils/buffers.py` `populate_from_forward_batch`:静态输入缓冲的 `mamba_track_*` padding 行清零,元数据缺失时对应行也清零。

验证:同一 DSpark 池 A/B,10.5k token 提示词在 fresh 与 cache-resumed 前缀下(cached=8960)对比——修复前 verify logits 偏差 0.5-3.0、greedy top-1 从第二步起翻转;修复后每一步 bit-identical(0.0000),捐赠 checkpoint 与非投机池完全一致;70 题 SWEBench 子集 DSpark 0.486 vs 非投机 0.471(修复前 0.257 vs 0.457),达到 parity。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:开启 mamba extra buffer + radix cache 的 prefill graph 路径(投机解码场景从「不跟踪」变为「跟踪」);sconv 后端 capture 语义变化(总启动 track scatter);静态缓冲 padding 行清零为纯防御性改动。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
