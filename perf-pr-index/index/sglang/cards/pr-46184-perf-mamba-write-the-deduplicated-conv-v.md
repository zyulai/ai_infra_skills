# PR #46184: [Perf] mamba: write the deduplicated conv verify window as one tile

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jianc99 |
| 合并 | 2026-10-29 |
| 分类 | Speculative Decoding (tier: major) |
| Label | run-ci, jit-kernel, memory-pool, bypass-fail-fast |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/46184 |

## 一句话结论

Mamba 线性链投机验证的 deduplicated conv 窗口从每 token 3 次散写改为一次 tile 写,GB300 上 bs128/T8 核时 106.45→17.55 us,Qwen3.8+DFlash2 e2e decode +5%。

## 动机

混合模型的线性链投机验证(EAGLE chain、MTP、DFLASH、DSPARK 等)在 verify 时调用 `causal_conv1d_update` 并携带中间 conv 窗口,用于恢复被接受前缀的 conv state。CUDA/ROCm 上 `MambaPool` 对该窗口去重存储:每 slot 一条 `[dim, T+K-2]` 行,以步长 stride 等于窗口 stride 的重叠 `[T, dim, K-1]` 视图暴露。但内核仍按 token 写窗口,每个 draft token 做 3 次 lane-strided 窄散写,每列最多被写 3 次。GB300(dim 10240、bf16、T=8、bs=128)上去重窗口 verify conv 需 106 us,而稠密窗口仅 39 us,差距明显。

## 优化点

在 `_causal_conv1d_update_kernel` 中新增 `DEDUP_CONV_WINDOW` 编译期路径:利用线性链下"第 t 步窗口即共享行 `[state[1], state[2], x_0, ..., x_{T-1}]` 的第 t..t+2 列"这一性质,在 token 循环前把整行按 `[BLOCK_N, next_pow2(T+2)]` tile 一次性写出,并跳过每 token 的窗口写;tile mask 精确覆盖原路径写的 T+2 列,padded 行在 store 前退出。`causal_conv1d_update` wrapper 仅在三个条件同时成立时走 tile:窗口视图的 step stride 等于 window stride、验证为链式(无 tree mask)、conv width 为 4——此时 tile 写与逐 token 写的地址和值完全相同,任何满足该 stride 的布局均等价,MambaPool 的去重视图是其中之一。两条 `tl.static_assert` 保证非 width-4 或 tree mask 场景不会编译该分支;不走 tile 的 kernel 传 `NP2_WINDOW_COLS=1`,不产生额外编译变体。无新参数,调用方不变。

## 关键改动

```text
python/sglang/kernels/ops/mamba/causal_conv1d_triton.py: 新增 DEDUP_CONV_WINDOW tile 写路径与 stride/链式/width=4 三重 gate
python/sglang/srt/mem_cache/memory_pool.py: 更新去重窗口注释(tile 写语义)
test/registered/layers/mamba/test_causal_conv1d.py: 新增 test_causal_conv1d_update_dedup_window_matches_per_token
```

```python
+    dedup_conv_window = (
+        intermediate_conv_window is not None
+        and retrieve_next_token is None
+        and width == 4
+        and stride_inter_step == stride_inter_win
+    )
```

## 性能声明

- 单核每调用耗时(GB300,dim 10240,K 4,bf16,CUDA graph):bs1/T8 5.35→2.49 us;bs64/T8 57.39→9.96 us;bs128/T8 106.45→17.55 us(稠密窗口 38.58 us);bs128/T16 322.63→39.10 us。
- 端到端:Qwen3.8-27B + DFlash2(block 8),TP1、单 GB300,链式验证、bf16 mamba state、overlap 调度、`--max-running-requests 128`,SPEED-Bench 首轮、T=1、top_p 0.95、top_k 20、最多 4096 输出 token,3 次交错运行均值:decode tok/s c64 6048→6361(+5.2%),c128 6992→7329(+4.8%)。
- 低并发单次运行:c8 +3.6%(2031→2104)、c32 +5.3%(4723→4973);c1 在噪声内;greedy c64 6746→7145(+5.9%)。acceptance length 各轮 3.59–3.63。
- 仅 GB300 实测;ROCm 走相同路径但未运行。

## 验证与回归风险

- 新测试覆盖 T∈{2,3,8,16}、`num_accept_tokens` 偏移、MambaPool 轴-0 转置视图、fp32/bf16、部分 channel block、有无 padded rows,对共享行/窗口/conv state 做逐位比对;另做 compute-sanitizer memcheck/racecheck/initcheck 清洁,72 个扩展用例逐位对齐,端到端 greedy 输出与 main 完全一致。
- 回归风险面:改动作用于所有 topk≤1 链式验证(GDN 系列 Qwen3.5/3.8、Mamba2,CUDA 与 ROCm);稠密窗口(UnifiedMambaPool、KDA/Kimi-Linear、tree verify)与 width≠4 的去重窗口(如 LFM2 width 3)保持原路径。
- 与 #38897 的 `causal_conv1d_update_varlen`(仍逐 token 写窗口)无文本冲突,但两者都合并后 packed verify 不享受 tile 写;#46189 是覆盖 packed 行的另一路线。CI 显示 Extra 与 AMD ROCm 10 两条失败。

## 关联

- 同系列:本 PR 为 DFlash adaptive-verification 系列 component 1,集成与全系统结果见 #46182;相关内核工作 #38897(varlen packed kernel)、#46189(packed 行并入本 kernel)。
- 同分类:DFLASH 投机解码 #22077 等。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
