# PR #39769: Skip fully-masked KV tiles in sliding-window extend attention

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jiaqiang-dot-liu |
| 合并 | 2026-10-13 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 显示改动 extend_attention.py kernel 循环边界 + swa 测试新增 3 个用例 |
| 链接 | https://github.com/sgl-project/sglang/pull/39769 |

## 一句话结论

Triton SWA extend attention 的 stage-2 KV 循环从窗口下界 tile 起步,跳过全被 mask 的 KV tile,长 extend 小窗口下 kernel 最高 −39%。

## 动机

sliding-window 层在 `_fwd_kernel` 的 stage-2 循环中从 0 扫描 extend KV。窗口 mask 只保留满足 `q <= kv + SLIDING_WINDOW_SIZE` 的 (q, kv) 对,因此所有整体位于 `cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE` 之下的 tile 全被 mask、贡献为零。这些 tile 虽不读 K/V、不跑 dot(SKIP_TILE 已守护),但每个 tile 仍要付出"证明自己为空"的成本:构建 `final_mask`(causal 比较、窗口比较、`&`)以及计算 `SKIP_TILE` 的 `[BLOCK_M, BLOCK_N]` 跨 wave `tl.max` 归约。在 extend_len=8192、W=127 时,最后一个 M block 的下界约为 8192−127,几乎每轮迭代都耗在证明 tile 为空上。

## 优化点

把 KV 循环起点从 0 改为包含窗口下界的 BLOCK_N 对齐 tile:`extend_start = (max(cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE, 0) // BLOCK_N) * BLOCK_N`。该下界直接由 mask 谓词推导(而非估算):kernel 在 `cur_block_m * BLOCK_M >= cur_seq_len_extend` 时提前返回,块内最小 query index 恰为 `m0 = cur_block_m * BLOCK_M`;被跳过 tile 满足 `start_n + BLOCK_N - 1 < m0 - W`,窗口 mask 对其全部元素为假,且在 tile 粒度上是紧的——前一 tile 需 `floor_aligned >= m0 - W + 1` 才可能持有存活元素,与 `floor_aligned <= m0 - W` 矛盾,故任何 BLOCK_M/BLOCK_N 下都不会漏算。`SLIDING_WINDOW_SIZE` 为 `tl.constexpr`,全注意力层在 trace 时消除该分支、循环不变;`extend_start` 是 BLOCK_N 的倍数,`tl.multiple_of` 约束保持有效;custom mask/非 causal 路径只增加合取项,下界仍然成立。

## 关键改动

```text
python/sglang/kernels/ops/attention/extend_attention.py: _fwd_kernel stage-2 KV 循环
  下界改为窗口 floor 对齐 tile
test/registered/attention/unittests/swa/test_triton.py: 新增 3 个 extend 长度超过
  BLOCK_M 且窗口小于它的用例(无 prefix / prefix + 非 tile 整除窗口 / 窗口恰在
  tile 边界),守护下界的正确触发
```

关键摘录(diff):

```python
extend_start = 0
if SLIDING_WINDOW_SIZE > 0:
    extend_start = (
        tl.maximum(cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE, 0) // BLOCK_N
    ) * BLOCK_N
for start_n in range(extend_start, extend_end, BLOCK_N):
```

## 性能声明

Kernel(MI355X,do_bench median,7 次取最小;baseline 前后复测漂移最大 0.30%):N_CTX=8192 时 W=127/256/512 分别 −39.0%/−33.5%/−24.5%;4096 时 −27.3%/−20.9%/−12.6%;2048 时 −15.2%/−9.6%/−3.5%;1024 时 −6.2%/−2.6%/−1.0%。全注意力对照(sliding_window_size=-1,分支被 constexpr 消除)漂移内(−0.1%..+0.2%);下界不触发的配置(W>=N_CTX/2)无回退(−1.0%..−1.7%)。

端到端 prefill(gemma-4-26B-A4B-it,bench_one_batch,单张 MI355X):bs=1 input 32768 时 −11.3%(1.5027→1.3333 s),16384 时 −8.1%;serving(bench_serving,random-ids,两台 MI355X 互换取均值):median TTFT −2.9%..−7.5%,input throughput +1.7%..+7.8%。

## 验证与回归风险

单元测试:swa/dense test_triton.py 15 passed、2 skipped、58 subtests passed。kernel 输出 bitwise 一致:21 种配置(12 SWA、4 全注意力、5 下界不触发),B=32/H_Q=64/H_KV=8/D=128 bf16 causal,输入与 `o_extend` 同时哈希比对,21/21 输出 bitwise identical。E2E greedy parity 32/32 输出一致;GSM8K 1319 题两轮:0.954/0.956 vs main 0.960/0.959,差距在运行间噪声内(每题 std 0.20)。回归风险:改动不限定 ROCm,CUDA 上的 Triton 后端同样生效(正确性由 CI 单测覆盖);作者报告 SGPR spill 76→80,未见测量影响但如实标注;受影响默认路径包括 Gemma 4(25/30 层为 SWA 层)、OLMo 3、EXAONE 4.0 等及所有 `--attention-backend triton` 的 SWA 模型。

## 关联

无(pack 内未提及具体关联 PR;属 Triton attention kernel 优化方向)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
