# PR #36557: MiniMax-M3: Triton split-K router GEMV with in-kernel fixup

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zcnrex |
| 合并 | 2026-09-08 |
| 分类 | MoE(major) |
| Label | amd, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 显示新增 router_gemv.py(175 行)+ minimax_m3.py 调用点接入 |
| 链接 | https://github.com/sgl-project/sglang/pull/36557 |

## 一句话结论

为 MiniMax-M3 的 skinny MoE router GEMV 新增 Triton split-K kernel,gfx950 上权重读取接近带宽上限,端到端输出吞吐 +1.07%..+3.27%。

## 动机

MiniMax-M3 的 MoE router gate 是一个 skinny GEMV:`[M, K]` bf16 x `[N, K]` bf16 → `[M, N]` fp32,其中约 N=128 个 experts、decode 时 M<=64。厂商 BLAS 对这种形状的求解远低于 memory roofline:在 gfx950 上,M<=8、N=128 时读取约 1.6MB 权重只跑到约 0.1 TB/s(约 12us)。router GEMV 每 decode step 约 57 次 launch,是值得优化的高频小 kernel。

## 优化点

新增 `python/sglang/kernels/ops/gemm/router_gemv.py`:Triton split-K kernel,对 gate 权重以接近带宽上限的方式单次读取。配置按 M bucket 选择(BLOCK_K 全程 512,BLOCK_N 与 SPLIT_K 随 batch size 变化),覆盖 M<=8/16/32/64;`BLOCK_M==1` 走专门的 GEMV 路径,用 `tl.sum` 替代 `tl.dot`。split-K 归约在同一 kernel 内完成:对每个 `(n-block, row-tile)` 对,最后完成的 CTA 通过原子计数器判定身份,归约全部 SPLIT_K 份部分和并把计数器槽位清零,使 scratch buffer 无需 zero-init launch 或独立 reduce launch 即可跨 launch 复用——在 gfx950 上每次额外 launch 约 2us,在这个尺寸下占 kernel 本身相当比例。形状不满足条件(错误 dtype、非连续 last dim、N 不整除 BLOCK_N、K 不整除 BLOCK_K、M>64)时经 `router_gemv_supported()` 回落原有 `torch.mm` 路径。

## 关键改动

```text
python/sglang/kernels/ops/gemm/router_gemv.py: 新增 split-K router GEMV kernel + 支持
  判定 + per-device scratch 缓存(partials/counter)
python/sglang/srt/models/minimax_m3.py: _compute_router_logits 在 bf16 router gemm
  路径上接入 router_gemv(满足条件时)
```

关键摘录(in-kernel fixup):

```python
count = tl.atomic_add(
    counter_ptr + pid_n * tl.num_programs(1) + pid_m, 1, sem="acq_rel"
)
if count == SPLIT_K - 1:
    total = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for s in range(SPLIT_K):
        total += tl.load(partials_ptr + ...)
    tl.store(out_ptr + ..., total, mask=(offs_m < M)[:, None])
    tl.atomic_xchg(counter_ptr + ..., 0)
```

## 性能声明

8x MI350X(gfx950),MiniMax-M3-MXFP8,TP8,80k 输入 / 600 输出,cache hit rate 90%,baseline 为 upstream/main,每 boot 跑两轮取 warm rep,噪声区间 ±0.4%:输出吞吐 +1.07%..+3.27%(如 bs=1:55.35→57.16 tok/s,bs=32:934.92→947.49 tok/s);输入吞吐 +0.33%..+0.99%(如 bs=1:187984→189401 tok/s)。精度:gsm8k 512 例 greedy,score 97.27% vs baseline 97.07%(+0.20pp,作者注明按"无可测精度损失"解读而非提升),error_rate 两侧均 0.00%。

## 验证与回归风险

结果不是 bit-identical:split-K 重排了 fp32 累加顺序,router logits 与 `torch.mm` 路径在末位不同,因此以精度 gate 而非位一致性为准。回归风险面较宽:`router_gemv_supported()` 只检查 `x.device.type == "cuda"`(ROCm torch 构建同样为 true),即该路径在 NVIDIA 上也默认开启,而非 opt-in ROCm 选项(作者在正文中明确提请 reviewer 按此评估);不满足条件的形状回落 `torch.mm` 不变。另注意:PR 复现命令中 `--moe-runner-backend aiter` 在当时 main 上未被采纳、实际回落 triton MoE runner,两侧 A/B 一致故对比成立,但被测的并非 aiter MoE 路径。

## 关联

同区域:同 AMD MiniMax 系列优化的 #23620、#23611;属于 MoE router/gate 优化方向。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
