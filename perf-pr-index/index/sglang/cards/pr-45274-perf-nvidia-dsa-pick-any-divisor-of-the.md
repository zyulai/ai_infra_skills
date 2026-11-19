# PR #45274: [Perf] [NVIDIA] DSA: Pick any divisor of the tile count for the TileLang sparse decode split

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JackChuang (co-authored-by @HanHan009527) |
| 合并 | 2026-11-19 |
| 分类 | Speculative Decoding(major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/45274 |

## 一句话结论

让 CUDA TileLang FP8 稀疏 decode 的 partial+combine 切分可取 tile 数的任意因子而不仅是减半:GLM-5.3-Flash 的 2051 宽 top-k 补到 66 tiles 后,attention DP8 + MTP 1/1/2 在 128 并发从每 query 33 组改为 6 组,稀疏 decode attention 内核 GPU 时间省 27%(1.36×),端到端稳态 decode 吞吐最高 +4.3%。

## 动机

更早的 DSA 模型每 query 选 2048 个 KV token,GLM-5.3-Flash 选 2051(indexer 追加至多 3 个未池化尾部 KPool token),`_forward_tilelang` 补到 64 的倍数 2112:66 个 32 宽 tile,不再是 2 的幂。#42957 的 `tilelang_sparse_fwd` partial+combine 路径把每 query 的 `ni` 个 top-k tile 按 `ni / inner_iter` 分组,`_pick_inner_iter` 只试 `ni, ni//2, ni//4, ...`——64 tiles 时全是因子,66 减半到 33 后只有 33 和 2 整除,永远试不到 3、6、11、22。搜索还要求每 SM 至少一个 block(132-SM GPU 上限约 `seq/2` tiles/组),于是 4–65 行时落到 2 tiles/组:33 个部分结果逐个写出再被 combine 读回,而 11 tiles/组(6 组)在 32 行时同样满足限制。

## 优化点

- `python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py`:`_pick_inner_iter` 增加 `any_divisor` 与 `head_blocks` 参数——在原 pick 之上尝试更大的 `ni` 因子,并做 fill 检查(其 block 数 `seq × head_blocks × ni / inner_iter` 须填满每个 wave 至少 80% 的每 SM 双 slot),取通过检查的最大因子。仅 CUDA FP8 调用点开启;ROCm(FP8/BF16 及 DeepSeek-V4 的 `dpsk_v4_fp8_attention_fwd`)保持默认搜索。
- 幂次 tile 数(top-k 2048,DeepSeek-V3.2 与 GLM-5 至 5.2)与 ROCm 的切分完全不变;无新 flag、无内核改动(内核只要求 `ni % inner_iter == 0`,main 本就运行非 2 幂切分)。
- 新增单元测试 `test/registered/unit/layers/attention/test_tilelang_dsa_decode_split.py`(`base-a-test-cpu`),钉住 132 SM 下 66 tiles 的选取、幂次宽度不变、默认搜索不变;去掉新搜索任一条件都会挂 case。

## 关键改动

```python
def _pick_inner_iter(seq, ni, cu, block_per_cu, any_divisor=False, head_blocks=1):
    ...
    if any_divisor:
        # Two blocks of the FP8 partial kernel run on each SM; fewer, or a mostly
        # empty last wave, was slower on H200 with 8 to 64 heads.
        slots = 2 * cu
        for larger in range(min(ni, max_it), it, -1):
            blocks = seq * head_blocks * (ni // larger)
            if ni % larger == 0 and blocks >= 0.8 * slots * -(-blocks // slots):
                return larger
    return it
```

调用点传 `any_divisor=not _is_hip`、`head_blocks=(num_heads + 15) // 16`。

## 性能声明

(全部照抄 PR 正文,未独立核实;H200/GLM-5.3-Flash/FP8 KV,LongBench-v2 约 8.2k prompt ×1024 输出)

- attention DP8 稳态 decode:128 并发 +2.5%(9,638→9,880 tok/s)、64 并发 +1.2%;MTP 1/1/2 在 256 并发吞吐 +4.3%(14,733→15,370)、TPOT 17.39→16.66 ms(−4.2%)、step time −4.0%;MTP 5/1/6 step time 64 并发 −3.3%、32 并发 −2.0%;单请求 step time 不变或 −0.7%;TTFT 与 main 差 1% 内。TP8 低延迟布局 2–8 并发 step time −0.7% 至 −1.2%。
- profile(128 并发):partial 168.7→144.4 µs(grid [128,33]→[128,6]),combine 32.9→3.4 µs,合计 201.6→147.8 µs(27% less,1.36×);64 并发 MTP 5/1/6 处 1.39×。
- 每 kernel 调用:改变切分的行数上 8–27% 更省、从不更慢(64 heads: +11.7% 至 +26.5% 等 5 档;另 89 个 case 7.8–26.9%)。H20 每 call 16–20% 更省,端到端 mean TPOT −1.1%(64 并发)/−1.3%(128 并发)。
- 成本:空 TileLang 缓存首次启动一次性约 +35 s(346→380 s),缓存已填时与 main 相同(145–146 vs 140–146 s)。
- 精度不变:GSM8K 97.04% vs main 97.35%(一个标准误 0.44 点内),accept length 相同;新切分的内核误差仍低于 main 66+ 行时本就使用的 33/66-tile 切分。

## 验证与回归风险

单元测试为纯 CPU 逻辑测试(4 种设备形状 × 幂次宽度全枚举);GPU 上 `test_tilelang_dsa_zero_rope.py` 跑了改动切分。PR 自述 CPU 与单卡套件中无测试仅因本 PR 失败;多卡、Blackwell、ROCm 套件未跑。风险面:fill 检查阈值 0.8 若移除会在 8 heads × 44 rows 出现 132-block 近空 wave(实测更慢),阈值 75%–100% 区间均安全;ROCm 路径未启用新搜索,行为不变。

## 关联

#42957(TileLang DSA FP8 backend)、#44262/#46424(attention DP + MTP 的 padded-extend 修复,测量时双臂均带)、#32890/#44612(同作者的 KDA verify 系列)。H20 数据见 @HanHan009527 的 PR 评论。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
