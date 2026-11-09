# PR #44632: [Perf] DeepSeek-V4.1: Raise the SM100 swap-AB decode cap to 40 rows

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @weireweire |
| 合并 | 2026-10-20 |
| 分类 | Speculative Decoding(major) |
| Label | jit-kernel |
| Diff | diffstat 缺失(素材未提供统计);diff 实际涉及 2 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/44632 |

## 一句话结论

SM100 swap-AB decode 的行数上限 8→40,让 TP4 下 DSpark verify bs≤8 走 swap-AB,对 FlashMLA 最快 2.7x。

## 动机

SM100 的 swap-AB decode attention 此前最多接受 8 个 query row(`MAX_BATCH = 8`)。DSpark verify 每请求发 5 行(block size 5),因此只有 batch size 1 能走 swap-AB,bs≥2 全部回退 FlashMLA;而 FlashMLA 会把 q padding 到 64 头并跑 split-KV,时间在 5-40 行区间几乎恒定(约 24 µs),swap-AB 在 5 行时快 2.7x 且随行数增长——两者 break-even 约在 48 行,8 行的门槛把大量本该赢的场景挡在了外面。

## 优化点

`decode_attention_sm100.py` 引入 `MAX_ROWS = 40` 替代对 `MAX_BATCH` 的判定:`can_use_swapab_attention` 与 kernel 内断言均改为 40 行上限,使 TP4 verify 到 batch size 8 都能分派到 swap-AB。kernel grid 是 (rows, KV tiles)、没有 program 依赖行数,该上限纯粹是 dispatch 选择。gfx950(AMD)路径保持 8 行 `MAX_BATCH` 不变。测试新增 20 行、40 行用例与 40 行 CUDA graph replay,per-row 长度模式改为重复 5 次以覆盖任意 batch。

## 关键改动

```text
python/sglang/kernels/ops/attention/dsv4/decode_attention_sm100.py: MAX_ROWS=40 门槛 + 断言更新
test/registered/kernels/ops/attention/dsv4/test_swapab_attention.py: 20/40 行用例 + 40 行 graph replay
```

```python
# SM100 row cap, below the ~48-row break-even against FlashMLA.
MAX_ROWS = 40
...
return (
    0 < q.shape[0] <= MAX_ROWS
    and num_heads in (NUM_HEADS, 2 * NUM_HEADS)
    and q.dtype == torch.bfloat16
    and q.shape[-1] == HEAD_DIM
)
```

## 性能声明

Kernel 计时(GB300,16 heads/TP4,单 decode layer shape:SWA 128 + extra KV 512,CUDA graph 计时,µs/次):

| rows | verify batch | swap-AB | FlashMLA | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 1 | 8.8 | 23.7 | 0.37 |
| 10 | 2 | 9.3 | 23.9 | 0.39 |
| 20 | 4 | 12.5 | 23.9 | 0.52 |
| 40 | 8 | 20.1 | 24.2 | 0.83 |

E2E:AgentX,DeepSeek-V4.1-Flash FP4,GB300,TP4/EP1,每 run 一小时,DSpark 模拟接受长度 3.51:concurrency 8 时 P90 tok/s/user 413.3→411.3(-0.5%,中位数 577→610,+5.7%);concurrency 16 时 P90 422.0→442.4(+4.8%),中位数 538→559(+4.0%)。

## 验证与回归风险

GSM8K 全量 1319 题,5-shot,真实 MTP verification(无模拟接受),64 并发,TP4 GB300:main 0.9696 → 本 PR 0.9711。`test_swapab_attention.py` 在 GB300 通过(新增 20/40 行与 graph replay 用例,gfx950 上 skip)。回归风险:上限贴近 48 行 break-even,若 FlashMLA 或 swap-AB kernel 的时间特性变化,40-48 行区间可能变为负收益;判定仍是纯 dispatch 阈值,gfx950 保持旧行为;CUDA graph 下 40 行 replay 已被测试覆盖。

## 关联

同区域:DSpark 投机解码 verify 路径(关联 PR 编号正文未提及,写无)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
