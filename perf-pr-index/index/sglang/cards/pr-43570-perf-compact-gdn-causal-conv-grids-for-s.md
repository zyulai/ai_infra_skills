# PR #43570: [Perf] Compact GDN causal-conv grids for skewed prefill batches

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-10-13 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/43570 |

## 一句话结论

GDN prefill causal-conv 改用紧凑 (sequence, chunk) 表启动,偏斜 batch 下 e2e 输出吞吐 +3.6%(C64)/约 +9%(C128)。

## 动机

GDN 的 prefill 卷积按最长序列启动矩形网格:8K prefill chunk 配 127 行 decode 时会调度约 310 万个 CTA,其中绝大多数立即返回;同一开销在 Qwen3.5-122B-A10B 的 36 个 GDN 层上每层重复。

## 优化点

每次 forward 构建一次紧凑的 `(sequence, chunk)` 表(每序列按 `CAUSAL_CONV1D_FWD_BLOCK_M` 切块的非空块列表),pin-memory 构建、non-blocking H2D,复用既有 causal-conv kernel 的 `block_table` 路径,把上例降到 27,264 个 CTA,不改变卷积算术与状态更新。启用条件保守:仅 CUDA 的 EXTEND/MIXED batch,且矩形网格的 sequence/chunk 块数超过紧凑网格 4 倍(即 75% 以上块空转)时才启用,均匀与轻度参差 batch 保持原 launch 以摊销约 280 µs/forward 的建表与传输开销。MIS、TBO、speculative 模式、非 CUDA 路径与 host 长度不匹配时保留原行为。该改动比 #36065 的 prefill/decode split 更窄:只压缩卷积网格,不改 GDN attention dispatch。

## 关键改动

```text
python/sglang/srt/layers/attention/linear/gdn_backend.py: _init_prefill_conv_block_table 建 4x crossover 的紧凑表,_convolve_prefill 传 block_table
python/sglang/srt/layers/attention/mamba/mamba2_metadata.py: ForwardMetadata 增加 conv_block_table 字段
test/registered/attention/test_gdn_prefill_layout.py 等 3 个测试文件: 紧凑输出/state 逐位一致与回退用例
```

```python
if not block_counts or len(block_counts) * max(block_counts) <= 4 * sum(block_counts):
    return  # uniform / mildly ragged batches keep the rectangular launch
metadata.conv_block_table = torch.tensor(
    pairs, dtype=torch.int32, device="cpu", pin_memory=is_pin_memory_available()
).to(self.device, non_blocking=True)
```

## 性能声明

条件:2x H200 TP2,Qwen3.5-122B-A10B-FP8(revision `a099dee7`),baseline `e35f60b9` 仅加本补丁;标称 8192-token 输入、固定 1024 输出、seed 42、C64/C320 请求;FlashInfer full attention、Triton GDN prefill、BF16 Mamba state、`--enable-mixed-chunk`、`--disable-radix-cache`。

- C64 输出吞吐 2126.54 → 2204.10 tok/s(**+3.6%**);C128 第一对 2470.12 → 2686.89(+8.8%),反序对 2475.35 → 2698.68(+9.0%);C128 median TPOT 50.92 → 46.36 ms,p99 E2E 79.48 → 72.63 s。
- 换 `--prefill-attention-backend fa3` 的独立一对:C64 2193.74 → 2275.06,C128 2564.33 → 2797.15(+9.1%)。
- 隔离卷积(`[8065]+[1]x127`,dim 6144,width 4):H200 1873.5 → 104.0 µs,GB300 1568.4 → 86.7 µs(约 18x kernel 加速,非 e2e);建表+传输约 280 µs/forward,跨层摊销。
- TP2 trace:8 个 mixed step、288 次卷积 launch,rank0 卷积 GPU 时间 525.87 → 29.26 ms,rank1 525.88 → 29.79 ms。

## 验证与回归风险

扩展 3 个既有测试文件(H200 与 GB300 上 29 tests + 55 subtests 通过);CUDA 测试对卷积输出与整个 state cache 逐位比较(含 strided QKV、ragged/空行、初始 state、乱序 cache index、padded slot);微基准在模型实际卷积维度 6144 下同样逐位一致。全量 held-out GSM8K 1,314 例(5 个 few-shot、temp 0、seed 42、C64):Baseline A/B 1270/1276,Compact A/B 1273/1273,逐例差异已在正文披露,无"两 baseline 均对而两 candidate 均错"的用例;默认 cached-path(关闭 mixed chunk)回归检查 C128 2693.96 → 2699.97 tok/s,GSM8K 两跑均 1277/1314。正文明确这些测量不构成"全模型输出一致"或统计非劣性结论。CI Extra 与 AMD ROCm 10 未通过。

## 关联

比 #36065 的 GDN prefill/decode split 更窄(仅压缩卷积网格);正文未列其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
