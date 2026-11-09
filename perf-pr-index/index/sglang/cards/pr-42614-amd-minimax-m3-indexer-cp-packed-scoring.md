# PR #42614: [AMD] MiniMax-M3 indexer CP: packed scoring for EAGLE verify rows

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kevin-mii |
| 合并 | 2026-10-09 |
| 分类 | Speculative Decoding (major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 含 5 个文件:packed Triton kernel、backend 接线与 AMD 测试 |
| 链接 | https://github.com/sgl-project/sglang/pull/42614 |

## 一句话结论

把 EAGLE verify 的逐 draft 行打分改为按请求打包进 16 行 tile,main 上吞吐 +4.6%、ITL p50 -8.6%。

## 动机

#41488 合入后,MiniMax-M3 indexer context partitioning 开始服务 EAGLE chain verify:ROCm verify funnel 把每个请求的 `ndt` 条 draft 行当作普通 decode 查询交给 indexer,CP scorer 再逐行打分。其 16 行 tile 每个程序只有 `WORLD`=4 个活跃行(每 lane 一个 head),占用率仅 25%,且每个 index-K block 对每条 draft 行都要重读一次——默认 4 个 draft token 下同一请求的 K 被读 4 遍。

## 优化点

新增 `_score_shard_packed` Triton kernel:`score_local_blocks` 接受 `packed_queries` 参数,当一批请求的行能放进 tile 时启动 packed 版本,把 tile 行 i 映射为(draft 行 `i // WORLD`,head `i % WORLD`),从而填满 tile、每请求只读一遍 K block。行组可能跨越 block 边界,因此 K 读取分支用组内最长行决定,masking 与强制 init/local block 则按各行自己的长度;装不下 tile 的 pack 回退现有 kernel;与 `_score_shard` 一样跳过超出请求表(graph padding)槽位的 K 读取。接线约 10 行:`minimax_sparse_decode` 与 `MiniMaxIndexerCP` 透传 `packed_queries`,backend 在 target-verify batch 传入 draft token 数、否则传 1;只有 CP 分支读它。packed kernel 保持独立:若把普通路径合并为 `PACK=1` 虽可精确复现,但向量行寻址在普通 decode 的 30/30 个 shape 上实测慢 3.6%。

## 关键改动

```text
python/sglang/kernels/ops/attention/minimax_sparse/decode/indexer_cp.py: 新增 _score_shard_packed 与 TILE_ROWS=16,score_local_blocks 增加 packed_queries 分支
python/sglang/srt/layers/attention/minimax_sparse_backend.py: target-verify 时传 speculative_num_draft_tokens(约 10 行接线)
python/sglang/srt/layers/attention/minimax_sparse_ops/minimax_sparse.py 与 indexer_cp.py: 透传 packed_queries
test/registered/amd/test_minimax_indexer_cp.py: 新增 packed verify 行 vs native selector 的逐 ID 相等测试,slot 表改为乱序带间隔
```

tile 映射核心:

```python
# tile row i is (draft row i // WORLD, head i % WORLD): one K read per request
group, chunk = tl.program_id(0), tl.program_id(1)
tile = tl.arange(0, TILE)
head, sub = tile % WORLD, tile // WORLD
in_tile = tile < PACK * WORLD
rows = group * PACK + sub
```

## 性能声明

测试条件:MiniMax-M3 MXFP4(`amd/MiniMax-M3-MXFP4`),TP4 on MI350X,EAGLE3(`Inferact/MiniMax-M3-EAGLE3-GQA`,3 步 / 4 draft tokens,真实 acceptance),`SGLANG_MINIMAX_M3_INDEXER_CP=1`、`SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=1`,AgentX agentic traces(SemiAnalysis AIPerf,seed 42),每点 900 s,双臂同节点两半 GPU、相同 flags。

- 开发分支 A/B(c=24:26,576→**28,380** tok/s,**+6.8%**,ITL p50 12.4→10.2 ms;c=32:32,294→**34,282** tok/s,**+6.2%**,ITL p50 17.1→12.7 ms)。
- main 上 A-B-B-A(单 TP4 replica,c=24,每臂 900 s):main 16,912/16,983 tok/s → main+PR **17,728/17,731** tok/s,**+4.6% 吞吐、−8.6% ITL p50**(30.8/31.0→28.5/28.0 ms);重复臂一致性 0.4% 内。20 步 decode profile 显示只剩 `_score_shard_packed`(每 rank 每步 57 次launch,每个 sparse 层一次)、无 `_score_shard`。main 上增益小于开发分支,因为 main 的其他每步开销更大。
- 准确率:GSM8K-500 5-shot 0.896(packed)vs 0.892(unpacked);主 A/B 上 0.864/0.866 vs 0.884/0.886,该 few-shot gate 在同 build 不同 server 实例间波动约 ±3.5 分,应读作无变化。
- 两臂均 `--cuda-graph-backend-prefill disabled`(main 在 breakable prefill graphs 上首个 M3 prefill 会崩,#41845 已修),本 PR 改动的 decode/verify 路径不受影响。

## 验证与回归风险

新增 `test_packed_verify_rows_match_the_native_selector`:按 funnel 布局构造 4 请求 × 4 draft 行(结尾 32767..32770,单 tile 混合 256/257-block 行),bf16 与 fp8 下要求与 native selector 逐 block ID 精确相等;能抓住五类注入 bug。共享 `_inputs` 改为乱序带间隔 slot,使按组号而非 slot 表寻址的 packed kernel 失败(此前会通过)。`test_minimax_indexer_cp.py` 在 gfx950(MI355X)2 passed、6 subtests;测试移入 MI35x suite 且非 gfx950 跳过(参考 kernel 需要 MI300 没有的共享内存,CP 路径本身 gfx950-only)。CI 状态:Base/Extra/AMD ROCm 10 均失败。风险面:仅 CP 分支的 score-only 层受影响;`batch % packed_queries != 0` 或超过 TILE_ROWS 时回退原 kernel。

## 关联

前序:#41488(EAGLE chain verify 接入 indexer CP,本 PR 的直接前提)、#41845(breakable prefill graphs 崩溃修复)。同区域:MiniMax-M3 sparse indexer 优化系列(pack 未列其他编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
