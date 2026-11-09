# PR #41488: [AMD] Add opt-in MiniMax-M3 TP4 indexer context partitioning

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ThomasNing |
| 合并 | 2026-10-02 |
| 分类 | Attention / Kernel (tier: major) |
| Label | documentation, run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/41488 |

## 一句话结论

MiniMax-M3 TP4 decode 下把 index-K 的块读取按 4 rank 分片,batch 32/128K 上下文 indexer 链延迟 109.84 → 41.63 µs(2.64x)。

## 动机

TP4 下 MiniMax-M3 的四个 index-query head 分属各 rank,而 index-K cache 是复制的;每个 rank 都要为自己的 head 扫描完整 index cache。本 PR 增加 opt-in decode 路径:把块读取按 4 个 rank 分片,并在每个 K tile 驻留时对全部 head 打分,从而减少读取流量(cache 仍复制,减的是读流量而非 cache 容量)。

## 优化点

新增 `SGLANG_MINIMAX_M3_INDEXER_CP`(默认 0),仅适用于 gfx950 TP4 普通 decode、attention CP/DP size 1、4 index heads、4 main KV heads、dim 128、128-token blocks、max scoring、top-k 16;不满足条件的运行配置记录原因并保留原路径。实现:先经 attention TP communicator 聚合现有 post-RoPE query,rank r 读块 `r, r+4, r+8, ...` 并为所有 head 打分;Triton kernel 做分片打分、每 head 本地 top-16 选择与最终合并(每分片保留 16 个候选,即使赢家集中在单一分片也保证全局 top-16 正确);初始/近期块优先级与 ROCm 原生 score/ID 排序保持一致。打分后的 packed score/ID 键经 communicator 交换(支持时用 AITER custom all-gather,否则回退),键用 FP32 bitcast 视图纯字节复制、无数值转换;graph capture 前预热两条 collective 路径。初始范围不含 speculation、TBO、HiSparse、FP8 query、dense sparse decode;无自动 batch/context 切换策略。

## 关键改动

```text
python/sglang/kernels/ops/attention/minimax_sparse/decode/indexer_cp.py: 分片打分/本地 top-16/合并的 Triton kernel
python/sglang/srt/layers/attention/minimax_sparse_ops/indexer_cp.py: 特性门控(unsupported_reason)、warmup、Q 聚合与候选交换
python/sglang/srt/layers/attention/minimax_sparse_ops/minimax_sparse.py: decode 路径插入 CP 分支
python/sglang/srt/environ.py: 注册 SGLANG_MINIMAX_M3_INDEXER_CP(EnvBool(False))
test/registered/amd/test_minimax_indexer_cp.py 等: 选择一致性测试与门控测试
```

## 性能声明

微基准(4x MI355X,BF16 query,FP8 E4M3 index cache,AITER custom gather;每次 HIP graph 8 次 indexer 调用、每轮 100 次 replay、7 轮 baseline/CP 交替,取 7 样本中位,每样本取最慢 rank;输入复用且预热、无 L2 flush;仅 indexer 链含通信,不含 projection 与 main attention,不能解读为模型吞吐收益):batch 32/128K 上下文延迟 109.84 → 41.63 µs(**-62.1%,2.64x speedup**);128K/batch 8 -26.9%、batch 16 -38.0%;单条 128K 行其余 1K 时 batch 16 -31.7%、batch 32 -49.7%。8K 全部 shape 与 128K 的 batch 1-2 回归(如 8K/batch 8 +137.3%,128K/batch 1 +25.4%)。32K/batch 16 的 -3.7% 需独立复测。全模型 serving 收益未测量。

## 验证与回归风险

4 张 MI355X 上 8/8 用例全 rank 通过(BF16 与 FP8 E4M3 cache x 随机分/平分/单分片赢家/混合长度),对原生 SGLang 与独立 FP32 参考做精确 selected-ID 对比,graph capture 后变长 replay 一致;17 个计时 shape 同时通过 eager 与 graph replay 的 ID parity。新增 `test_minimax_indexer_cp.py`(分片选择与原生 selector 逐位一致)与 `test_minimax_indexer_cp_gate.py`(门控/chain-draft allowlist)。作者自列未完成项:经加载 server 的运行时门控验证、全模型精度、serving TPS/TTFT/TPOT、fallback collective 下的测量;shape gate 接受 FP16 query 与其他 FP8 cache 变体但未覆盖。风险:短上下文/小 batch 明确回归,故默认关闭;CI Extra 与 AMD ROCm 10 未通过。

## 关联

同区域为 MiniMax-M3 sparse indexer(ROCm radix top-k)路径;正文未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
