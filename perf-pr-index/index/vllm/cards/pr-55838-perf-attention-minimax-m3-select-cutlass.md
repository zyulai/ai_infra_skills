# PR #55838: [Perf][Attention][MiniMax-M3] Select CUTLASS MSA by EAGLE3 decode shape

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @anish-shanbhag |
| 合并 | 2026-10-13 |
| 分类 | Speculative Decoding (tier: major) |
| Label | ready, nvidia, minimax |
| Diff | diffstat 缺失(pack 未提供加减行数);diff 实际改动 3 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/55838 |

## 一句话结论

MiniMax-M3 CUTLASS MSA decode 门控从固定 batch>=16 改为按 SM103 实测 crossover(batch*qlen*heads>=1024),选中区最小直接提速 1.221x。

## 动机

MiniMax-M3 的 CUTLASS MSA(sparse attention)decode 路径此前用固定 `batch_size >= 16` 门控。常见 EAGLE3 verify step(小 batch、每 token 多 query)即使 CUTLASS 更快也会回落 Triton。但一刀切降低门槛不安全:tensor parallelism 会改变 rank-local 的 query/KV head 几何,从而改变 crossover 点——需要按 query length、device capability 与 head geometry 实测定门。

## 优化点

在模型本地(msa_cutlass_sparse_decode.py)实现基于实测的保守 dispatch 规则:当 SM103(device capability (10,3))、decode query length 1-8、且 rank-local query/KV head 几何属于实测集合 {(64,4),(32,2),(16,1)} 时,门控条件改为 `batch_size * decode_query_len * num_q_heads >= 1024`(即 `_min_cutlass_batch_size` 用 ceil_div(1024, qlen*heads) 计算);query length 9-32、SM100、未知 head 几何保留原 batch-16 floor。所有既有 kernel 约束(显式 CUTLASS 选择、FP8 E4M3 KV cache、head geometry、page size 128、top-k 16、query length 上限、planner row bound)不变。部署方可用模型本地 env `VLLM_MINIMAX_M3_MSA_CUTLASS_MIN_BATCH` 覆盖自动 crossover(unset 走自动,设 16 恢复旧行为),override 在 metadata builder 初始化时解析一次。实测依据:768 行边界在旧镜像上有 win 但与当前 vLLM build 未对齐,故采用保守的 1024 行规则。

## 关键改动

```text
vllm/models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py: 新增 _min_cutlass_batch_size(SM103+EAGLE3 qlen1-8+实测 head 几何时按 1024 query-head-rows 计算),should_prepare_decode_metadata 接受 min_batch_size
vllm/models/minimax_m3/nvidia/sparse_attention_msa.py: builder 读取 VLLM_MINIMAX_M3_MSA_CUTLASS_MIN_BATCH 覆盖
tests/kernels/attention/test_minimax_m3_msa_cutlass_sparse_decode.py: dispatch 边界用例扩展(qlen 1-8 行阈值、unmeasured 形状、SM100)
```

关键代码摘录(节选,<=15 行):

```python
if (not current_platform.is_device_capability((10, 3))
        or not 1 <= decode_query_len <= _B300_EAGLE3_MAX_QUERY_LEN
        or (num_q_heads, num_kv_heads) not in _B300_EAGLE3_HEAD_GEOMETRIES):
    return _DEFAULT_MIN_CUTLASS_BATCH_SIZE
# Use CUTLASS once batch_size * decode_query_len * num_q_heads >= 1024.
query_head_rows = decode_query_len * num_q_heads
return (_B300_EAGLE3_QUERY_HEAD_ROWS + query_head_rows - 1) // query_head_rows
```

## 性能声明

GB300 隔离微基准(EAGLE3,CUDA graph replay,30 warmups/300 iters/5 trials,FP8 E4M3 KV cache,page top-k 16,8K 与 200K cached context):
- qlen=4 各 head 几何的首个测得 win(保守口径,含独立 BF16→FP8 query 量化开销):64/4 heads batch 4/8/16 为 1.069x/1.545x/1.875x,首 win 在 batch 4;32/2 为 0.800x/1.100x/1.642x,首 win batch 8;16/1 首 win batch 16(1.099x)。
- qlen 1-8 扫描(1,344 个 rank 级对比,全部通过 atol=rtol=0.02,最大绝对误差 0.00430298):得到各 qlen/TP 的安全 crossover 表(如 qlen=4:TP1 batch 3、TP2 batch 6、TP4 batch 12)。
- 选中区(318 cases)最小直接 kernel 提速 1.221x,含量化的最小提速 1.098x;恰在 1024 行的 24 cases 为 1.221-1.223x direct / 1.098-1.101x 含量化。
- 1024 行规则在 576 个新聚合 case 中零误选(无 CUTLASS 回归);旧镜像上有 61 个 direct-kernel win 未被选中(有意保守)。
- 参考性:MiniMax-M3-NVFP4 disaggregated EAGLE3 下,早期 blanket batch-4 实验约 +3-5% p90 per-user output-token rate、+1-2.5% total TPS/GPU(该实验只证明机会,不对应最终策略)。

## 验证与回归风险

- 验证:test_minimax_m3_msa_cutlass_sparse_decode.py 静态 dispatch 用例(monkeypatch SM103/SM100)与 GB300 correctness/benchmark(41 passed kernel suite、9 passed env suite);pre-commit(Ruff/mypy/tethering)通过。
- 回归风险面:仅 MiniMax-M3 NVIDIA sparse-attention decode dispatch,不改共享 scheduler/model-runner/KV-cache 接口。未实测形状(SM100、qlen 9-32、未知 head 几何)保留 batch-16 floor;NVFP4 KV cache 仍无条件走 CUTLASS(无 Triton fallback)。768 行更激进规则留待同 build 复测后再启用。

## 关联

- 同区域:PR #60684(MiniMax-M3 MSA prefill query 量化,与本 PR 不重叠,PR 正文声明);索引库已有 pr-45381(MiniMax-M3 支持)、pr-56151(MSA indexer decode)、pr-60684、pr-62712(M3 monokernel decode)等卡片。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
