# PR #36596: [Perf] add packed recurrent fast path for decode

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @caozuoba |
| 合并 | 2026-03-12 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | ready, qwen |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾无 "file(s) dropped from diff" 说明,diff 完整保留 5 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/36596 |

## 一句话结论

为 Qwen3Next GDN decode 加 packed recurrent 快速路径,bf16 TP2 下输出吞吐 +6.37%、TPOT −5.34%。

## 动机

Qwen3Next 的 GDN(gated delta rule)在 non-spec uniform decode(`T=1`)路径上,此前需要把 packed `mixed_qkv` 物化成 contiguous 的 q/k/v 再做 recurrent 计算,产生不必要的中间内存流量。PR 为这条 decode 热路径加一条专用 fast path。

## 优化点

新增 packed recurrent decode fast path:直接消费 packed `mixed_qkv`(不再 split 出连续 q/k/v),把 packed data flow 与 recurrent decode 的 state update / output writeback 融合,减少 decode 热路径的中间内存流量。该 fast path 面向 Qwen3Next GDN 的 non-spec uniform decode(`T=1`),并支持 strided `mixed_qkv`(即投影大 buffer 的 view,`stride(0) > shape[1]`)。用环境变量 `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` 守卫,默认 `0`(关闭)。新增 `fused_recurrent_gated_delta_rule_packed_decode` 算子,在 `fused_recurrent.py` / `fla/ops/__init__.py` 中导出,由 `qwen3_next.py` 接入。

## 关键改动

```text
vllm/model_executor/layers/fla/ops/fused_recurrent.py: 新增 fused_recurrent_gated_delta_rule_packed_decode,融合 packed 读取与 recurrent state update / output writeback
vllm/model_executor/layers/fla/ops/__init__.py: 导出新算子
vllm/model_executor/models/qwen3_next.py: 在 GDN non-spec uniform decode(T=1)接入 fast path
vllm/envs.py: 新增 VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE 开关(默认 0)
tests/kernels/test_fused_recurrent_packed_decode.py: 新增,对比 packed 路径与 reference 的连续 q/k/v 路径
```

关键代码(测试中 reference 路径需物化连续 q/k/v,正是 fast path 要消除的开销):

```python
    # Reference path: materialize contiguous Q/K/V + explicit gating.
    q, k, v = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    q = q.view(B, H, K).unsqueeze(1).contiguous()
    k = k.view(B, H, K).unsqueeze(1).contiguous()
    v = v.view(B, HV, V).unsqueeze(1).contiguous()
```

## 性能声明

正文称 "Compared to the main branch, this PR improves Output token throughput (tok/s) by ~6.37%, reduces Mean TPOT (ms) by ~5.34%, reduces Mean E2EL (ms) by ~6.56%, and reduces Mean TTFT (ms) by ~9.79%"。条件:`python3 -m vllm.entrypoints.openai.api_server --dtype bfloat16 --model /nas/disk1/Qwen3.5-35B-A3B --served-model-name Qwen3.5-35B-A3B --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max-model-len 32768 --max-num-batched-tokens 32768 --trust-remote-code --no-enable-prefix-caching`(PR 正文未注明硬件与 workload 的请求构造;benchmark 结果中 800 requests、102400 input / 80000 output tokens)。Main:800 successful / 0 failed、Output token throughput 8533.93 tok/s、Request throughput 85.34 req/s、Mean TTFT 2289.22 ms、Mean TPOT 62.00 ms、Mean E2EL 8427.53 ms;PR:800 successful / 0 failed、9077.62 tok/s、90.78 req/s、Mean TTFT 2064.99 ms、Mean TPOT 58.69 ms、Mean E2EL 7874.99 ms。其余指标(Main → PR):Peak output token throughput 13948.00 → 14744.00 tok/s、Total token throughput 19457.36 → 20696.97 tok/s、Median TTFT 2349.07 → 1807.52 ms、P99 TTFT 8254.40 → 7798.07 ms、Median TPOT 61.78 → 61.33 ms、P99 TPOT 69.39 → 66.13 ms、Mean ITL 61.63 → 58.25 ms、Median ITL 56.47 → 53.12 ms、P99 ITL 343.68 → 310.54 ms、Median E2EL 8461.47 → 7882.78 ms、P99 E2EL 9300.76 → 8737.92 ms、Benchmark duration 9.37 → 8.81 s;两臂 Total input tokens 102400、Total generated tokens 80000、Peak concurrent requests 800.00 相同。

## 验证与回归风险

`python3 -m pytest -q tests/kernels/test_fused_recurrent_packed_decode.py`:4 passed in 8.55s。测试用 B=32、H=4、HV=8、K=128、V=128 的小型 GDN 配置(注释说明 Qwen3Next 默认 K=128、V=128),按 dtype(float16 / bfloat16 / float32)与 `strided_mixed_qkv`(False/True)参数化:strided 分支先建 `(B, qkv_dim + 64)` 的投影大 buffer 再切片,模拟 `mixed_qkv` 是更大投影缓冲的 view;`ssm_state_indices` 末 3 个置为 `PAD_SLOT_ID=-1` 覆盖连续 batching,并分别克隆 `state_ref` 与 `state_packed` 做状态对比。reference 路径先 `torch.split` 出 q/k/v 再 `.contiguous()`,显式计算 `a + dt_bias`、softplus 与 `g = -exp(A_log) * softplus_x` 门控,正是 packed 快速路径要消除的物化与中间张量开销。风险面:fast path 受环境变量 `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` 守卫且默认关闭,仅作用于 Qwen3Next GDN 的 non-spec uniform decode(`T=1`),改动 `fla/ops` 的 fused recurrent 算子与 `qwen3_next.py` 的 decode 分支;正文未列精度(eval)结果或其他模型的回归测试。

## 关联

无(PR 正文未引用其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
