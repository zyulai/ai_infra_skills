# PR #40172: [Perf] [Hybrid] Fused Triton kernel for GPU-side Mamba state postprocessing

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fuscof-ibm |
| 合并 | 2026-05-21 |
| 分类 | Speculative Decoding |
| Label | ready, v1, ready-run-all-tests, verified |
| Diff | diffstat 缺失(素材未记录 +/- 行数与文件数);diff 因超长被截断,3 个文件未含(核心 kernel 实现不在素材内) |
| 链接 | https://github.com/vllm-project/vllm/pull/40172 |

## 一句话结论

把 hybrid Mamba 模型的 postprocess_mamba 从 CPU Python 循环搬进 fused Triton kernel,消除 CPU bubble:TPOT -18%、P99 ITL -36%。

## 动机

前序 #35442 已在 prefix caching 关闭时消除 `num_accepted_tokens` 阻塞拷贝造成的 CPU-GPU 同步;本 PR 处理更难的场景:开启 prefix caching 时,`postprocess_mamba` 仍要在 CPU 上拿到 `num_accepted_tokens` 与每请求元数据,才能计算 Mamba state 的 copy/reset 决策,decode 每步都因此出现 CPU bubble。

## 优化点

三步走:1) `mamba_state_idx` 从按 `req_id` 的 `dict[str, int]` 重构为按 `req_index` 的 numpy 数组,与 `num_computed_tokens_cpu` 等既有 per-request 布局对齐,可直传 GPU;2) `_prepare_inputs` 期间把 `mamba_state_idx`、`num_scheduled_tokens`、`num_computed_tokens`、`num_draft_tokens` 作为 `CpuGpuBuffer` 上传 GPU;3) 新增 `postprocess_mamba_fused_kernel`(针对 "align" cache mode),在 GPU 上完成全部 postprocess:设备端计算 block-boundary 条件与 copy 决策,用 `tl.load`/`tl.store` 直接完成 Mamba state 传输,发生 state copy/reset 时写回 `num_accepted_tokens = 1`(prefix caching 路径),彻底免掉 per-request 元数据的 CPU 同步。测试中的 CPU golden 参考保留了原决策逻辑:

```python
num_tokens_running_state = (
    num_computed_tokens + num_scheduled_tokens - num_draft_tokens
)
new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens - 1
aligned_new_computed_tokens = (
    new_num_computed_tokens // mamba_spec.block_size * mamba_spec.block_size
)
if aligned_new_computed_tokens >= num_tokens_running_state:
    accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
    ...
    if src_block_idx == dest_block_idx:
        num_accepted_tokens_cpu[i] = 1
```

## 关键改动

```text
tests/v1/worker/test_mamba_utils.py: golden tests + CPU 参考实现 + stride/边角路径测试
tests/v1/e2e/general/test_mamba_prefix_cache.py: 移除 CPU postprocess 的 fake hook
vllm/v1/worker/mamba_utils.py(推断): MambaSpecDecodeGPUContext / run_fused_postprocess(实现文件未含在素材 diff 中)
```

注:pack 的 diff 截断,3 个文件未包含(含 kernel 本体与 worker/scheduler 接线),机制描述来自 PR 正文与测试。

## 性能声明

- 条件:Qwen/Qwen3-Next-80B-A3B-Instruct-FP8,4x H100,TP4,`--max-model-len 262144`,MTP `num_speculative_tokens=2`,`--enable-prefix-caching`;latency 与 serving 各跑 3 次取 mean±std(首次丢弃)。
- `vllm bench latency`:avg 1.4895 → 1.2271 ms(-17.6%),P50 -18.0%,P99 -16.8%,全百分位 -16.8%~-18.0% 且方差更小。
- `vllm bench serve`(sharegpt,request-rate 1,200 prompts,`--ignore_eos`):mean TPOT 5.28 → 4.32 ms(-18.2%),median ITL 11.45 → 9.18 ms(-19.8%),mean TTFT 57.18 → 53.06 ms(-7.2%),P99 ITL -35.9%(±7.6 ms 方差大);output throughput 220.84 → 221.28 tok/s(+0.2%,持平);acceptance 66.02% → 66.15% 不变。
- GSM8K:main 0.8567 vs PR 0.8552(1319 题,均在容差内)。

## 验证与回归风险

新增 golden tests(`TestPostprocessMambaFusedKernel`,GPU kernel 与 Python 参考逐状态比对,batch 1–16)、生产形状 block table stride 测试(`[max_num_reqs, max_num_blocks_per_req]` 预分配表,防 stride 按字节/元素混淆)、三条分支边角测试(`src_addr==dst_addr` 跳过、同块偏移 copy、跨块 copy+accepted=1);e2e `test_mamba_prefix_cache.py` 同步更新。风险:CPU 侧 `postprocess_mamba` 生产实现被删除,只留测试里的 golden 参考;影响 hybrid Mamba 系模型在 spec decode + prefix caching("align" mode)下的路径;P99 ITL 数字受单次离群 run 影响。

## 关联

前序:#35442(消除 num_accepted_tokens 阻塞拷贝)、#35406(GSM8K correctness config 来源)。索引同区域另有 pr-61855(GDN MTP Triton port)、pr-65874(fused GDN MTP 支持 DSpark)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
