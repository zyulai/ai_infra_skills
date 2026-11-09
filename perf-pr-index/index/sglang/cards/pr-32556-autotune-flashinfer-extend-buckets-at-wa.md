# PR #32556: Autotune flashinfer extend buckets at warmup

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @sshleifer |
| 合并 | 2026-08-07 |
| 分类 | Attention / Kernel (tier: major) |
| Label | run-ci |
| Diff | pack 未提供 diffstat(diff 完整可见,改 12 个文件,含 environ 与各 attention backend) |
| 链接 | https://github.com/sgl-project/sglang/pull/32556 |

## 一句话结论

warmup 时补跑一次 EXTEND 形状的 flashinfer autotune,修复 8k+ token prefill 落入未调优 bucket 的约 8% 吞吐回退。

## 动机

warmup 阶段的 flashinfer autotune 只跑 decode 形状的 dummy forward(1 token/req),调优 bucket 只覆盖到 decode batch size。更大的 prefill/extend batch 在服务时落在调优范围之外,回退到 flashinfer 默认启发式:flashinfer 0.6.15.post1 上 `trtllm_fp4_block_scale_routed_moe` 在 >=8192 token、sm100(GB300)时默认启发式比调优后的 tactic 慢约 30%(flashinfer-ai/flashinfer#4190),TP-only NVFP4 MoE 服务在 >=8k token extend batch 出现 −7~−9% extend 吞吐。

## 优化点

在 decode 形状 autotune 之后(`warmup` → `maybe_flashinfer_autotune_extend`)再跑一次 EXTEND 形状的 dummy forward,一次覆盖到最大 bucket 的所有更小 bucket。要点:token 数取 `max_prefill_tokens`(scheduler 每 forward 的 extend token 预算,即 kernel 可见的最大形状;用 `chunked_prefill_size` 会欠覆盖,因为多个 chunk 会合批;代码注释指出 max_prefill_tokens 是 per-dp-rank 预算且 warmup 在所有 dp rank 同时运行,不除以 dp_size)。dummy batch 按 `req_to_token_pool.size` 封顶打包(如 1024 seq x 16 token):per-seq attention metadata(如 triton `kv_indptr`)按 req pool 预分配,不打包会在 `max_prefill_tokens > max_num_seqs` 的配置上崩溃;后端通过新类属性 `extend_dummy_seqs_capped_by_req_pool` 声明是否需要封顶。dummy buffer 跳过 `(num_tokens, vocab)` fp32 logits buffer(16k token 时 >10 GB),因为 autotune 以 `skip_logits` 运行。投机解码 runner 与多模态模型暂时跳过(spec 是 follow-up);新 env `SGLANG_FLASHINFER_AUTOTUNE_EXTEND` 默认关闭(opt-in),OOM 时记录日志并回退到未调优 bucket。

## 关键改动

```text
python/sglang/srt/model_executor/runner/flashinfer_autotune.py: 新增 maybe_flashinfer_autotune_extend,EXTEND 形状 autotune 主逻辑
python/sglang/srt/model_executor/runner/base_runner.py: _dummy_run 支持 extend_num_tokens_per_req;logits buffer 可选跳过
python/sglang/srt/environ.py: 新增 SGLANG_FLASHINFER_AUTOTUNE_EXTEND(默认 False)
python/sglang/srt/layers/attention/base_attn_backend.py: 新增 extend_dummy_seqs_capped_by_req_pool 类属性
flashinfer/triton/aiter/dsa/hybrid/wave 等 backend: 按需声明该属性(True 表示 metadata 按 req pool 预分配)
```

```python
if mr.attn_backend.extend_dummy_seqs_capped_by_req_pool:
    pool_size = mr.req_to_token_pool.size
    num_tokens_per_req = (num_tokens + pool_size - 1) // pool_size
else:
    # Packed dummies tune measurably worse tactics for the same token
    # bucket, so pack only where the backend would otherwise crash.
    num_tokens_per_req = None
```

## 性能声明

测试条件:GB300,Kimi-K2 架构 NVFP4 MoE,tp4,flashinfer 0.6.15.post1,`batch_full` bench,prefill extend tok/s(bs=1, n=3, IQR≈0)。

| input_len | 0.6.13 | 0.6.15.post1 未修复 | 0.6.15.post1 + 本 PR |
|---|---|---|---|
| 4096 | 47,270 | 47,357 | 48,473 |
| 8192 | 53,155 | 48,957(−7.9%) | 53,401(vs 0.6.13 +0.7%) |
| 16384 | 51,946 | 47,877(−7.8%) | 51,349(vs 0.6.13 +0.4%) |

decode_long@8192 +9.6%、saturated-throughput +9.3%(ttft −17%,vs 未修复);wikitext/long-decode ppl 4 位数字一致,数值无变化。微基准(Kimi-K2 形状,384 experts、top-8、hidden 7168,µs/call):16384 token 时 0.6.15.post1 启发式 1713 µs vs autotuned 1304 µs(+30%);打包 dummy(1024x16)在 `max_num_seqs=1024, max_prefill_tokens=16384` 配置端到端验证通过(不打包该配置会崩溃)。

## 验证与回归风险

修改了 `test/registered/unit/layers/attention/test_verify_mask.py` 与 `test/registered/unit/spec/test_dflash_overlap_hostsync.py` 中的 fake backend 以携带新属性;CI 中 PR Test(Base)通过。回归面:各 attention backend 的 warmup/autotune 路径与显存峰值(opt-in 开关 + OOM 回退缓解);投机解码与多模态 runner 明确跳过;打包/不打包两种 dummy 形状的选择由后端属性驱动,新 backend 需正确声明否则可能在 warmup 崩溃或调出次优 tactic。

## 关联

上游:flashinfer-ai/flashinfer#4190(sm100 上 fp4 MoE 未调优回退)。同区域:warmup flashinfer autotune、`--disable_flashinfer_autotune`;pack 正文未列出其他 SGLang PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
