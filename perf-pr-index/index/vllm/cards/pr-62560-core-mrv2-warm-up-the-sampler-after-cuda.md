# PR #62560: [Core][MRV2] Warm up the sampler after CUDA graph capture to stop memory growth under load

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @GodRishUniverse |
| 合并 | 2026-11-02 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | ready, nvidia, mrv2 |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 完整可用(单文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/62560 |

## 一句话结论

MRV2 在 CUDA graph capture 后补一次最大 batch 的 sampler warm-up,把 fp32 logits 拷贝 buffer 预先缓存住,消除负载爬升时的显存逐级增长。

## 动机

Model Runner V2 上,sampler 每一步都会对 `[num_reqs, vocab]` 的 logits 做一份 fp32 拷贝。`capture_model()` 以 `empty_cache()` 结束,serving 启动时 caching allocator 里没有该尺寸的缓存块;负载爬升中每个新的最大 batch 都要重新 reserve 一块稍大的显存,而较小的块留在缓存里,表现为启动后显存随负载逐步增长。V1 已避免此问题——`compile_or_warm_up_model` 会在 `capture_model()` 之后特意以最大 batch 跑一次 sampler("on purpose to prevent memory buffers from being cleared by `torch.accelerator.empty_cache`"),本 PR 把同样的步骤补到 MRV2。

## 优化点

改动集中在 `gpu_worker._compile_or_warm_up_model`:当 `use_v2_model_runner` 且处于最后一个 PP rank、非 pooling 模型时,先以 `max_num_reqs = min(max_num_seqs, max_num_batched_tokens)` 跑一次 `_dummy_run`(取其 hidden states,保证是 LM head 的输入尺寸,如 OPT 会投影到更小的维度),再调用 `_dummy_sampler_run`,让 sampler 在最大 batch 下的 fp32 logits 等缓冲在 `empty_cache()` 之后仍被 caching allocator 持有;后续每个批量大小的 step 直接复用缓存块,而不是每次 reserve 新的更大块。该 warm-up 复用 V1 既有 dummy 采样路径,不引入新机制,也不额外消耗 KV cache。

## 关键改动

```text
vllm/v1/worker/gpu_worker.py: _compile_or_warm_up_model 在 capture 后为 MRV2 增加一次 max-batch dummy sampler run
```

核心片段:

```python
max_num_reqs = min(self.scheduler_config.max_num_seqs,
                   self.scheduler_config.max_num_batched_tokens)
_, sample_hidden_states = self.model_runner._dummy_run(
    num_tokens=max_num_reqs, skip_eplb=True)
self.model_runner._dummy_sampler_run(hidden_states=sample_hidden_states)
```

## 性能声明

测试环境:AMD Radeon AI PRO R9700(gfx1201,32 GB),ROCm 10.1,torch 2.13,vLLM `8a6c0449b3`;`vllm bench serve`,random 数据集、`--ignore-eos --request-rate inf`。

- Qwen3-4B-Instruct-2507,TP1,`--gpu-memory-utilization 0.9`:负载下显存增长从 main 基线 +1.78 GB(18 步)降至 +0.11 GB(1 步);峰值显存 25.72 → 24.11 GB;KV cache 105,680 tokens(与 #57926 基线一致,#57926 本身为 106,912→105,680);输出吞吐 c=16/64/160/256 为 608/1276/1561/1499 tok/s,与 main(608/1279/1562/1519)基本持平。
- Gemma-4-31B-it,TP4,`--gpu-memory-utilization 0.95 --kv-cache-memory 3GB`(固定 KV 尺寸):每卡显存增长 +0.24 GB(3 步)→ +0.003 GB;KV cache 18,806 tokens 不变;吞吐 c=8/32/64/128 为 133/496/521/537 tok/s,持平。
- 声明:No KV cache cost on top of #57926;No throughput change。

## 验证与回归风险

验证为上述两组 e2e 显存/吞吐对照(在 main、#57926、#57926+本 PR 三个构建上运行);diff 未新增单元测试。PR 明确注明"Not tested on NVIDIA; the code path is the same"。风险面:依赖 #57926——该 PR 使 `_dummy_sampler_run` 走 fp32 路径并计入 profile_run,否则 dummy run 跳过拷贝、warm-up 缓存不住任何东西;warm-up 在启动阶段多一次 max-batch dummy forward+sample,略增启动时间;对 pooling 模型与非末级 PP rank 无操作。

## 关联

依赖前序 PR #57926;同区域:pr-24128(CUDA graph capture 后运行 GC)、pr-21146(capture 期间冻结 GC)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
