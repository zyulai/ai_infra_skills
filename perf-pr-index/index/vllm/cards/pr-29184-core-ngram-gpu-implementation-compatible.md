# PR #29184: [Core] NGram GPU Implementation compatible with Async Scheduler

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @PatchouliTIS |
| 合并 | 2026-03-07 |
| 分类 | Speculative Decoding (tier: major) |
| Label | speculative-decoding, ready, v1, ready-run-all-tests |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾标注 "2 file(s) dropped from diff for size"(见 PR 完整 diff),diff 保留 8 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/29184 |

## 一句话结论

实现 GPU 版 ngram 投机解码并兼容 Async Scheduler,H20 Qwen3-1.7B 上 async ngram 比 sync ngram 快 20.6%~39.4%。

## 动机

本 PR 基于 #24799,目的是实现 GPU 版本的 ngram speculative decoding,并使其与 Async Scheduler 兼容。正文 Purpose 仅此一句,未提供更多瓶颈描述;PR 正文内多段 Cursor Bugbot 自动摘要说明引入 GPU proposer 并把 `ngram_gpu` 接入 runner 与 config。

## 优化点

新增 `vllm/v1/spec_decode/ngram_proposer_gpu.py`:完全向量化、torch.compile 优化的 `NgramGPUKernel` 与 `NgramProposerGPU`,不做 CUDA Graph capture,含 warmup。`gpu_model_runner.py` 端到端支持 `ngram_gpu`:维护 GPU 上的 `token_ids` / `num_tokens_no_spec` buffer、batch reorder / condense 后增量更新 GPU tensor、处理 empty / invalid draft,并有 async output 路径与 drafter 无法运行时的 fallback(含对 scheduler output 的安全拷贝)。config 侧新增 `NgramGPUTypes`、`use_ngram_gpu()`、`ngram_gpu` 方法归一化与 async scheduling 校验,并在选中时把 model 设为 `ngram_gpu`(`config/speculative.py`、`config/vllm.py`)。`compilation/backends.py` 在启用 `ngram_gpu` 时禁用 vLLM 的 torch.compile cache 以避免缓存错误。`gpu_input_batch.py` 把 `num_tokens_no_spec` 存成 pinned CPU torch tensor 以加快传输。测试侧新增 `test_with_ngram_gpu_spec_decoding`、把原 spec decode 测试重命名为 eagle3、记录 `spec_method`、把测试 `max_model_len` 提到 4096。

## 关键改动

```text
vllm/v1/spec_decode/ngram_proposer_gpu.py: 新增 NgramGPUKernel(torch.compile 优化、向量化、无 CUDA Graph)与 NgramProposerGPU,含 warmup
vllm/v1/worker/gpu_model_runner.py: 端到端接入 ngram_gpu(GPU token/state buffer、增量更新、empty/invalid draft、async 输出与 fallback)
vllm/config/speculative.py, vllm/config/vllm.py: NgramGPUTypes、use_ngram_gpu()、方法归一化、async scheduling 校验
vllm/compilation/backends.py, vllm/v1/worker/gpu_input_batch.py: 启用 ngram_gpu 时禁用 torch.compile cache;num_tokens_no_spec 存为 pinned CPU tensor
tests/v1/e2e/test_async_scheduling.py: 新增 test_with_ngram_gpu_spec_decoding、重命名 eagle3 用例、max_model_len 提到 4096
```

关键代码(新增 e2e 测试的 ngram_gpu 配置):

```python
    ngram_gpu_config = {
        "method": "ngram_gpu",
        "num_speculative_tokens": 3,
        "prompt_lookup_max": 3,
        "prompt_lookup_min": 2,
    }
```

注:上述机制要点来自 PR 正文内的 Cursor Bugbot 摘要;pack 末尾标注 2 个文件因超长被丢弃。

## 性能声明

Test 条件:Async Scheduler + NGram + `Qwen/Qwen3-1.7B`,`api_server --max-num-seqs 128 --max-model-len 2048 --model Qwen/Qwen3-1.7B --tensor-parallel-size 1 --dtype bfloat16 --enable-chunked-prefill --disable-log-requests --async-scheduling --speculative_config '{"method": "ngram_gpu", "num_speculative_tokens": 3, "prompt_lookup_max": 2, "prompt_lookup_min": 2}'`,dataset 为 CMU-DoG(input-grounded),Test Device: NVIDIA H20。Performance(tps,`async_ngram` vs `sync_ngram`,speedup):num_prompts 2 → 466 vs 357(30.5%);8 → 1378 vs 988(39.4%);16 → 2082 vs 1726(20.6%)。PR 正文未给出该 benchmark 的精度字段,也未提供 accuracy / eval 结果。注:pack 的 `perf claims detected in body (regex)` 为 "(none detected by regex)",但正文 Test Result 表格含上述定量对比。

## 验证与回归风险

Test Plan 仅列出上述 async_scheduler + ngram + Qwen3-1.7B 的服务启动配置与 tps 对比;测试侧新增 `test_with_ngram_gpu_spec_decoding`(覆盖 num_speculative_tokens 2-6、prompt_lookup min/max 窗口、async scheduling、不同 executor 与 chunking 设置),并重命名原 eagle3 用例。风险面:改动 `gpu_model_runner.py` 的共享 decode / draft 路径、`config/speculative.py` 与 `config/vllm.py` 的方法归一化与 async scheduling 校验、`compilation/backends.py` 的 torch.compile cache 开关(仅 ngram_gpu 启用时)与 `gpu_input_batch.py` 的 pinned tensor;正文未列精度评测或其他模型的回归测试;PR 正文含多段重复的 Cursor Bugbot 自动摘要。

## 关联

PR 正文明确:本 PR 基于 https://github.com/vllm-project/vllm/pull/24799(#24799)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
