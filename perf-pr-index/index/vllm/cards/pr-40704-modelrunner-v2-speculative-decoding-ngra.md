# PR #40704: [ModelRunner V2] Speculative Decoding NGram GPU Implementations

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @PatchouliTIS |
| 合并 | 2026-10-06 |
| 分类 | Speculative Decoding \| tier: major |
| Label | speculative-decoding, ready, torch.compile, ci/build, v1, mrv2 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/40704 |

## 一句话结论

为 ModelRunner V2 新增 NGram GPU speculator；sonnet bench 的 V2 结果接受率 52.64%、接受长度 2.58。

## 动机

PR 正文 Purpose 列出三点：新增 NGram GPU speculator（实现位于 `vllm/v1/worker/gpu/spec_decode/ngram/speculator.py`，类似 #29184）；更新 NGram GPU 的请求状态存储，使 `all_token_ids` 在 `ngram_gpu` 激活时保持 GPU 稠密驻留而非默认 UVA——因为新 n-gram speculator 需反复扫描活跃请求的 token 历史，GPU 稠密存储比经 UVA 拉取更合适（对应 #29184 的讨论），`model_runner.py` 还向需要直接访问持久 token store 的 speculator 注入 `req_states`；新增变长 draft token 管道，`DraftTokenIds` 增加 `num_valid_draft_tokens: list[int] | None`，scheduler 按它截断 speculative tokens，`EngineCore` 增加 `_maybe_update_async_draft_token_ids()`。

## 优化点

核心是新增的 Triton `_ngram_scan_kernel`：一次把长度 MAX_N 的后缀载入寄存器，对 `[MIN_N, MAX_N]` 的每个 n 用 `tl.static_range` 逐位比对历史 token 与后缀，把 `(n, pos)` 打包成 `n*(L+1)+pos+1` 用一次 max 得到最长 n、最右位置；不合格行与超出最后候选位置的 block 直接写 0 退出。配合 `RequestState` 新增 `use_dense_all_token_ids`，在 ngram_gpu 下把 `all_token_ids` 的 `StagedWriteTensor` 从 `uva_instead_of_gpu=True` 改为 GPU 常驻。config 侧把 `ngram`/`ngram_gpu` 从 V2 model runner 的 unsupported features 列表移除，并加 `use_ngram()`。

## 关键改动

```text
vllm/v1/worker/gpu/spec_decode/ngram/speculator.py: 新增 NGram GPU speculator 与 Triton _ngram_scan_kernel（278 行新文件）。
vllm/v1/worker/gpu/states.py: RequestState 增加 use_dense_all_token_ids，ngram_gpu 下让 all_token_ids GPU 稠密驻留。
vllm/v1/worker/gpu/model_runner.py: 向 speculator 注入 req_states。
vllm/config/speculative.py / vllm/config/vllm.py: 增加 use_ngram()，把 ngram/ngram_gpu 从 V2 model runner 不支持列表移除。
tests/v1/spec_decode/test_ngram_gpu.py: 新增 NGram GPU 测试。
```

```python
    for n_iter in tl.static_range(MIN_N, MAX_N + 1):
        max_pos_n = seq_len - n_iter - 1
        match = (pos >= 0) & (pos <= max_pos_n)
        for j in tl.static_range(0, n_iter):
            tok = tl.load(
                token_ids_ptr + row_off + (pos + j),
                mask=match, other=0,
            ).to(tl.int32)
            suf_idx = (MAX_N - n_iter) + j
            suf_val = tl.sum(tl.where(suf_iota == suf_idx, suffix, 0))
            match = match & (tok == suf_val)
        cand = n_iter * Lp1 + pos + 1
        best_score = tl.where(match, cand, best_score)
```

## 性能声明

测试条件：`vllm bench serve --port 8000 --backend vllm --model Qwen3-8B --endpoint /v1/completions --dataset-name sonnet --dataset-path sonnet.txt --max-concurrency 128 --sonnet-input-len 128 --sonnet-output-len 100 --sonnet-prefix-len 10 --num-prompts 256 --ignore-eos --percentile-metrics "ttft,tpot,itl,e2el" --seed 1234`；硬件与精度 PR 未注明测试条件。正文给出两组结果（未标注 before/after，标为 Async NGram GPU V1 与 V2）：V1 为 256 请求、duration 10.42 s、Request throughput 24.56 req/s、Output tok/s 2456.33、Total tok/s 5497.29、Mean TTFT 1348.49 ms、Mean TPOT 34.86 ms、Mean ITL 40.09 ms、Mean E2EL 4799.33 ms、Acceptance rate 5.16%、Acceptance length 1.15；V2 为 duration 8.41 s、Request throughput 30.43 req/s、Output tok/s 3042.78、Total tok/s 6809.77、Mean TTFT 980.36 ms、Mean TPOT 31.17 ms、Mean ITL 35.77 ms、Mean E2EL 4065.85 ms、Acceptance rate 52.64%、Acceptance length 2.58。

## 验证与回归风险

PR 正文的 Test Plan 仅给出上述 `vllm bench serve` 命令与两组结果，未列出单测命令或测试通过数；diff 中含 `tests/v1/spec_decode/test_ngram_gpu.py`、`tests/test_config.py`、`tests/v1/spec_decode/test_max_len.py`、`tests/v1/worker/test_gpu_extract_hidden_states_speculator.py`、`tests/v1/worker/test_gpu_model_runner_v2.py` 等测试文件改动。可能影响路径：ModelRunner V2 的 speculative decoding（ngram/ngram_gpu）、`all_token_ids` 显存占用（GPU 稠密驻留而非 UVA，源码注释提示该张量可达数 GB）、变长 draft token 的 scheduler/EngineCore 截断逻辑，以及 `vllm/compilation/backends.py`、`vllm/distributed/elastic_ep/elastic_execute.py` 的适配。

## 关联

#29184（类似实现，V1 NGram speculator 与 UVA 讨论）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
