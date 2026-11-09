# PR #63842: [Perf][Frontend] Compute prompt block hashes in the Rust frontend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @njhill |
| 合并 | 2026-10-31 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | ready, ci/build, cpu, rust, kv-cache-manager |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾标注 "12 file(s) dropped from diff for size"(见 PR 完整 diff),diff 保留 9 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/63842 |

## 一句话结论

把 prompt block 哈希移到 Rust frontend,消除 engine 持 GIL 的哈希,GB300 长前缀缓存 decode 吞吐 +9.6%(sha256)/+39%(sha256_cbor)。

## 动机

此前 engine 的 input thread 在持有 GIL 时对长 prompt 计算 block hashes,而 GIL 正是 engine loop 所需,长 prompt 会 stall engine loop。PR 正文给出约 50 ms/227K-token prompt(sha256)、约 150 ms(sha256_cbor)的哈希开销。该改动属于长上下文 P/D CPU-overhead 工作,并在 Rust frontend 上复现 #15652 的思路。

## 优化点

让 Rust frontend 计算请求的 prompt block hashes 并随请求下发,engine 直接接受 frontend 计算好的哈希。引擎新增 `EngineCoreReadyResponse.prefix_hash_config`(算法、hash_block_size、NONE_HASH、lookahead),发送 NONE_HASH 可免去 frontend 复现 seed 逻辑;`EngineCoreRequest.block_hashes` 作为 `Request.block_hashes` 的种子,已有 hasher 从该处继续,engine 只校验数量 `(prompt_len - lookahead) // hash_block_size`,否则自行哈希。Rust encoder 精确写 pickle protocol 5 / canonical CBOR,覆盖 sha256、sha256_cbor 及 plain、LoRA、cache-salt、lookahead(EAGLE)请求;multimodal、prompt-embeds、xxhash*、超 64 KiB pickle frame、可被 pickle 按 identity memoize 的请求仍由 engine 哈希。前缀缓存哈希字节一致,cache hit、KV connectors、KV events 不受影响;Python frontend 暂不发送哈希。

## 关键改动

```text
rust/src/engine-core-client/src/prompt_block_hasher.rs: 新增 PromptBlockHasher,在 Rust 侧生成与 engine 字节一致的 prompt block hashes(pickle 5 / canonical CBOR)
rust/src/engine-core-client/src/client.rs: 仅当所有 engine 上报相同 prefix_hash_config 时构建 hasher,并在 call() 中于请求无 block_hashes 时填充
rust/src/engine-core-client/src/protocol/handshake.rs, protocol/request.rs: 新增 prefix_hash_config 与 block_hashes 字段
rust/src/engine-core-client/Cargo.toml, src/lib.rs, src/mock_engine.rs: 引入 sha2 依赖、注册模块、mock ready_response 增加 None 配置
rust/src/engine-core-client/src/tests/client.rs, tests/prompt_block_hash_vectors.json: 用 engine 自身 Request/get_request_block_hasher 生成的 known-answer 向量校验
```

关键代码(仅当各 engine 配置一致才启用 frontend 哈希):

```rust
fn prompt_block_hasher(engines: &[ConnectedEngine]) -> Option<PromptBlockHasher> {
    let config = (engines.iter())
        .map(|engine| engine.ready_response.prefix_hash_config.as_ref())
        .all_equal_value()
        .ok()??;
    let hasher = PromptBlockHasher::new(config);
    hasher
}
```

## 性能声明

Benchmark 条件:GB300、`Qwen3-0.6B`、Rust frontend(`VLLM_USE_RUST_FRONTEND=1`)、CUDA graphs;64 路 streaming decode 请求,同时把完全前缀缓存的 30K-token prompt(token ids、`max_tokens=1`)以 20/s 持续 40 s 送入,模拟 P/D decode 实例接纳长 prompt;两臂同一二进制,"engine" 臂停止 advertise `prefix_hash_config`;取 2 轮交错测量均值(各轮相差 2% 内)。Decode tok/s:`sha256` engine(main)16,870 → frontend 18,494(+9.6%);`sha256_cbor` engine 13,244 → frontend 18,366(+39%)。ITL mean/p90/p99(ms):sha256 由 3.77/8.36/10.10 到 3.44/6.51/9.83,sha256_cbor 由 4.80/10.41/15.64 到 3.46/6.45/10.04;长请求 TTFT p50/p99(ms):sha256 27.6/46.1 → 25.6/37.8。PR 注明增益取决于哈希占用 engine loop 的比例:小模型 + 长 prompt 高到达率时大,更大模型相对增益更小。

## 验证与回归风险

`pytest tests/v1/core/test_kv_cache_utils.py`:114 passed;`cargo test -p vllm-engine-core-client -p vllm-llm`:全部通过,含由 engine 自身 `Request`/`get_request_block_hasher` 生成的 known-answer 向量。E2E 在 GB300 用 test-only engine patch 重算每个带 frontend 哈希的请求并比对:Qwen3-0.6B `sha256`/`sha256_cbor` + LoRA 各 12 请求 12/12 匹配、Qwen3-1.7B + EAGLE3(lookahead 1)8 请求 8/8 匹配,重复请求缓存 token 5424/5427(新 salt 或 LoRA 从 0 开始)。风险面:哈希字节一致性关系 prefix cache 命中、KV connectors 与 KV events,正文称三者不受影响;Python frontend 路径未启用。

## 关联

PR 正文明确:复现 #15652 的思路、与 #63454(更便宜的 engine 侧哈希)互补、与 #43483(DP routing 的 frontend 哈希)相关、属于 #63439 长上下文 P/D CPU-overhead 工作。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
