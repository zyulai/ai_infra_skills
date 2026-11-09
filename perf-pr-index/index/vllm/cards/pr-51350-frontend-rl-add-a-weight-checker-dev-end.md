# PR #51350: [Frontend][RL] Add a weight checker dev endpoint

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @shiyuan680 |
| 合并 | 2026-09-30 |
| 分类 | Speculative Decoding（tier: major） |
| Label | documentation, frontend, ready, ci/build, kv-connector, rust |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/51350 |

## 一句话结论

新增 POST /weight_checker 开发端点，用 SHA-256 校验 RL 权重更新是否完整落到每个 rank。

## 动机

正文 Purpose 说明：需要验证一次 RL 权重更新（`start_weight_update → update_weights* → finish_weight_update`）是否完整到达每个 engine、每个参数，避免静默的部分更新演变成 off-policy 训练，对应 RFC 为 #52689。正文强调这是该 RFC 的现有实现，且「没有其它开着的 PR 增加 weight-checksum 端点」。本 PR 是可观测/校验能力，不改变模型输出路径。

## 优化点

在 Python 与 Rust 两套 frontend 各实现一个 dev 端点 `POST /weight_checker`（需 `VLLM_SERVER_DEV_MODE=1`），支持 checksum（返回 `{key: sha256_hex}`）、reset（置零、返回 status）、compare（与 baseline 比较返回 `match`/`mismatches`）三种 action。key 形如 `dp{dp}:pp{pp}:pcp{pcp}:tp{tp}:{param_name}`，DP 部分用 `data_parallel_index`（dense workers 会把 `data_parallel_rank` 重置为 0），跨 worker 重复 key 返回 500 而非静默合并。覆盖范围只含模型 parameters（weight loading 写入的对象），排除 buffers（loading 不恢复它们，置零会让 compare 永远失败）；`SharedWeightParameter` 分区与 tensor-subclass 权重（如 TorchAO `qdata`/`scale`）展开成内部张量。reset 走 `collective_rpc` 就地置零，与 `update_weights` 同路径。checksum/compare 用新增 `EngineClient.compute_weight_checksums`：async client 经 `call_utility_all_async`（#59240）在每个 managed engine 上调用 engine utility 并拼接结果——因 `collective_rpc` 在 DP load balancing 下只返回首个 engine 结果，而各 engine 持有不同 shard；Rust 侧 `collective_rpc` 已收集所有 engine 故直接调用。端点无状态、baseline 由调用方持有，未知 action 或 compare 缺 baseline 返回 400。文档另列限制：checksum 到 compare 需保持 paused、外部 LB 需发每个 API server、不检查 draft/LoRA/buffers，以及部分后端与 offload 配置不支持、engine 必须 awake。

## 关键改动

```text
vllm/entrypoints/serve/dev/rlhf/api_router.py: 新增 POST /weight_checker 路由，实现 checksum/reset/compare 与重复 key 500 检查
vllm/model_executor/model_loader/weight_checksum.py: 新文件，compute_tensor_digests 用 sha256 逐参数摘要，zero_weights 置零，_tensors 展开分区/subclass
rust/src/server/src/routes/weight_checker.rs 与 routes.rs: 新增 Rust 路由模块并注册 /weight_checker
vllm/engine/protocol.py、vllm/v1/engine/async_llm.py、core.py、core_client.py、vllm/v1/worker/gpu_worker.py: 打通 compute_weight_checksums 到各 engine/worker 的 fan-out 调用链
docs/training/weight_checker.md 与 docs/training/rlhf.md: 新增端点文档与限制说明、补入口链接

# 机制摘录 (weight_checksum.py: compute_tensor_digests)
def compute_tensor_digests(model: nn.Module) -> dict[str, str]:
    return {
        name: hashlib.sha256(
            weight.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy()
        ).hexdigest()
        for name, weight in _weights(model)
    }
```

## 性能声明

PR 正文未给出定量性能数据（无吞吐/延迟 benchmark）。正文明确「No model output path changes, so no accuracy eval」，即不改变模型输出路径、故不做 accuracy 评测；文档另注明 hashing 会把每个权重拷到 CPU，应避开 latency-sensitive 路径。正文给出的是测试通过数量而非性能数字，见下节。

## 验证与回归风险

Test Plan 命令为 `pytest tests/model_executor/model_loader/test_weight_checksum.py` 与 `pytest tests/entrypoints/serve/dev/rlhf/state_transitions/test_weight_checker.py`。结果：Unit（CPU）6 passed（覆盖逐参数、reset 不动 buffer、strided/0-dim 参数、SharedWeightParameter 分区、tensor-subclass 内部张量、dense-DP key 前缀、DPLB 收集所有 engine，`tests/v1/engine/test_engine_core_client.py`）；Rust 路由测试 2 passed（`cargo test -p vllm-server weight_checker`，rustfmt/clippy clean）；E2E（Qwen3-0.6B）10 passed（单 GPU、IPC 传输、DP=2、TP=2）于 GB200 nightly 容器。并行拓扑 TP2/PP2/DP2/TP2+EP/DP2+EP（Qwen3-30B-A3B）/TP2xPP2 全通过，单 rank 与 1-ULP 损坏被报为恰好一个 key；pre-commit（ruff、mypy、markdownlint）clean。风险：端点需 `VLLM_SERVER_DEV_MODE=1` 才启用；hashing 需拷权重到 CPU；文档列出的不支持场景；正文明确依赖 #57163——在它落地前 compressed-tensors 把 attention KV scales 暴露为 parameters，FP8 的 reset→reload 会失败；多卡 accuracy 为手工运行、2-GPU E2E 未进 CI。

## 关联

PR 正文明确提及 RFC #52689（本端点即其实现）、依赖 #59240（`call_utility_all_async` 的 fan-out 能力）与前序依赖 #57163（compressed-tensors KV scale 作为参数的问题）；正文声明「No other open PR adds a weight-checksum endpoint」，未给出同分类其它 vLLM PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
