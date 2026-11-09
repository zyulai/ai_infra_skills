# PR #35777: [Kernel] Add fused_sigmoid_gating_delta_rule_update kernel for Qwen3 Next

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyang16 |
| 合并 | 2026-03-09 |
| 分类 | Speculative Decoding \| tier: major |
| Label | ready, qwen |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/35777 |

## 一句话结论

融合 GDN gating 与 recurrent delta rule 两 kernel，H200 上 Qwen3-Next 输出吞吐 +1%、MTP 下 +2%。

## 动机

Qwen3 Next 的 GDN 路径原本分两次 launch `fused_gdn_gating` 与 `fused_recurrent_gated_delta_rule`，存在额外的内存流量与 launch 开销。PR 正文称新增 `fused_sigmoid_gating_delta_rule_update` kernel 是为了节省 memory traffic 和 launch overhead，思路参考 vllm-ascend 的 `sigmoid_gating.py`。正文说明该融合对 spec decoding 与非 spec decoding 两条路径都适用，Profiling 部分以截图对比了改动前后的 kernel 启动情况，并给出 Main 与 PR 两组的耗时估计。

## 优化点

把 `fused_gdn_gating` 与 `fused_recurrent_gated_delta_rule` 融合为单一 `fused_sigmoid_gating_delta_rule_update` kernel，同时支持 spec decoding 与非 spec decoding；把 `b_q / sqrt` 改写为 `b_q * rsqrt`。Profiling 显示 Main 需启动 `fused_gdn_gating_kernel` 与 `fused_recurrent_gated_delta_rule_fwd_kernel` 两个 kernel、约 5µs，PR 只启动 `fused_sigmoid_gating_delta_rule_update_kernel`、约 3µs。新增单元测试 `test_fused_sigmoid_gating_delta_rule.py`。正文在 Summary 中列出四项改动：把两个 kernel 合一、同时支持 spec decoding 与非 spec decoding、把 `b_q / sqrt` 改写为 `b_q * rsqrt`、以及新增上述单元测试，未说明 `rsqrt` 改写的进一步动机。正文在 H200 上共给出三组 serving benchmark（Qwen3-Next 非 MTP、Qwen3-Next MTP 开启 4 个投机 token、以及 Qwen3.5-35B-A3B TP1），三组均使用 sharegpt 数据集、1000 prompts、max-concurrency 8 与 temperature 0，并采集 ttft / tpot / itl / e2el 分位数。

## 关键改动

```text
vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py: 新增 fused_sigmoid_gating_delta_rule_update 融合 kernel
vllm/model_executor/layers/fla/ops/__init__.py: 导出 / 注册新 kernel
vllm/model_executor/models/qwen3_next.py: Qwen3 Next 模型接入融合 kernel
tests/kernels/test_fused_sigmoid_gating_delta_rule.py: 新增单元测试
```

## 性能声明

Benchmarked on H200。模型 `Qwen/Qwen3-Next-80B-A3B-Thinking`，serve 参数 `--tensor-parallel-size 8 --max-num-seqs 8 --no-enable-prefix-caching`；`vllm bench serve` 用 sharegpt dataset、`--sharegpt-output-len 300 --num-prompts 1000 --max-concurrency 8 --num-warmups 50 --ignore-eos --temperature 0`、`--percentile-metrics "ttft,tpot,itl,e2el"`。非 MTP：Output token throughput 1082.26 → 1094.38 tok/s，正文"Output token throughput improves 1%."。Qwen3-Next MTP（`qwen3_next_mtp`，`num_speculative_tokens` 4，其余命令相同）：1202.86 → 1230.99 tok/s，"improves 2%."。另测 Qwen3.5-35B-A3B（TP1）：775.71 → 780.68 tok/s，正文未给出该项百分比结论。PR 未注明精度（dtype）条件。

## 验证与回归风险

Test Plan 为 `pytest -s -v tests/kernels/test_fused_sigmoid_gating_delta_rule.py`，Test Result 记"Unit tests passed."。正文未说明回归风险；融合 kernel 同时覆盖 spec decoding 与非 spec decoding 路径，可能影响 Qwen3-Next 及同样使用 GDN 的 Qwen3.5 等模型，正文未逐一列出受影响路径。

## 关联

无。正文仅提到思路灵感来自 vllm-ascend 的 `sigmoid_gating.py`（外部仓库文件，非 PR 编号），未给出可关联的同分类 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
