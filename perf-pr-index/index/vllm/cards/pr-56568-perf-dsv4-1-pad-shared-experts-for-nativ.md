# PR #56568: [Perf][DSV4.1] Pad shared experts for native MegaMoE fusion

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gcanlin |
| 合并 | 2026-09-16 |
| 分类 | Attention / Kernel \| tier: major |
| Label | ready, deepseek, verified, DSv4, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/56568 |

## 一句话结论

shared expert 补零对齐后 DSv4.1 MegaMoE 全 40 层融合，输出吞吐最高 +17.7%。

## 动机

正文给出 issue #56217：DeepSeek-V4.1-Flash 宽度为 2304 的 shared expert 无法与 MegaMoE 中 pad 到 2560 的 routed experts 融合。本 PR 通过对 shared weights 补零并配 unit scales，使全部 40 层都能走融合路径，同时保留 checkpoint-shaped 参数与 unsupported-layout fallback。

## 优化点

对 shared expert 的权重用 zeros、scales 用 unit scales 做 padding，把 2304 对齐到与 routed experts 一致的 2560 宽度，从而启用 native MegaMoE fusion；正文强调保留原有 checkpoint 形状的参数与不支持布局时的 fallback。部署侧以 `VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION=1 / 0` 分别走 serial shared experts 与 fused 两个 arm，load-time hook 在 8 个 rank 上校验融合状态。

## 关键改动

```text
vllm/models/deepseek_v4/nvidia/model.py: 对 shared expert 权重补零、补 unit scales，启用 40 层 MegaMoE 融合并保留 fallback
tests/models/test_deepseek_v4_mega_moe.py: 覆盖 shared expert padding 与融合状态的测试
```

## 性能声明

测试条件：8×B200，TP8/EP/SP，FP8 KV，16K context，`VLLM_USE_V2_MODEL_RUNNER=1`，`--moe-backend deep_gemm_mega_moe`，`vllm bench serve`（2 warmups/run，除 8192/1024 为 1 run 外取 3 次中位数，无 profiler），1,256 个请求全部以请求 token 数成功完成。输出吞吐（serial → fused）：128/256 C1 122.6 → 134.3 tok/s（+9.6%）、1024/128 C1 109.2 → 121.0（+10.9%）、1024/128 C64 2853.9 → 3359.2（+17.7%）、8192/32 C16 192.7 → 226.1（+17.3%）、8192/1024 C16 1292.1 → 1428.8（+10.6%）；对应 Mean TPOT 分别 -8.8%、-8.8%、-13.6%、-14.3%、-9.6%。Mean TTFT 1024/128 C64 706.90 → 558.69 ms（-21.0%）。正文注明 C64 首次运行两侧都撞上 quantization JIT，剔除后吞吐增益从 +17.7% 变为 +16.6%。GSM8K：flexible-extract 96.21%（1269/1319）→ 96.13%（1268/1319，-0.076pp），strict-match 78.17%（1031/1319）→ 79.23%（1045/1319，+1.061pp）。

## 验证与回归风险

Test Result：B200 上 31 tests passed；Ruff、typos、mypy 3.10/3.12 与适用的 pre-commit 检查通过（不匹配的 Actionlint 与 requirements-compilation hooks 跳过）；8 个 rank 均确认 0 → 40 层融合。回归风险：因原始 shared 参数仍被分配，model-load 显存由 41.47 增至 43.34 GiB/rank（+1.87 GiB）；C1 每次仅 8 个请求，尾延迟估计有限；GSM8K flexible-extract 略降 0.076pp。正文未给出其他模型或硬件的回归评估。

## 关联

正文给出 issue #56217；说明 Extends #53040（benchmark workloads 沿用 #53040 并增加 8192/1024 C16）；并称检查 #45861 及相关 open PR 未找到该 padding gap 的修复，#53567 与 #54049 覆盖不同路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
