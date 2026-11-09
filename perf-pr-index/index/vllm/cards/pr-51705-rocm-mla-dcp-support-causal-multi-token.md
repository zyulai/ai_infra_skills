# PR #51705: [ROCm][MLA][DCP] Support causal multi-token verification

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @YukioZzz |
| 合并 | 2026-08-31 |
| 分类 | Speculative Decoding (tier: major) |
| Label | rocm, speculative-decoding, ready, deepseek, kv-connector, nvidia, mrv2, dflash, kimi, k3, scheduler, kv-cache-manager |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明 "2 file(s) dropped from diff" |
| 链接 | https://github.com/vllm-project/vllm/pull/51705 |

## 一句话结论

为 ROCm AITER MLA 加入 DCP 因果多 token 验证，Kimi-K3 TP8/DCP8 下 GSM8K strict-match 0.9613。

## 动机

本 PR 为 ROCm AITER MLA 的 decode context parallelism 启用 causal multi-token target verification：让每个验证 token 得到正确的 per-rank causal KV window，rank-local partial outputs 通过现有 DCP attention merge 合并。范围明确限定在 ROCm AITER target backend，hybrid prefix-cache geometry 与 external connector integration 留作单独改动。正文以 Summary 形式给出目标，未描述此前具体故障现象或性能瓶颈数据。

## 优化点

Implementation 列出七点：(1) 计算每个验证行的全局 causal boundary 并映射到 rank-local KV length；(2) 把物理 KV blocks 展开为 segmented AITER MLA 消费的有界 subpages；(3) 用未归约 partial output 跑 segmented MLA，把 base-2 statistics 合并成 BF16 输出加 natural-log LSE 供跨 rank merge；(4) 让 metadata builder 成为 decode routing 的唯一 owner，`AiterDCPVerifyView` 的存在即选择 segmented DCP verification，impl 消费该决定而不重算；(5) 分配稳定 row metadata 并在 FULL graph capture 期间用 maximum DCP-local KV bound，replay 仍按实际 row lengths 掩码非活跃 segments；(6) 单 token decode 保持现有 Gluon/ASM 选择，DCP 多 token verification 不进入那些路径；(7) 允许 ROCm DCP 配置保留 FULL CUDA graphs，PCP 仍在现有 PIECEWISE fallback。

## 关键改动

```text
vllm/v1/attention/backends/mla/rocm_aiter_mla.py: 计算 causal row boundary、block→subpage 展开、segmented MLA 未归约输出与 base-2/LSE 合并、AiterDCPVerifyView 路由
vllm/platforms/rocm.py: 允许 ROCm DCP 配置保留 FULL CUDA graphs
tests/kernels/attention/test_rocm_aiter_mla_causal_verify_mask.py: verify routing 由 metadata 决定的回归测试
tests/kernels/attention/test_rocm_aiter_mla_mtp_split.py 与 tests/v1/attention/test_rocm_aiter_mla_fp8_decode_routing.py: MTP split 与 FP8 单 token / DCP 多 token 路由
```

## 性能声明

Focused ROCm tests（revision `e1843114b7c233a9c71ad44b28bf63426ad64836`）：`540 passed, 14 warnings in 31.56s`，含 `DCP=2`、`qlen=3`、96 gathered query heads 的 FP8 segmented-kernel 正确性测试。Negative control：仅把 row-boundary 计算改回旧的 committed-prefix-only 形式，因果参考测试确定性失败，`2,671/147,456` 输出元素超容差、最大绝对差 `0.1311`（仅测试改动，未提交）。Full-model regression：`vllm/vllm-openai-rocm:nightly`、real Kimi-K3 target weights、TP8/DCP8、A2A DCP、KV interleave 1、FP8 KV cache、max model length 1,048,576、FULL_AND_PIECEWISE graphs（capture 1-32）、max batch size 32、prefix caching enabled；Full lm-eval GSM8K 5-shot 1,319 samples、concurrency 64、`max_length=2047`：strict-match 与 flexible-extract `exact_match` 均 `0.9613343442 ± 0.0053105832`，evaluation time `335 seconds`，server errors/tracebacks `0`/`0`，HSA memory-access faults `0`；server 选 `ROCM_AITER_MLA`，暴露 `27,012,229` KV-cache tokens。Stacked run（PR 54546 stacked）：`543 passed, 14 warnings in 31.75s`，GSM8K `1319/1319` at concurrency 64，strict-match `0.9628506444 ± 0.0052095163`、flexible-extract `0.9636087945 ± 0.0051581135`，evaluation time `353 seconds`，`mean acceptance length 2.67-2.74 in the final 10 windows`，errors/tracebacks `0/0`、HSA faults `0`。PR 正文未给出吞吐/延迟加速数字。

## 验证与回归风险

除上述 focused 与 full-model/stacked 测试外，正文说明该 target-only run 验证 DCP8/FP8-KV 全模型精度基线，因果多 token target 路径由聚焦因果参考测试直接覆盖；stacked run 把本 PR 的 segmented causal target-verification 与 PR 54546 的 non-causal draft 一起跑，server 完成 FULL graph capture 并在评测后保持健康。风险面：ROCm AITER MLA backend、DCP/PCP、FULL vs PIECEWISE CUDA graph、FP8 KV cache、投机解码 target verification；正文声明单 token decode 仍走 Gluon/ASM、DCP 多 token 不进那些路径。

## 关联

#54546（stacked 于本 PR head、为 non-causal draft group 启用 Triton MLA，正文给出 revision `8474084088b12b7bc3ca35cf6d74f87cb7161f92`）；正文另给出 base commit `2c7d7dd64a2eaba0feedf42cab2f527486d7479c`（非 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
