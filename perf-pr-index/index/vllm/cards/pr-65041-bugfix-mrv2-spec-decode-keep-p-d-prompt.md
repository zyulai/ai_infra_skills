# PR #65041: [Bugfix][MRV2][Spec Decode] Keep P/D prompt tails on FULL graphs with adaptive verification

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Pb314314 |
| 合并 | 2026-11-03 |
| 分类 | Model-specific Fast Path(major) |
| Label | bug, speculative-decoding, ready, kv-connector, nvidia, mrv2 |
| Diff | diffstat 缺失(素材包未提供 +x/-x);diff 含 runner/drafter 条件修改与三个测试文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/65041 |

## 一句话结论

修复 AV 下 P/D 到达步因 has_prefill 误判而退出 FULL decode graph 的问题(条件改为 decode_graph_eligible,两行),GB200 1P1D 吞吐 2.0x、D TPOT 19.2→6.95 ms。

## 动机

P/D decode 实例上,KV 迁移到达的请求只重算最后一个 prompt token(1 real token + K placeholder draft,decode 行形状),但仍计为 prefilling(has_prefill=True)。AV 路径(#57263)对任何 has_prefill 的 batch 丢弃 query-length bound,导致 AV 的 varlen FULL decode graph 拒绝该步:FULL_DECODE_ONLY 下整步 eager、FULL_AND_PIECEWISE 下 PIECEWISE;DP 各 rank 同步 bound 且任一 rank 发 None 即全体 None,一个 rank 的到达使所有 rank(包括纯 decode rank)eager。

## 优化点

runner(execute_model)与 drafter(TargetDependentARSpeculator.propose,EAGLE/EAGLE3/MTP)的判定从 has_prefill 改为 not decode_graph_eligible(#58400 引入、非 AV decode 路径已在用的分类器):单 token 在既有 KV 上的扩展视为 decode,保留 max_query_len bound,AV varlen FULL graph 得以 replay,drafter 保持目标侧 padded layout(#62796)。匹配单调:bound 从 None 变为有值只会增加匹配;不捕获新 graph,graph 数/显存/KV cache 不变;条件是每步已计算的 host-side bool,无新 GPU sync 或 DP collective。真实 prefill(多 token prompt、无 KV 的单 token、PCP)仍被拒绝,eager/PIECEWISE 不变。

## 关键改动

```text
vllm/v1/worker/gpu/model_runner.py: has_prefill → not decode_graph_eligible
vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py: drafter 同条件
tests/v1/worker/test_gpu_model_runner_v2.py 等: 回归测试(bound 保留、drafter
  layout 匹配)
```

## 性能声明

DeepSeek-V4.1-Flash + DSpark,vLLM P/D recipe(gb300_agentx_pd)1P1D DEP4,两台 GB200 over NIXL:baseline eager 步(约比 FULL 慢 6x)占 D 时间 77%(OSL 256)/60%(OSL 1024),其中大部分是被其他 rank 拉入 eager 的步;本 PR 后全部 replay FULL:输出吞吐 4,100/4,385 → 8,559/8,519 tok/s(OSL 256,2.0x)、8,533/8,666 → 15,099/15,231 tok/s(OSL 1024,1.76x);D TPOT 19.0/19.4 → 6.93/6.97 ms 与 10.7/10.6 → 5.71/5.66 ms。TP4 交叉验证:FULL_DECODE_ONLY+AV 的 TPOT 与 FULL_AND_PIECEWISE 一致(6.7 ms 级)。

## 验证与回归风险

单元测试 116 passed(runner v2、cudagraph manager、target_dependent_ar speculator、dp_utils、adaptive_verification);单独 revert 两个条件会分别挂 test_prompt_tail_keeps_decode_graph_bound 与 test_draft_prefill_matches_shared_target_metadata_shape。正确性:GSM8K 全集与 draft acceptance 在 baseline repeat 波动内(DSpark/MTP、TP4/DEP4);到达步 padding 到 capture size(与其他 decode 步一致)。限制:GPU 验证限 DeepSeek-V4(.1)-Flash 的 MLA 后端,FA3/FlashInfer 仅 layout contract 单测;性能数据为 DEP4(recipe 为 DEP16,未测)。

## 关联

#57263(引入 has_prefill 清 bound)、#58400(decode_graph_eligible)、#62796(drafter padded layout)、#62793(FA3 上 layout mismatch 证据)、#64915(包含同一 runner 条件的更大 feature,本 PR 先合并则其删重叠行)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
