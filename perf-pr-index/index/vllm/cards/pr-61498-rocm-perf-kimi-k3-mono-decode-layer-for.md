# PR #61498: [ROCm][Perf][Kimi-K3] Mono decode layer for Kimi-K3 on MI355X w/ vanilla decode

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jamesETsmith |
| 合并 | 2026-11-10 |
| 分类 | Speculative Decoding(tier: major) |
| Label | rocm, ready, ci/build, mrv2, verified, kimi, k3, scheduler |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/61498 |

## 一句话结论

把 Kimi-K3 的 MI355X MonoKernel 从 spec-verify 扩展到普通 autoregressive decode,低并发输出吞吐 +46.2%。

## 动机

基座 PR #60683 要求 speculative decoding,只把 spec-verify 步骤路由进 Kimi-K3 MonoKernel;普通 autoregressive decode 在配置阶段即被拒绝,尽管其 one-token decode 形状本可复用现有 `qlen=1` 的 K1 specialization 与 K2 内核。本 PR 只移除这一限制,并把 regular-decode metadata 适配到现有内核接口,是 #60683 的 stacked extension,须在其后合并。

## 优化点

regular decode 时,`non_spec_state_indices_tensor` 为每个 active row 提供一个 KDA state slot,state indices 以 `[requests, 1]` 形状暴露给 K1;runner 预分配全 1 的 `regular_num_accepted` tensor 作为 graph-stable 的 accepted-token counts(一次性分配,使已捕获 graph 保持其地址)。eligibility 要求:无 prefill rows、无 speculative rows、每个 active request 恰一个 regular-decode row、一维有效 state indices、launch width 不超过 MAX_ROWS(16)。宽度超过 `min(max_num_seqs, 16)` 的步骤继续走标准 vLLM layers;配置了 speculative decoding 时,原 spec-verify 路径不变。约束沿用 #60683:MI355X/gfx950、TP8、FULL CUDA graphs、仅 pure decode steps。注意:PR 正文两个版本的宽度上限表述不一(1–16 与 1–8);README 与 `MAX_ROWS = 16` 取 16,而性能表注明 c16 一行是在 8-row cap 的 commit `746699b4` 上测得、待当前 head 重测。

## 关键改动

```text
vllm/models/kimi_k3/amd/specialized_decode/runtime/__init__.py: DecodeConfig 支持 q=1(无 spec),移除"decoding without speculative decoding"拒绝项
vllm/models/kimi_k3/amd/specialized_decode/runtime/decode.py: step_reason() 增加 q 参数,接受 pure uniform regular-decode 步骤
vllm/models/kimi_k3/amd/specialized_decode/runtime/layers.py: _k1_metadata() 按 spec/regular 分派 state indices 与 num_accepted
vllm/models/kimi_k3/amd/specialized_decode/runtime/runner.py: 预分配 regular_num_accepted 全 1 tensor
tests/models/kimi_k3/amd_specialized_decode/: numerics/runtime/engine_fault 覆盖 regular decode
```

```python
state_indices = m.non_spec_state_indices_tensor
if state_indices is None:
    raise RuntimeError("regular decode requires KDA state indices")
state_indices = state_indices.unsqueeze(-1)
return (
    1,
    state_indices,
    runner.regular_num_accepted[: state_indices.size(0)],
)
```

## 性能声明

`vllm bench serve`:64K shared cached prefix + 16K unique input,forced 1K output with ignore-EOS,16 warm-ups,`10 × concurrency` measured prompts;单 TP8 MI355X 节点,`moonshotai/Kimi-K3`(revision `f831ab66814297da540d832a5235f8e904f29d06`),无 speculative decoding/MTP/DSpark;mono off/on 仅 `--additional-config '{"decode_backend":"latency_specialized"}'`(另一版本正文写 `--monokernel`)不同。

| Concurrency | Output tok/s off→on | Change | Median TPOT off/on | Mono route share |
|---:|---:|---:|---:|---:|
| 1 | 55.20 → 80.68 | +46.2% | 16.69 / 10.99 ms | 99.8% |
| 4 | 165.12 → 226.90 | +37.4% | 21.15 / 13.82 ms | 99.1% |
| 8 | 278.00 → 326.78 | +17.5% | 23.99 / 19.65 ms | 98.3% |
| 16 | 397.23 → 402.21 | +1.3% | 33.48 / 33.41 ms | 0.4%(按设计回退) |

准确度(GSM8K 5-shot/temp 0/c8;AIME25 30 题 4 seeds/temp 0.6;GPQA-Diamond 198 题×两序/temp 1.0):GSM8K 96.13%→95.45%(−0.68%),AIME25 82.50%→85.00%,GPQA c1 88.89%→86.87%、c8 87.37%→91.41%;NIAH 除 multikey_3 256K 82.5%→75.0% 外持平或 +5%。AIME25/GPQA 仅 30 题、对 seed 敏感,mono on/off 用相同 seeds。

## 验证与回归风险

单测覆盖 routing/configuration、regular-decode state handling、K1/K2 TP8 numerics(对 vLLM ops)、300 CUDA-graph replays、compile-without-launch warm-up、engine fault/fail-stop(spec 与 regular 两模式参数化);正文一个版本给出 `78 passed (434.35s)`,另一版本在当前 head 标记 TODO 未重跑。准确度已跑 GSM8K/AIME25/GPQA-Diamond。风险面:只影响 Kimi-K3 MI355X/gfx950 TP8 FULL-graph decode 路径;`step_reason` 把混合批次、超宽、非法 state indices 的步骤回退标准 vLLM 执行;NIAH multikey_3 256K −7.5% 值得留意。

## 关联

#60683(基座,spec-verify MonoKernel)、#61435(spec 路径扩展至 64 rows 并加 M1,明确拒绝非 spec 配置)、#60997(gfx942 port)、RFC #60904(MonoKernel 设计与验证预期)、#61568(随后打开,待评估重叠)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
