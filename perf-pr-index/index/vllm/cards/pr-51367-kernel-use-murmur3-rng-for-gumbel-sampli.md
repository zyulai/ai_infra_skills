# PR #51367: [Kernel] Use Murmur3 RNG for Gumbel sampling

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @freyfwt |
| 合并 | 2026-09-16 |
| 分类 | Attention / Kernel（tier: major） |
| Label | structured-output, ready, mrv2 |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/51367 |

## 一句话结论

Gumbel 采样改用 Murmur3 RNG；开发期 RTX 5090 微基准 128×163840 达 1.49x（非最终 head）。

## 动机

正文 Purpose：把 Gumbel sampler 基于 Philox 的 uniform 生成替换为以 request seed、token position、vocabulary offset 为键的手动 Murmur3 x86_32 mixing 路径，目标是在保持目标采样分布的同时提升 RNG resolution 并避免 backend-specific math paths。Related RFC #54143。正文明确这不是 #47524 的重复——后者保留 Philox、通过组合更多输出位提升 uniform resolution；本 PR 评估不同 RNG 与 backend-independent logarithm path，既有 Philox helpers 保留供 rejection sampling。

## 优化点

Implementation 列出：从完整 32-bit Murmur3 结果经两次精确 16-bit 整数到浮点转换生成 fp32 uniforms，规避 backend-dependent 的 uint32→float 行为；从 53 random bits 生成 centered fp64 uniforms 并在 fp64 rounding 后显式保持两端开放；在 fp32 winning tail 用稳定的 degree-8 近似替代 `log1p(-u)` 的 backend-specific libdevice 调用；通过 salting token position 让 draft 与 target 采样走独立 noise streams；新增 Murmur3 known-answer 与 fp64 endpoint 测试、保留 distribution 测试、在 RNG stream 非被测行为处使 output-sensitive API 测试确定性，并移除「并行 completions 必须随机以不同长度结束」的 API 级假设（staggered child completion 已由 ParentRequest 单测确定性覆盖）。diff 中 `gumbel.py` 新增 `_murmur3_rotl32`、`_murmur3_mix`、`_murmur3_fmix32`、`murmur3_hash32`、`murmur3_uniform32`（split-16 转换）、`_uniform64_from_random53`、`murmur3_uniform64`、`_log1p_neg_stable`；`gumbel_noised_argmax` 把 `tl.randint` + `tl_rand32/64` 换成 `murmur3_uniform32/64`，fp32 路径以 `_log1p_neg_stable(u)` 替代 `tldevice.log1p(-u)`。 Compatibility Impact：seeded non-greedy 请求的确定性 token 序列会变，目标 categorical distribution、public API、logits-cache、greedy path、kernel launch structure 不变。

## 关键改动

```text
vllm/v1/worker/gpu/sample/gumbel.py: 新增 Murmur3 hash/uniform/degree-8 log1p kernel，gumbel_noised_argmax 改用之，保留 Philox helper 供 rejection sampling
tests/v1/worker/test_gpu_gumbel_sample.py: known-answer、fp64 endpoint、distribution 与 drafting-stream 测试
tests/entrypoints/llm/test_struct_output_generate.py、tests/entrypoints/openai/chat_completion/test_include_reasoning.py、tests/entrypoints/openai/completion/test_completion.py: 三个 RNG-sensitive entrypoint 测试的确定性调整

# 机制摘录 (gumbel.py: murmur3_uniform32)
def murmur3_uniform32(seed, pos, offset):
    random32 = murmur3_hash32(seed, pos, offset)
    hi16 = (random32 >> 16).to(tl.int32)
    lo16 = (random32 & 0xFFFF).to(tl.int32)
    return (
        hi16.to(tl.float32) * 1.52587890625e-05
        + (lo16.to(tl.float32) + 0.5) * 2.3283064365386963e-10
    )
```

## 性能声明

Serving A/B（RTX 5090，Qwen2.5-1.5B-Instruct vocab 151,936，Model Runner V2，seeded non-greedy temperature=0.8/top_p=0.95，vLLM `0.17.2rc1.dev3682+g426e59f62`、PyTorch 2.11.0+cu130、CUDA 13.0；5 runs/revision、128 requests/run、256 input+128 output、concurrency 32、fixed seed、ignore_eos，640/640 成功）：Request throughput 49.22→51.26 req/s (+4.15%)、Output throughput 6,299.86→6,561.30 tok/s (+4.15%)、Mean TTFT 71.40→75.99 ms (+6.43%)、Mean TPOT 4.524→4.291 ms (-5.15%)、Mean ITL 4.527→4.311 ms (-4.79%)、P99 TTFT 91.21→103.83 ms (+13.84%)、P99 TPOT 6.817→5.720 ms (-16.09%)、P99 ITL 8.880→8.626 ms (-2.86%)。正文明确主机与无关 8-GPU workload 共享（per-run CV 11.5%/7.4%），「used as a no-material-regression check, not as a clean speedup claim」。Kernel 微基准（2026-08-08 开发期采集，早于最终 fp32-uniform/log/fp64-endpoint 改动，「not presented as final-head performance numbers」，绝对 latency 不可跨设备比）：RTX 5090（10 warmups/100 rounds）1×151,936 为 0.0100→0.0100 ms (1.00x)、64×151,936 为 0.0408→0.0301 ms (1.35x)、128×163,840 为 0.0755→0.0508 ms (1.49x)；Ascend A3 最大为 128×163,840/20 candidates 160.4597→1.7982 ms (89.23x)。正文注明 current-head rerun 仍需要。统计检验：Split-16 fp32 reconstruction 对 2,000,000 个 uint32 输入 0 mismatches；Joint RNG Pearson z mean -0.009/std 0.998/max |z| 3.90；Monte Carlo（vocab 129,280、100,000 draws）tail-mass z 为 -1.18/-1.09（早期 24-bit 路径 -5.89/-5.16）。

## 验证与回归风险

Local commit hooks（Ruff check/format、typos、mypy、SPDX、import、policy）、Python 3.12 mypy、compileall、git diff --check 通过；Buildkite CI #86309 零失败 selected jobs，NVIDIA H200 与 AMD MI300 basic-correctness 及 selected NVIDIA/AMD entrypoint integration lanes 通过（aggregate status 因 optional manual gates 仍 open）。Model-level smoke：每 revision 两次无 prefix-cache 运行均完成 34/34 HTTP 请求，head 的 16/16 sequential prompt/seed 对与 seeded n=4 完全复现，baseline 的 n=4 未完全 identical；base/head 输出序列如预期改变（仅 1/16 first-pass 匹配）。Method note：runtime image 早于 `is_drafting` call-site 更新，两变体均用 `is_drafting=False` harness adapter，故只验证正常 non-drafting 路径，drafting/speculative 不在范围。风险：seeded non-greedy 的确定性 token 序列变化会打破依赖特定 seeded sequence 的假设；Philox helpers 仍保留供 rejection sampling；Ascend/NVIDIA 微基准非 final-head 且需 rerun。

## 关联

PR 正文明确提及 Related RFC #54143，并声明不是 #47524 的重复（后者保留 Philox 并组合更多输出位提升 uniform resolution）；CI 结果为 Buildkite #86309。其它同分类 PR 编号未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
