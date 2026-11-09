# PR #36631: [Sampling] Support sampling masks with overlap scheduling

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ByronHsu |
| 合并 | 2026-09-10 |
| 分类 | Overlap / Scheduler |
| Label | documentation, run-ci, apple-silicon, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计);diff 有省略(14 个文件未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/36631 |

## 一句话结论

让 sampling mask 兼容 overlap 调度:去掉 sampler 内的 CPU 同步,bs64 解码吞吐 +51.82%。

## 动机

Sampling masks 此前不支持 overlap scheduling:在 sampler 内部物化 mask(把 support 索引、长度、logprob 逐行搬回 CPU)会强制 GPU 与 CPU 同步,打断 overlap 调度流水的异步执行。

## 优化点

核心改动是把 mask 结果保留在 GPU 上:sampler 不再逐行 `.cpu().tolist()` 物化 Python list,而是构建固定容量的 packed 设备端 `SamplingMaskOutput`(token_ids/lengths/selected_logprobs/statuses),异步拷回 CPU,推迟到 result processing 阶段再物化响应。配套引入 `--sampling-mask-max-tokens`(默认 4096)作为每个 mask 的容量上限:top_k 超限的请求在入口直接拒绝(HTTP 400);realized mask 超容量(含 cutoff ties 导致超出 top_k)标为 OVERFLOW 并在提交该步 token 前 abort;selected-token logprob 非有限(NaN/inf)标为 INVALID,同样提前 abort。多副本场景下 statuses 通过 TP/CP 组内 all_reduce(MAX)保证各副本做出一致的 abort 决策。此外用 server argument 取代 `SGLANG_DISAGGREGATION_SAMPLING_MASK_MAX_TOKENS`,PD 场景 metadata 默认关闭,需在 prefill/decode 两侧显式设 `SGLANG_ENABLE_DISAGG_SAMPLING_MASK=1`,未使用时不分配、不传输 metadata buffer。

## 关键改动

```text
python/sglang/srt/layers/sampler.py: _build_sampling_mask_output,GPU 上 packed 固定容量输出 + 状态编码
python/sglang/srt/layers/logits_processor.py: 新增 SamplingMaskOutput/SamplingMaskStatus 数据结构
python/sglang/srt/managers/scheduler.py: 入口校验(top_k/容量)与 _reject_sampling_mask_request
python/sglang/srt/managers/scheduler_components/batch_result_processor.py: 结果处理阶段物化 mask 并处理 abort
python/sglang/srt/disaggregation/utils.py: PD metadata 改为 SGLANG_ENABLE_DISAGG_SAMPLING_MASK 开关
```

## 性能声明

Qwen3-8B BF16、TP1、单卡 NVIDIA H200,FlashInfer sampling 与 attention,CUDA graphs 开启;每请求 128 输入/512 输出 token,temperature=1,top_k=4096,top_p=1,return_sampling_mask=true,`--sampling-mask-max-tokens=8192`。Before 为 upstream 756d0e0a85(overlap 关闭),After 为 PR cb30f7c6a9(overlap 开启),测的是实现与调度的合并效果。三次运行取中位稳态解码吞吐:bs 1:155.47 到 182.37 tokens/s(+17.30%);bs 16:1659.03 到 2216.61(+33.61%);bs 64:3179.15 到 4826.58(+51.82%)。

## 验证与回归风险

作者报告 240 个受影响单测与 35 个 subtest 通过;Qwen2.5-0.5B-Instruct 上 TP2/PP1 实机 sampling-mask 测试(含 overlap 与真实 status collective);合并后的分布式用例 29 项在 249.286 秒内全部通过;修复前测试曾复现 offload 期间 KV slot 提前复用、prefill abort 丢失错误类型等问题。CI 106 通过/25 失败,作者归因于 MLX fixture、AMD/NVFP4、Xeon 与 NPU Wan 等既有问题,无 sampling mask 回归。回归面:sampler 输出结构变化影响 mask 消费方;TP/CP>1 时新增 statuses all_reduce 集体通信;PD metadata 默认关闭属行为变化;启动时显式拒绝旧环境变量。

## 关联

PR 正文提到将无关的 MLX 测试修复拆至 #38739;同区域为既有 sampling mask 功能与其 PD(disaggregation)支持路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
