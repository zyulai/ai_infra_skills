# PR #33395: [Feat] speculative: Seed rejection-sampling draft proposals for deterministic inference

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ajit283 |
| 合并 | 2026-10-22 |
| 分类 | Speculative Decoding |
| Label | bug, speculative-decoding, run-ci, deterministic, bypass-fail-fast |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 含 10 个文件,另有 1 个文件因超长被丢弃 |
| 链接 | https://github.com/sgl-project/sglang/pull/33395 |

## 一句话结论

给 EAGLE/NEXTN rejection sampling 的 draft proposal 采样加 request 级确定性种子,修复 `--enable-deterministic-inference` 下输出随 batch 形状与 CUDA graph replay 变化的问题。

## 动机

classic EAGLE/NEXTN rejection sampling 的 draft proposal 用 `fast_sample()` 的全局 PyTorch RNG(`exponential_()`)抽取;虽然 target sampling 与 verify 侧 rejection coin 已支持 request-local seed,这条 proposal 路径仍是全局 RNG,导致开启确定性推理后输出仍随 batch 形状和 CUDA graph replay 改变——修复前 Qwen3.5-9B NEXTN rejection sampling 在 12 次变 batch 试验中产出 12 个不同输出。此前 rejection sampling 与 `--enable-deterministic-inference` 组合被直接禁止(启动时报 ValueError),本 PR 是 #30822 记录的 draft-worker 后续。

## 优化点

`fast_sample()` 增加可选确定性路径,随机性按四元组(请求 sampling seed、绝对序列位置、draft step/MTP layer、vocabulary entry)键控:draft step 编码进 uint32 列块(`draft_step * vocab_size` 列偏移),seed 加域分隔盐值 `_DRAFT_GUMBEL_SEED_SALT` 与 target sampling / verify coin 流分离(保持 rejection sampling 所需独立性);`murmur_hash32` 输出经 midpoint binning 映射到开区间 (0,1) uniform,再取负对数得 Exp(1) 变量,以 float64 做 exponential-race categorical 采样(argmax(probs/Exp)),避免 hash 端点产生无穷值。seed 为 None 时保留原 RNG 快路径(`SGLANG_OPT_USE_GUMBEL_SAMPLE=0` 回退 multinomial)。seed 与 position 经 eager 与 CUDA graph 两条 EAGLE/NEXTN worker 路径贯通:draft cuda graph runner 与 multi-layer 单图 draft-extend runner 增加 `sampling_seed` buffer,prepare 时从 sampling_info 拷贝;支持普通与 multi-layer EAGLE(含单 CUDA graph multi-layer draft-extend)。确定性 rejection sampling 因使用 Triton hash 内核而限制在 CUDA 兼容设备;启动校验从"禁止组合"改为"非 CUDA 设备报错"。

## 关键改动

```text
python/sglang/srt/speculative/spec_utils.py: fast_sample 增加 seeded 路径(murmur_hash32 -> float64 exponential-race),域分隔盐值
python/sglang/srt/speculative/eagle_worker_v2.py: draft_forward/_draft_extend_for_prefill/_draft_extend_for_decode 传入 seed/position/draft_step
python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py: 增加 sampling_seed 静态 buffer 并在 execute 时拷贝
python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py: 单图 multi-layer draft-extend 的 seeded 采样
python/sglang/srt/arg_groups/speculative_hook.py: 组合校验从禁止改为仅非 CUDA 报错
```

## 性能声明

确定性路径有可量化的开销(B200,vocab 163,840,CUDA-graph replay,`bench_deterministic_fast_sample.py`):B=1 31.9 us -> 47.2 us(1.48x 慢);B=8 38.0 -> 63.5 us(1.67x);B=32 54.5 -> 131.9 us(2.42x);**B=128 138.9 -> 544.1 us(3.92x 慢)**。峰值显存(B=128,vocab 163,840):input probabilities 80.00 MiB,baseline allocated 80.00 MiB,peak allocated 321.25 MiB,增量 241.25 MiB。确定性模式为 opt-in;PR 提出若端到端 profiling 显示 hash/score 矩阵物化影响吞吐,可后续考虑 fused hash/exponential/argmax 内核。

## 验证与回归风险

精度修复数字:Qwen3.5-9B(TP2,NEXTN,rejection sampling,temp 1.0,seed 42)修复前 12 trials 12 个唯一输出,修复后 12 trials 1 个;B200 post-rebase 7 passed;multi-layer EAGLE(Step-3.5-Flash,TP8,8xB200,FA4,3 MTP 层,单图 draft extend)6 samples 1 个唯一输出。新增单测覆盖 replay 位级一致、batch 置换/切分不变性、seed/position/step 流分离、hash 端点有限性、CUDA graph 输入更新、categorical 分布 sanity(100,000 次 seeded 采样,[0.1,0.2,0.7] 频率误差 <0.01);两个新 E2E 确定性 smoke 测试;`test_block_verification_args.py` 更新为 CUDA 上允许组合。风险面:float64 分数矩阵物化在大 batch/大 vocab 下有 3.92x 采样延迟与 241 MiB 增量显存;`row_argmax` 仅 FP32,seeded float64 路径走普通 argmax;draft_step*vocab_size 超 2^32 时报错;非 CUDA 设备被拒绝。

## 关联

前序:#30822(verify 侧 seeded rejection coins,draft 侧的 follow-up)。同分类:PR #27988 无关;EAGLE/NEXTN 确定性采样与 block verification 共用 seeded draft/verify draws。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
