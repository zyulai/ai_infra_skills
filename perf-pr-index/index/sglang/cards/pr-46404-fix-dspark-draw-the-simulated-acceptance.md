# PR #46404: [Fix] dspark: Draw the simulated-acceptance bonus at the simulated row

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @weireweire |
| 合并 | 2026-10-27 |
| 分类 | Speculative Decoding(major) |
| Label | jit-kernel |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/46404 |

## 一句话结论

修正 DSpark 模拟接受模式下 bonus token 取自错误 verify 行的问题,消除模拟基准吞吐虚高(约 2.7%)。

## 动机

`SGLANG_SIMULATE_ACC_LEN` 开启时,DSpark 把接受长度替换为模拟值 `c`,但 bonus token 仍取自真实接受计算所用的行:真实首个 mismatch 行的 target argmax,或真实 reject 行的 rejection-sampling token。该 token 在 `drafts[:c]` 之后提交,落在 target 从未为其预测的位置,同时是下一个 draft block 的锚点。结果提交文本退化,verify batch 路由到的 expert 数少于同等接受长度下的真实验证,模拟接受的吞吐被高估。此前唯一正确的配置是 `SGLANG_SIMULATE_ACC_GREEDY=1` + 纯 temperature 请求(#45433 起 opt-in)。

## 优化点

模拟模式下 bonus 改为在 verify 行 `c` 上按请求的采样律抽取。`build_accept_outs` 在无 sampled bonus 时取行 `c` 的 argmax(覆盖 greedy batch 与 folded greedy accept);`sample_simulated_bonus` 对 top-k/top-p batch 从 `build_speculative_verify_target_probs` 的 target 分布采样行 `c`,纯 temperature batch 沿用 Gumbel kernel;`static_accept_tail` 的 Triton kernel 与 torch reference 改为在 committed length 处读 argmax-id bonus。sampling-accept region graph 在模拟下保持 eligible,region 内用 `begin_sim_bonus` 暂存的 temperatures 在行 `c` 以 graph-safe sampler 抽 bonus。采样 bonus 在 `DSPARK_ACCEPT_SAMPLE` TP sync 站点广播,保证 `rng` preset 下各 rank 一致。非模拟路径完全不变(新分支均在 `simulated_correct_len is not None` 之后)。

## 关键改动

```text
python/sglang/srt/speculative/dspark_components/dspark_verify.py: build_accept_outs 取行 c argmax;sample_simulated_bonus 支持 top-k/top-p
python/sglang/kernels/ops/speculative/dspark/static_accept_tail.py: Triton/torch tail 从 committed length 读 bonus
python/sglang/srt/speculative/dspark_components/dspark_sampling_accept_graph.py: region 内在行 c 用 sim_bonus 抽样,region 模拟下保持 eligible
python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py: 所有 sampled batch 传 simulate_bonus_sampling_info
```

## 性能声明

GB300,DeepSeek-V4.1-Flash FP4,DSpark block 5,AgentX-style agentic traces,32 concurrent,temperature 1,`SGLANG_SIMULATE_ACC_LEN=3.51`,TP2 EP2,3600 s:模拟(before→after)total 122.2K→118.9K tok/s/GPU(-2.7%),output 697→683 tok/s/GPU,ITL p90 4.40→4.92 ms,p90 interactivity 227.3→203.3 tok/s(-11%)。作者说明 before 的退化文本让 MoE 每步便宜约 1 ms(B=4-10),故降低是预期修正;各为单次运行、不同节点,且基于较早 nightly + eager accept。

TP4 EP1、batch 1、NVIDIA megakernel、`SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH=1`(1800 s):region graph 保持回放(~110k 次/rank)时 interactivity p50/p90 为 976/949 tok/s,模拟步骤回退 eager accept 则为 939/863,region 图化值 4%/10%(p50/p90);与 before(987/954)的剩余 ~1% 差距在文本变化与更短运行时长可解释范围内。

## 验证与回归风险

GB300 单元级检查:greedy/folded/fused tail 三条路径 bonus 从 row-0 变为 row-3 argmax;top-p bonus 分布(200k draws)与 renormalized target 的最大绝对差 0.21→0.0016;region bonus 分布(6000 replays)0.65→0.0099。现有 `test_dspark_accept_tail.py`、`test_dspark_sampling_accept_graph.py`、`test_dspark_sim_draw_ahead.py` 通过。端到端重复率指标(6-token 窗口重复 0.375→0.145、4-gram 重复 0.387→0.197、每 MoE 层 distinct experts 回升至接近真实接受)。风险面:仅影响 `SGLANG_SIMULATE_ACC_LEN` 基准模式,非模拟路径不变;min-p 与非模拟 sampling accept 一样不施加。

## 关联

#45433(此前唯一的正确模拟配置);EAGLE/MTP 与 DFlash 本就从 forced row 取 bonus。同分类其他 PR 本卡素材未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
