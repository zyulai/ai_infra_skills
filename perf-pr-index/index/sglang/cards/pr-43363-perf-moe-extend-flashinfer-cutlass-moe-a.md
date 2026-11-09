# PR #43363: [Perf] moe: Extend FlashInfer CUTLASS MoE autotune coverage to prefill chunks

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JackChuang |
| 合并 | 2026-10-20 |
| 分类 | MoE(tier: major) |
| Label | (none) |
| Diff | 素材未提供 diffstat |
| 链接 | https://github.com/sgl-project/sglang/pull/43363 |

## 一句话结论

把 FlashInfer CUTLASS MoE 的启动 autotune 范围从 decode batch 扩大到 prefill chunk,H200 上 8k 提示 TTFT 降低 30.3–57.8%。

## 动机

sglang 启动时已对 FlashInfer 的 `cutlass_fused_moe` 做 autotune(#12306,默认开启于 #14357),但两个调用点都传 `next_power_of_2(x.shape[0])`(承自 #7327),而启动批是 decode 尺寸,因此只有 decode 尺寸拿到 tuned tactic,更大的 prefill chunk 只能跑 FlashInfer 未调优的 fallback tactic。此外 `MoERunner.refine_tuning_config` 带 `lru_cache`,桶列表按"最近一次新 token 数"重建,导致运行期 tactic 随先前请求漂移(如 65–128 请求的 decode 批会把后续 8k chunk 压到 128-token bucket)。flashinfer#5625 即用户同时踩中这两个问题的报告。

## 优化点

新增 `_cutlass_moe_tune_max_num_tokens(num_tokens)`:全服务统一取 `min(prefill chunk size × attention DP size, 8192)`(8192 为 FlashInfer 该调用自身的默认 tune 范围上限),并让 `flashinfer_cutlass` runner 与 `flashinfer_mxfp4` CUTLASS 路径两个调用点都用它作为 `tune_max_num_tokens`,使同一次启动 autotune 同时覆盖 decode 与 prefill 尺寸。不新增 flag、不新增 kernel、不加 tuning pass;机制上仅替换同一 GEMM 的 tile 配置(tactic)。PD decode server 保留原按调用取值逻辑;大于 8192 的 chunk 复用 bucket 8192 的 tactic(单层实测不慢于按各自尺寸调优)。

## 关键改动

```text
python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py: 新增 tune 范围 helper,替换两个 cutlass_fused_moe 调用点的 tune_max_num_tokens
test/registered/unit/layers/moe/test_flashinfer_cutlass_swiglu_params.py: 新增 TestCutlassMoeTuneMaxNumTokens(80 sub-cases,60 个在 main 失败)
test/registered/unit/layers/quantization/test_mxfp4_sm{90,120}_cutlass.py、test_moe_runner_config_placement.py: 测试改在 get_context().override_server_args() 下运行
```

```python
def _cutlass_moe_tune_max_num_tokens(num_tokens: int) -> int:
    if get_disagg().disaggregation_mode == "decode":
        return next_power_of_2(num_tokens)
    prefill_tokens = max_prefill_buffer_tokens() or get_schedule().max_prefill_tokens
    return min(
        prefill_tokens * deployment_attn_dp_size(),
        _FLASHINFER_DEFAULT_TUNE_MAX_NUM_TOKENS,
    )
```

## 性能声明

H200、CUDA 13.0、torch 2.14.1、FlashInfer 0.7.0.post1,`--chunked-prefill-size 8192`,bench_serving 随机数据集。七个 MoE 模型(DeepSeek-V4.1-Flash、Nemotron-3.5、Ling-3.0、gpt-oss-20b、GLM-5.3-Flash、MiniMax-M3、Gemma-4)全部改善:8k 提示 TTFT 30.3–57.8% 更低(DeepSeek-V4.1-Flash 573.9→400.0 ms,8×H200)、32k TTFT 23.0–52.8% 更低、prefill throughput 43.4–143.4% 更高;三个受测模型 decode ITL 不变。单次 MoE 调用:8192 tokens 下 tuned tactic 1.514 ms 对 fallback 3.990 ms,即 2.6× faster。与 #42736 合用再降 8k TTFT 24.7%(对比 #42736 alone)。代价:无缓存时启动 autotune 变长(DeepSeek-V4.1-Flash 41–42→176–177 s,Ling-3.0-flash 116→604 s),每卡常驻显存少 0.4–1.2 GiB(0.3–0.9% H200),KV cache 大小不变。

## 验证与回归风险

新增 CPU 单测 `TestCutlassMoeTuneMaxNumTokens`,stub 掉 kernel,在 5 种服务器配置下校验 token 数 1–16384 都映射到启动 autotune 调优过的 bucket,并验证 PD decode server 保持按调用取值;若干既有 mxfp4/moe_runner_config_placement 测试需包在 `override_server_args` 中。影响面:该 kernel 所在全部 GPU(SM90/SM100/SM110/SM120)及默认走它的模型(DeepSeek-V4.1/V4、Ling-3.0、Nemotron-H、gpt-oss、ModelOpt NVFP4 MoE);启动变慢与显存小幅下降是已知成本,精度设计上不受影响,GSM8K 七模型均在 95% 噪声带内(−0.83 至 +0.15 分)。PD decode server 与其他 MoE kernel 不受影响。

## 关联

互补于 open PR #42736(额外启动 tuning pass,两者作用于同一批 bucket);前序:#7327(引入 `next_power_of_2` 取值)、#12306 与 #14357(启动 autotune 机制)、#41765(SM110 ModelOpt NVFP4);外部关联 flashinfer#5625。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
