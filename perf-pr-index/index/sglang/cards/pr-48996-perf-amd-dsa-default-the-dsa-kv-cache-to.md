# PR #48996: [Perf] [AMD] dsa: Default the DSA KV cache to fp8_e4m3 on gfx950

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @xiaofei-zheng |
| 合并 | 2026-11-12 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, run-ci, run-ci-amd-extra |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/48996 |

## 一句话结论

gfx950(MI355X)上把 DSA KV cache 的 `auto` 默认解析为 fp8_e4m3(AITER 开启且 DSA backend 为 triton/tilelang 时),启用 gfx950-tuned sparse-MLA fp8 路径;MI355X 实测 DeepSeek-V3.2 TP8 输出吞吐 +5.9% 至 +11.9%,gsm8k/RULER2 精度统计持平。

## 动机

`_dsa_kv_cache_dtype_default` 只在 device capability major ≥ 10 时把 `--kv-cache-dtype auto` 解析为 `fp8_e4m3`(#11655);MI355X(gfx950)报告 9.5,于是 DeepSeek-V3.2 与 GLM-5.x 在 ROCm 上默认拿到 bf16 KV cache。HIP 默认 DSA backend 是 triton(#30575),其 gfx950-tuned sparse-MLA 路径(`_is_gfx950_sparse_mla_fp8`)只在 fp8 KV cache 下运行。GLM-5.3 MI355X cookbook 已要求手动 `--kv-cache-dtype fp8_e4m3`(BF16 KV 拖慢 DSA prefill);本 PR 让不带该 flag 的启动取得同样效果。

## 优化点

- `overrides.py`:`auto` 在 HIP 且 `is_gfx95_supported()`、`SGLANG_USE_AITER=1`、且两个 DSA backend 均为 unset/`triton`/`tilelang` 时也解析为 `fp8_e4m3`。Learnable sinks、`aiter` backends、gfx942、`SGLANG_USE_AITER=0` 保持 bf16;显式 `--kv-cache-dtype` 按传入使用。aiter 条件是必需的:没有 aiter 时 prefix-hit MHA 路径会走到 `dequantize_k_cache_paged_dsa`,它拒绝 576-wide 的 fp8 行。
- `test_model_overrides.py`:新增 gfx950 各分支用例(新默认 + 每个保持 bf16 的情形);新用例在 main 上失败(`bfloat16 != fp8_e4m3`)。
- `hisparse_guide.mdx`:`auto` 默认的说明更新为覆盖 MI355X 与 opt-out(`--kv-cache-dtype bfloat16`)。
- 注意事项(正文):gfx942 prefill + gfx950 decode 的 PD setup 现在需要显式 `--kv-cache-dtype`(bootstrap 拒绝不匹配的 KV dtype)。

## 关键改动

```text
python/sglang/srt/arg_groups/overrides.py: _dsa_kv_cache_dtype_default 增加 hip_gfx95_fp8 条件(major >= 10 or hip_gfx95_fp8)
test/registered/unit/test_model_overrides.py: gfx950 capability (9, 5) 各分支用例(fp8 默认 / learnable_sink / aiter backend / SGLANG_USE_AITER=0 / 非 gfx95 均 bf16)
docs/docs/advanced_features/hisparse_guide.mdx: DSA auto 默认说明更新
```

```python
if kv_cache_dtype == "auto":
    # gfx950 reports capability 9.x, but its triton/tilelang DSA kernels
    # take fp8 KV and the gfx950-tuned sparse-MLA path requires it. The raw
    # fp8 MLA layout is only read back on the aiter path (forward_mha.py).
    hip_gfx95_fp8 = (
        get_platform().is_hip
        and is_gfx95_supported()
        and envs.SGLANG_USE_AITER.get()
        and view.dsa_prefill_backend in (None, "triton", "tilelang")
        and view.dsa_decode_backend in (None, "triton", "tilelang")
    )
    kv_cache_dtype = (
        "fp8_e4m3"
        if (major >= 10 or hip_gfx95_fp8) and not has_attention_sinks
        else "bfloat16"
    )
```

## 性能声明

- 环境:MI355X(gfx950),torch 2.11.0+rocm10.0.0,sglang-kernel 0.4.10,aiter `285579b62`,`SGLANG_USE_AITER=1`,main `14e395aa5e` + 本 patch。`python -m sglang.bench_serving --backend sglang --dataset-name random-ids --random-range-ratio 1.0 --num-prompts 4x concurrency (min 32) --seed 1 --flush-cache`,每臂 fresh server 一次。
- DeepSeek-V3.2,TP8,triton(bf16 臂先跑),输出 tok/s bf16 → PR:1024/1024 c4:329→349(+5.9%),median TPOT 11.89→11.22 ms;1024/1024 c64:2456→2655(+8.1%),TPOT 24.27→22.86 ms;8192/1024 c4:252→267(+6.1%);8192/1024 c64:1183→1323(+11.9%),TTFT 22192→18129 ms。
- GLM-5.3-Quark-MXFP4-AttnFP8,TP4 EP4,triton(PR 臂先跑):1024/1024 c4:333→334(+0.2%);c16:934→982(+5.1%);c64:2282→2330(+2.1%);8192/1024 c16:670→735(+9.8%);8192/1024 c64:1166→1288(+10.4%),TTFT 27444→23911 ms。
- `max_total_num_tokens`(bf16 → PR):DeepSeek-V3.2 TP8 2125952→3855488;GLM-5.3 TP8 1534016→2978880;GLM-5.3 MXFP4 TP4 1605632→3118016。
- pack 的 regex 未检出 perf claim 行(数字均来自正文表格,原样引用)。

## 验证与回归风险

- 精度(sgl-eval 0.1.3 gsm8k 全量 1319 题,temperature 1.0,top_p 0.95,seed 42;配对 McNemar):DeepSeek-V3.2 TP8 triton bf16 95.45% → fp8 95.83%(16/21,p 0.51);GLM-5.3(FP8)TP8 tilelang 97.50% → 96.89%(14/6,p 0.12);GLM-5.3-Quark-MXFP4-AttnFP8 TP4 EP4 triton 97.80% → 97.35%(12/6,p 0.24)。
- Nightly 配置(`sglang.test.few_shot_gsm8k`,各 3 runs):`test_deepseek_v32_eval_mi35x`(TP8 tilelang,5-shot,1319)0.948/0.944/0.946 → 0.947/0.945/0.945(threshold 0.93);`test_deepseek_v32_mtp_eval_mi35x`(EAGLE 3/1/4,20-shot,200 题)0.980 x3、accept length 3.07–3.08 → 3.07–3.10(threshold 0.94 / 2.7);`test_glm53_eval_mi35x`(5-shot,1319)0.926/0.932/0.926 → 0.922/0.922/0.914。
- RULER2(main `62384de1ae` + patch,同环境同 flags,32k context,12 子任务 x 50 样本):DeepSeek-V3.2 87.07% → 86.97%;GLM-5.3 94.04% → 94.85%。
- 其他检查:`--dcp-size 2`(DeepSeek-V3.2 TP8)以 fp8 启动并在 200 题上得 0.960;`SGLANG_USE_AITER=0` 时默认解析回 bf16,且此时 DeepSeek-V3.2 在 decode CUDA graph capture 期间于 DSA indexer RoPE 中失败(两种 dtype 皆然,正文如实记录)。
- 风险:PD(gfx942 prefill / gfx950 decode)需显式 KV dtype;非 gfx95/HIP 平台行为不变(major ≥ 10 的 NVIDIA 路径保持原逻辑)。

## 关联

#11655(`auto` 默认解析的既有实现)、#30575(HIP 默认 DSA backend triton);正文注明 Found by Hyperloom(AMD-AGI/Hyperloom)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
