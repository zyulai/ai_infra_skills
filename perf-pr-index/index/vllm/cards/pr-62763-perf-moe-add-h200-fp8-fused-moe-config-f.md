# PR #62763: [Perf][MoE] Add H200 FP8 fused MoE config for Beam-1.0 TP8

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ischencheng |
| 合并 | 2026-10-26 |
| 分类 | MoE |
| Label | performance, ready, beam |
| Diff | diffstat 缺失(素材未记录 +/- 行数与文件数);diff 可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/62763 |

## 一句话结论

为 Beam-1.0 FP8 在 H200 TP8 的 fused MoE 形状(E=256, N=320)新增调优 config:uniform 路由 96–1536 tokens kernel 快 9–45%,128/256 请求 decode step -9.4%/-16.7%。

## 动机

Reflection Beam-1.0 FP8 按 recipe 在 H200 上跑 TP8 + `--moe-backend triton`,其 fused MoE 形状(E=256, N=320, fp8_w8a8)没有调优 config,所有层都用默认 config;uniform 路由下 1024 tokens 时 kernel 仅达 HBM bound 的 31%,中等 batch decode 明显欠优。Follow-up to #62074。

## 优化点

1) 新增 `E=256,N=320,device_name=NVIDIA_H200,dtype=fp8_w8a8.json`:覆盖 1–16384 tokens,`BLOCK_SIZE_K=128`;≤64 tokens 与 ≥2048 tokens 的条目保持与默认相同,使小 batch decode 与 prefill 不受影响(DFlash2 recipe 7 draft tokens 下 verify step 每请求 8 行,12–192 请求落在受益区间)。2) `benchmark_moe.py` 支持 per-token activation / per-channel weight FP8 路径:compressed-tensors checkpoint 的 `input_activations.strategy == "token"` 时,`benchmark_config` 以 `per_act_token_quant=True` 计时模型实际运行的 kernel 变体(此前总测 per-tensor FP8;两种路径读同一 config 文件名),并新增 `BeamForCausalLM` 解析。调优搜索空间被裁剪(`BLOCK_SIZE_M` 按 token 数分档、`GROUP_SIZE_M ∈ {1,16,64}`)以保持单次调优约半小时。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/configs/E=256,N=320,device_name=NVIDIA_H200,dtype=fp8_w8a8.json: 新调优 config
benchmarks/kernels/benchmark_moe.py: per_act_token_quant 路径与 BeamForCausalLM 解析
```

```json
"1024": {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 16,
    "num_warps": 4,
    "num_stages": 3
}
```

## 性能声明

- Recorded routing(1xH200,top-k ids 取自 Beam-1.0 FP8 真实权重前 6 层,TP8 形状 E=256/N=320/K=5120/top-8/per-token FP8,每 M 24 步于一个 CUDA graph、7 次 replay 取中位):code corpus M=256 384.7→311.4 µs(−19.1%),M=96–512 快 3–19%;wikitext M=256 −18.2%;uniform random 路由 M=96–1536 快 9–45%。
- Uniform 路由 kernel(`benchmark_config` per-token 路径):M=1024 856.7→470.1 µs(−45.1%,达 HBM bound 的 31%→57%);M=1536 −42.4%;M=96 −8.7%。
- 单个 TP8 rank(52 层全量、dummy weights、`VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random`、`--moe-backend triton`、kv fp8;两臂同容器按 default→tuned→tuned→default 顺序,各 4 次取中位,arm 内 spread <0.3%):decode 128 req 25.95→23.51 ms(−9.4%),256 req 31.67→26.38 ms(−16.7%),16 req −0.1%,8K prefill chunk 0.0%。
- 1–64 tokens:uniform 下 −4%~+0.3%,recorded code corpus 64 处 +1.9%,故保留默认 config。

## 验证与回归风险

正确性:仅 Triton tile 参数变化;Beam shape 在 M=96/128/256/1024/1536 下,新旧 config 输出对 `tests/kernels/moe/test_triton_moe_ptpc_fp8.py` 的 per-column reference 平均相对误差均为 1.41–1.44%(阈值 5%),两 config 同为 `BLOCK_SIZE_K=128` 且输出相同;`benchmark_moe.py --model reflection/Beam-1.0-501B-A23B-FP8 --tp-size 8` 在无 config 的 L4 上可运行并打印 per-token 提示。风险:未覆盖 H100、TP4(N=640)、DFlash2 verify 与完整 503 GB 模型(需 8xH200,未跑);recorded routing 仅来自 3 个早期 MoE 层,后续层可能不同;专家负载分布敏感——dummy weights 不加 uniform 模拟时该 config 在 128/256 请求反而慢 9.8%/13.0%;per-tensor 与 per-token FP8 共享同一 config 文件名(该形状下调优结果两者差异 <1%)。

## 关联

Follow-up to #62074;相关的 open `benchmark_moe.py` PR:#62526/#49088/#50711/#51353(模型解析/错误处理)、#57100(zipf 偏斜专家负载)、#61688(runtime Triton autotune,draft)。索引同区域另有 pr-28892(trtllm MoE NVFP4 kernel)、pr-36022(FlashInfer MoE A2A)、pr-46340(batched MoE operand loads)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
