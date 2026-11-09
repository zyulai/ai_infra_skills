# PR #28373: [Hardware][AMD][Model] Add Triton MoE tuning support and optimized configs for Qwen3 omni for MI308X

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @sammysun0711 |
| 合并 | 2025-11-10 |
| 分类 | MoE (tier: major) |
| Label | performance, rocm, ready, qwen |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 含 `benchmark_moe.py` 调参空间扩展与新增 MI308X fused MoE 配置 JSON |
| 链接 | https://github.com/vllm-project/vllm/pull/28373 |

## 一句话结论

为 Qwen3-Omni 打通 Triton MoE 调参并提交 MI308X 调优配置,输出吞吐 +18.76% 至 +25.05%。

## 动机

Qwen3-Omni 系列(Qwen3OmniMoeForConditionalGeneration,Thinking/Instruct/Captioner 三个 30B-A3B 变体)的 fused MoE Triton kernel 在 MI308X 上没有调参支持与优化配置:一是 `benchmark_moe.py` 无法从该架构 config 读取专家参数,二是 ROCm 调参空间中 `waves_per_eu` 固定为 [0],缺少 AMD 官方推理优化指南建议的可调范围,导致 MI308X 用户只能跑默认 kernel 参数。

## 优化点

三项改动。其一,`benchmark_moe.py` 为 `Qwen3OmniMoeForConditionalGeneration` 添加分支,从 `config.thinker_config.text_config` 取 num_experts/num_experts_per_tok/moe_intermediate_size/hidden_size,使 `--tune` 可对 Qwen3-Omni 生成 MoE 配置。其二,`get_rocm_tuning_space` 的 `waves_per_eu_range` 从 `[0]` 扩展为 `[0, 1, 2, 4]`,对齐 AMD rocm-for-ai workload 自动调参文档。其三,提交调优产物 `E=128,N=768,device_name=AMD_Instinct_MI308X.json`(triton 3.4.0,覆盖 batch 1-4096 各档,含 BLOCK_SIZE/warps/stages/waves_per_eu/matrix_instr_nonkdim/kpack),由 `benchmark_moe.py --model Qwen3-Omni-30B-A3B-Thinking --tp-size 1 --tune` 在 MI308X + ROCm 7.0.0 上生成。

## 关键改动

```text
benchmarks/kernels/benchmark_moe.py: Qwen3OmniMoeForConditionalGeneration
  参数解析分支;waves_per_eu_range = [0, 1, 2, 4]
vllm/model_executor/layers/fused_moe/configs/
  E=128,N=768,device_name=AMD_Instinct_MI308X.json: 新增调优配置
```

## 性能声明

vLLM v0.11.1rc2,单卡 MI308X TP=1,`VLLM_ROCM_USE_AITER_MOE=0`,关闭 prefix caching,Qwen3-Omni-30B-A3B-Thinking,random 数据集 1000 prompts、并发 64。1024 in/128 out:输出吞吐 747.93 → 888.23 tok/s(+18.76%),Peak 输出吞吐 1792 → 2176(+21.43%),Median TTFT 1262.78 → 1013.85 ms(-19.71%),Median TPOT 74.69 → 61.93 ms(-17.08%),Median ITL 41.90 → 33.88 ms(-19.14%)。128 in/1024 out:输出吞吐 1448.75 → 1811.69 tok/s(+25.05%),Median TPOT 43.05 → 34.38 ms(-20.14%),Median ITL 43.12 → 34.28 ms(-20.50%)。作者总结:输出吞吐 +18.76-25.05%,TTFT 最高 -19.71%,TPOT 最高 -20.14%,ITL 最高 -20.50%;并预期惠及 Qwen3-30B-A3B 与 Qwen3-Omni-30B-A3B 系列(注意:PR 摘要句写 "MI300X",与调优目标 MI308X 不一致,以配置文件与测试命令的 MI308X 为准)。

## 验证与回归风险

验证:`vllm bench serve` 两种 workload 方向各 1000 请求前后对比。风险:配置 JSON 按 E/N/设备名精确匹配(E=128,N=768,MI308X),其他形状或设备不受益;`waves_per_eu_range` 扩大调参搜索空间,重新 `--tune` 的耗时变长;非 Qwen3-Omni 的 ROCm MoE 调参也会考虑新的 waves_per_eu 值,理论上不会变差但依赖调参结果。

## 关联

同 MoE 分类其他 PR 可按索引分类检索;同硬件(ROCm)优化:#60084、#64094。调参基础设施与 #28373 使用同一 `benchmark_moe.py`。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
