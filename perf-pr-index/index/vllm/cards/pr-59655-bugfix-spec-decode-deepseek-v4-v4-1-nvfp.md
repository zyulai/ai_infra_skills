# PR #59655: [Bugfix] [Spec Decode] DeepSeek V4/V4.1 NVFP4: keep ignored MTP/DSpark experts on MXFP4

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @v-shobhit |
| 合并 | 2026-10-08 |
| 分类 | Speculative Decoding(major) |
| Label | bug, ready, ci/build, deepseek, nvidia, verified, DSv4, dflash, DSv4.1 |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用(6 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/59655 |

## 一句话结论

修复 DeepSeek V4/V4.1 NVFP4 checkpoint 的 MTP/DSpark 草稿层被误按 NVFP4 加载导致 draft 全拒收的问题,MT-Bench 提速约 2.3×。

## 动机

`nvidia/DeepSeek-V4.1-Flash-NVFP4` 上 DSpark 投机解码接受率为 0(GSM8K 上 604,200 个 draft token 全部被拒,acceptance length 1.00,而原生 checkpoint 为 3.57),投机只增加开销。根因:ModelOpt NVFP4 导出把 `mtp.*`(MTP/DSpark 层)留在 `quantization_config["ignore"]` 中、保持 MXFP4 格式,但 `DeepseekV4FP8Config.get_quant_method` 只要 `moe_quant_algo == "NVFP4"` 就对所有 `RoutedExperts` 层返回 `ModelOptNvFp4FusedMoE`。MXFP4 packed 权重形状与 NVFP4 相同所以能加载,但 block-scale 有一半没被写入,NVFP4 专属的 per-tensor scale 保持 `torch.empty` 未初始化值,草稿 MoE 算出垃圾结果——全程无报错,只有一条 `w1_weight_scale_2 must match w3_weight_scale_2` 警告。

## 优化点

`DeepseekV4FP8Config.from_config` 记录 checkpoint 的 `ignore` 列表;新增 `_is_nvfp4_expert_layer(prefix)`:用正则提取 `layers.<idx>.ffn.experts`,若 idx ≥ num_hidden_layers 则重映射为 checkpoint 名 `mtp.{idx - num_layers}`(草稿层在模型里建成 `layers.{num_hidden_layers + k}`),再用 `fnmatch` 匹配 ignore 列表——匹配中的层保留 `Mxfp4MoEMethod`,其余(target experts、无 ignore 列表、非 NVFP4)行为不变。`moe_quant_algo` 仍读当前 target hf_config,单独发布的 draft(#49133 场景)不受影响。

## 关键改动

```text
vllm/models/deepseek_v41/quant_config.py: from_config 记录 ignore;_is_nvfp4_expert_layer 按 mtp.* 匹配
tests/quantization/test_fp8.py: 7 个 CPU 单测覆盖 target/ignored-mtp/无 ignore/非 NVFP4
tests/evals/gsm8k/configs/DeepSeek-V4.1-Flash-NVFP4-DSpark-TP4.yaml: min_acceptance_length: 2.5 门禁
```

## 性能声明

GB300 TP4,vLLM v0.30.0 挂载本 PR 的 quant_config.py。GSM8K(1319 题 5-shot greedy):acceptance length 1.000 → 3.559(原生 checkpoint 为 3.570),accuracy 0.9166 → 0.8984(作者说明 ±0.015 属 batch-dependent 数值噪声)。MT-Bench(80 prompts,greedy,TP4,三节点):PR vs v0.30.0 输出吞吐 c1 2.19–2.45×(中位 2.34×)、c8 1.90–2.08×(中位 1.98×);c1 TPOT 8.49 → 3.28 ms(native 3.06 ms);与原生 checkpoint 差距 -0.6% 至 -13.1%,作者判定在 run-to-run 噪声内(节点间吞吐波动可达 ~25%)。

## 验证与回归风险

验证:CPU 单测 7 例(v0.30.0 下 2 例失败,本 PR 全过);GSM8K config 加入 Buildkite `models-spec-decode.txt`(B200 "LM Eval Spec Decode" 步骤),`vllm/models/deepseek_v41/quant_config.py` 加入 Quantization 步骤依赖;load log 确认 draft 走 `FLASHINFER_TRTLLM_MXFP4_MXFP8`、警告消失。风险:只有 V4.1-Flash-NVFP4 行跑过端到端,DeepSeek-V4 行由单测覆盖;`*-nvfp4-DSpark` 导出(MTP 层本就 NVFP4)行为不变;FlashInfer `moe_ep` mega-MoE 后端仍在 `fi_moe.py` 全局决定 NVFP4。

## 关联

同区域:#49133(同类症状,单独 shipped drafter,已关闭)、#56054(Quark layer_quant_config 的同款 mtp.{i} 重映射)、#55498、#49900、#43319(均有区别说明,非重复)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
