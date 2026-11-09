# PR #34718: [torch.compile] Turn on silu+fp4 quant fusion by default for O1+

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ProExpertProg |
| 合并 | 2026-02-18 |
| 分类 | Quantization(major) |
| Label | performance, ready, torch.compile, nvidia |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 与文件数);diff 内容可见 1 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/34718 |

## 一句话结论

对 O1+ 编译级别默认启用 SiLU-Mul + NVFP4 量化 pointwise 融合(NVFP4 模型自动命中),减少一次中间张量读写。

## 动机

FP4 量化在 Inductor 视角永远是 custom op,而 `enable_act_fusion` 原先只看 `silu_and_mul`/`quant_fp8` custom op 是否启用,导致 NVFP4 模型的 silu+quant 融合默认不开——尽管这是低编译开销、收益稳定的 pointwise 融合。作者另注明:该融合对 MoE 模型暂无效,因为 fused kernel 已在 #31832 中手动调用(另见 #31985 的 wrap 讨论)。

## 优化点

`vllm/config/vllm.py` 的 `enable_act_fusion` 增加第三个条件:模型为 NVFP4 量化(`cfg.model_config.is_nvfp4_quantized()`)时也启用 act fusion。效果是 O1 及以上级别下,NVFP4 模型的 SiLU+Mul 与 FP4 quant 自动融合为单个 custom op;非 NVFP4 模型行为不变。无新功能,仅默认行为变化。

## 关键改动

```text
vllm/config/vllm.py: enable_act_fusion 增加 is_nvfp4_quantized() 条件
```

```python
return (
    cfg.compilation_config.is_custom_op_enabled("silu_and_mul")
    or cfg.compilation_config.is_custom_op_enabled("quant_fp8")
    or (cfg.model_config is not None and cfg.model_config.is_nvfp4_quantized())
)
```

## 性能声明

nvidia/Llama-3.1-8B-Instruct-NVFP4(tp=1):gsm8k(limit 100,5-shot)flexible 0.63 / strict 0.61,与 main 完全一致。startup(秒,p50):冷启动 total 40.529 vs 37.967(+6.75%),冷 compile 12.949 vs 12.71(+1.88%),warm total 20.585 vs 24.659(-16.52%),warm compile 3.451 vs 3.213(+7.41%)——warm 路径更快但冷启动编译略慢。TPOT/TTFT E2E 数据以截图给出,文字数字未注明。nvidia/Llama-3.3-70B-Instruct-NVFP4(tp=4):startup 基本持平(冷 -0.66%、warm -0.6%),TPOT/TTFT 亦为截图;70B 精度评测标注 "Eval TBD"。硬件型号未注明(仅 tp 配置)。

## 验证与回归风险

测试为 E2E perf sweep、startup sweep 与 lm_eval(8B);diff 无新逻辑。回归面:所有 NVFP4 量化模型在 O1+ 下默认走融合路径,编译冷启动开销小幅增加(8B 冷启动 total 约 +2.6s);MoE NVFP4 模型不受益(fused_moe kernel 已手动调用)。

## 关联

正文引用 #31985(fused_moe wrap)、#31832(fused kernel 手动调用);同分类 NVFP4 路径相关 #65230。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
