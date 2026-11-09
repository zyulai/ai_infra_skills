# PR #56050: [Bugfix][Quantization] Detect NVFP4 in ModelOpt mixed-precision checkpoints

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56050](https://github.com/vllm-project/vllm/pull/56050) |
| 作者 | stecasta |
| 合并日期 | 2026-10-02 |
| labels | bug, ready, torch.compile, quantization |
| 热度 | 👍 0 · 💬 18 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
ModelOpt MIXED_PRECISION checkpoint 的 NVFP4 未被识别,fuse_act_quant 全程关闭少一个融合 kernel;按 per-layer quant_algo 判定修复。

## 问题现象
`nvidia/Qwen3.8-27B-NVFP4`、`RadixArk/Qwen3.8-27B-NVFP4`(quant_algo: MIXED_PRECISION,208 个 FP8 层 + 193 个 NVFP4 层)上 `is_nvfp4_quantized()` 返回 False:由此 `fuse_act_quant` 在所有优化级别为 False,`ActivationQuantFusionPass` 不运行,fused op `silu_and_mul_nvfp4_quant` 不可达(除 Llama、Nemotron-H 外的模型没有手动 act+quant 融合,该 pass 是它们唯一通道),SiLU-and-mul 与 `scaled_fp4_quant` 以两个 kernel 运行、bf16 中间量往返 DRAM。显式传 `--quantization modelopt_fp4` 也无效(仍解析为 modelopt_mixed)。实测 DGX Spark (GB10, sm121)、`vllm/vllm-openai:v0.28.0`:kernel 级 M=2048 每次 forward 90.4 -> 48.3 ms、M=8192 347.9 -> 186.9 ms;e2e TTFT(`fuse_act_quant`)M=2048 -30.8%、M=1024 -6.7%;decode ITL 各并发持平或略好。修复后对 `nvidia/Qwen3.8-27B-NVFP4` config.json 谓词由 False 变 True。

## 根因
正文披露:`is_nvfp4_quantized()` 只检查解析出的 quantization method 是否为 `modelopt_fp4`;混精度 checkpoint 声明 `quant_algo: MIXED_PRECISION`,被 `ModelOptMixedPrecisionConfig.override_quantization_method` 认领后 `self.quantization` 变为 `modelopt_mixed`,检查返回 False。

## 修复方案
`vllm/config/model.py` 新增 `_modelopt_mixed_has_nvfp4()`,从 `quantized_layers` 的 per-layer `quant_algo` 判定;`is_nvfp4_quantized()` 在 `modelopt_mixed` 分支调用它。`W4A16_NVFP4` 故意排除(仅权重量化,无 activation 量化可融合):

```python
def _modelopt_mixed_has_nvfp4(quant_config: dict[str, Any] | None) -> bool:
    layers = (quant_config or {}).get("quantized_layers")
    if not isinstance(layers, dict):
        return False
    return any(
        isinstance(info, dict) and info.get("quant_algo") == "NVFP4"
        for info in layers.values()
    )
```

新增单元测试 `tests/config/test_modelopt_mixed_nvfp4.py`(NVFP4 混其他算法、无 NVFP4 层、W4A16_NVFP4、缺失/畸形 config)。

## 影响与回归风险
- 影响窗口:正文未提及具体受影响版本范围。
- 回归面:`is_nvfp4_quantized()` 判定路径,进而影响 fusion pass 启用;非 mixed 的 NVFP4/compressed-tensors 路径不变;不引入或修改 kernel,只让既有默认 fusion pass 应用于此前被跳过的 checkpoint 类(正文说明未跑完整精度评测,greedy smoke 两臂输出一致)。

## 关联
- #55367(同一函数的 compressed-tensors 分支修复;非重复,但 diff 会文本冲突,后落地者需 trivial rebase)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
