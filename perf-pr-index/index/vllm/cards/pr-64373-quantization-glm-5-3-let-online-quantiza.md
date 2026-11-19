# PR #64373: [Quantization][GLM-5.3] Let online quantization reach the BF16 attention projections

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @suiyoubi |
| 合并 | 2026-11-18 |
| 分类 | Quantization(tier: major) |
| Label | ready, quantization, glm |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/64373 |

## 一句话结论

GLM-5.3-Flash 把 attention 投影(KDA、MLA、DSA indexer)用 `quant_config=None` 构建,`--quantization-config` 的 online 目标会静默跳过它们;本 PR 把模型 quant config 传给这些模块(唯一例外:FP8/Quark checkpoint 的 MLA 投影,模型在加载时已反量化为 BF16),使 NVFP4 checkpoint 上可量化层从 84(仅 shared experts)扩到 196,GB300 上聚合服务输出吞吐 +2%(非 KDA 配置)、disaggregated prefill-only 最高 +12%。

## 动机

GLM-5.3-Flash 的 KDA `in_proj_qkvbfg_a`/`f_b_proj`/`g_b_proj`/`o_proj`、MLA `fused_qkv_a_proj`/`q_b_proj`/`o_proj`、indexer `wq_b` 在所有官方 checkpoint 都是 BF16(MA 在 Z.ai FP8 checkpoint 中是 FP8,加载时反量化)。模型为此传 `quant_config=None`(MLA)或在 KDA 构造时清空 `vllm_config.quant_config`。此外 Z.ai FP8 config 的 `modules_to_not_convert` 用 `model.layers.N.*` 命名,而 vLLM 权重是 `model.language_model.layers.N.*`,1,509 条 entry 无一匹配——所以共享 quant 代码不变的情况下,`--quantization modelopt_fp4 --quantization-config '{"targets": ...}'` 对 shared experts 生效(84 层)却静默跳过 112 个 attention 投影,且无任何警告。

## 优化点

- KDA 与 MLA 拿到模型的 `quant_config`:每个投影遵循 checkpoint 自己的 skip list(`exclude_modules`、`modules_to_not_convert`),online targets 经现有 `resolve_quant_method` 组合;未命中 online target 的层保持 checkpoint 决定。`kda.py` 删除清空 quant_config 的 try/finally。
- 特例:FP8 序列化的 `Fp8Config` 或 `QuarkConfig` 下 MLA 保持 `quant_config=None`(这些 checkpoint 存 FP8 MLA 投影、模型加载时反量化),其 KDA 仍可达。
- 名字映射:`hf_to_vllm_mapper` 增加 `model.layers.N.self_attn.` → `language_model.model.layers.N.self_attn.`(仅 self_attn:NVFP4 checkpoint 用 `model.layers.45*` 排除 MTP 层,该 entry 必须保持原名);`packed_modules_mapping` 列出 fused KDA `in_proj_qkvbfg_a` 的六个 shard。
- GB300 specialized decode(`decode_backend=latency_specialized`,#63102)直接以 BF16 读 KDA 权重,其 `KdaStage` 现在检查并启动时报错、点名应排除的投影,而不是跑在量化权重上。
- 推荐配置(正文):聚合/decode 实例用 MXFP8 量化 128 个非 KDA 投影(KDA 进/出保持 BF16,行为与 BF16 在 7 个 benchmark 上一致);disaggregated 在 prefill 实例上量化全部 196;全 196 聚合属 opt-in 且正文披露会缩短 reasoning。

## 关键改动

```diff
 # vllm/models/glm5next/common/kda.py
-        # KDA projections remain BF16 because fp8 checkpoints omit their scales.
-        saved_quant_config = vllm_config.quant_config
-        try:
-            vllm_config.quant_config = None
-            super().__init__(config, vllm_config, prefix)
-        finally:
-            vllm_config.quant_config = saved_quant_config
+        super().__init__(config, vllm_config, prefix)
```

```python
 # vllm/models/glm5next/common/model.py(pack diff 摘录)
                 quant_config=(
                     None if _mla_dequantized_at_load(quant_config) else quant_config
                 ),

def _mla_dequantized_at_load(quant_config: QuantizationConfig | None) -> bool:
    """Whether the checkpoint stores the MLA projections in FP8 (FP8 and Quark
    checkpoints), which the model dequantizes to BF16 at load (`bf16_linear`)."""
    return isinstance(quant_config, QuarkConfig) or (
        isinstance(quant_config, Fp8Config)
        and quant_config.is_checkpoint_fp8_serialized
    )

    hf_to_vllm_mapper = Glm4vForConditionalGeneration.hf_to_vllm_mapper | (
        WeightsMapper(
            orig_to_new_regex={
                re.compile(r"^model\.layers\.(\d+)\.self_attn\."): (
                    r"language_model.model.layers.\1.self_attn."
                )
            }
        )
    )
```

## 性能声明

- 聚合(GB300,1 节点 DP2 + expert parallel,sa-bench 8K in / 1K out,`nvidia/GLM-5.3-Flash-NVFP4`):BF16 投影(今天)C512/C1024 output tok/s/GPU 2,839 / 2,441;MXFP8 非 KDA 128 层 2,896(+2.0%) / 2,488(+1.9%);MXFP8 全 196(opt-in)3,075(+8.3%) / 2,641(+8.2%)。mean TTFT 7.8 s / 52.2 s → 7.3 s / 49.5 s → 6.0 s / 42.2 s。
- Disaggregated(2 prefill TP1 + 1 decode DP2+EP,NIXL):MXFP8 全 196 仅 prefill 侧 C128/C256/C512 output tok/s/GPU 1,446(+0.6%) / 2,254(+0.9%) / 3,025(+12.2%),mean TTFT 1.04/1.97/17.5 s → 0.97/1.69/8.7 s;C128/C256 的 sub-1% 差异不显著(单次运行/点),decode TPOT 20.6 / 25.8 ms 两臂相同。
- 精度(NVFP4 checkpoint,NeMo-Skills 0.7.0,VR200,温度 1.0/top_p 0.95,reasoning 剥离后评分):非 KDA 128 的 7 benchmark pooled macro +0.05 [-0.85, +0.94](AIME25 +0.83、GPQA-Diamond +1.14、MATH-500 +0.60、GSM8K +0.30、BeyondAIME +0.38、HMMT -0.21、APEX -2.66);全 196 pooled macro -0.62 [-1.69, +0.42],三个难长输出集 pooled -1.58 [-3.69, +0.44](正文披露为"可能代价,非已证");输出长度:全 196 在难集上 63.6% 的配对更短(z=+11.1),非 KDA 51.2%(z=+1.0)。
- 默认行为:GB300 上 NVFP4 checkpoint(含 MTP)与 Z.ai FP8 checkpoint 的默认 greedy 输出与 main 逐 token 相同;未加 `--quantization-config` 时无任何变化。

## 验证与回归风险

- 单测:`pytest tests/models/glm5next` 在 GB300(上游 nightly 96710245 + 本分支 Python 源)113 passed / 9 skipped;新 `tests/models/glm5next/test_online_quantization.py`(5 cases)用真实 config 与 `ColumnParallelLinear`:FP8 checkpoint 的 KDA o_proj / fused in_proj 保持 BF16、online mxfp8 target 从 BF16 起量化、NVFP4 的 MTP 层保持未量化;main @ bec08b701 上三个 KDA case 失败。
- 风险(正文明示):ROCm 走同代码路径(MXFP8 经 `RocmDotScaledMxfp8LinearKernel`)但未测;CI 只覆盖 config wiring,无 CI job 服务 GLM-5.3-Flash(191 GB checkpoint),端到端在 CI 外完成;吞吐/精度运行用的 PR 早期版本(量化同层同方法,`Quantizing N layers` 计数确认);disaggregated prefill-only 只测了吞吐未测精度;`kv_b_proj` 的 online 量化(被吸收进 bmm 权重)未测;不要对 decode 层用 per-tensor activation scale(FP8 per-tensor W+A 在作者运行中 AIME 94-97% → 53-60%、MATH-500 98% → 65%)。

## 关联

- 相关 #59833(加载 checkpoint 量化的 MLA 到 BF16,互补)、#55741(通用 MLA `kv_b_proj` bmm 路径,无冲突)、#61457(本分支基于其上)、#63102(specialized decode 的启动检查);详见正文 "Not a duplicate" 清单。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
