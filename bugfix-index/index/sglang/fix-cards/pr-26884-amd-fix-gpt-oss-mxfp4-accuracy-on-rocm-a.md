# PR #26884: [AMD] Fix GPT-OSS MXFP4 accuracy on ROCm AITER path

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#26884](https://github.com/sgl-project/sglang/pull/26884) |
| 作者 | kkHuang-amd |
| 合并日期 | 2026-06-02 |
| labels | amd, run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
GPT-OSS MXFP4 在 ROCm AITER 上权重 shuffle 布局错误且 is_shuffled 丢失,内核按未 shuffle 解读输出乱码(gsm8k 0.88→0.21),改 separated 布局并补标记。

## 问题现象
ROCm AITER 上 GPT-OSS MXFP4 fused-MoE 路径产生静默错误输出(silently wrong outputs)。MI355X(gfx950)用 `--prefill-attention-backend aiter --decode-attention-backend aiter --page-size 64` 等启动 gpt-oss-120b 时,AITER dispatcher 选了 `module_moe_ck2stages_b16_fp4x2_preshuffle_off_b16_swiglu_per_1x32_mulWeightStage2` 这个 `preshuffle_off` codegen 内核,lm_eval gsm8k `flexible-extract` 从约 0.88 跌到约 0.21。

## 根因
正文披露两个根因,且相互叠加:
1. **Shuffle 布局错误**:旧路径用 `shuffle_weight_a16w4(..., 16, True)`/`shuffle_scale_a16w4(...)`,产出 gate/up 交错(interleaved)的 tile 布局;而针对 GPT-OSS 调优的 AITER MXFP4 fused-MoE 内核——`gptoss_fp4_tuned_fmoe.csv` 的 FlyDSL `flydsl_moe1_afp4_wfp4_bf16_*` 条目(`gate_mode="separated"`)与 CK `module_moe_ck2stages_*_preshuffle_on_*` 回退——期望的是 separated(`is_guinterleave=False`)布局,与同硬件上 ATOM 的产物一致。
2. **`is_shuffled` 标记丢失**:`shuffle_weight_a16w4` 返回的张量带 `.is_shuffled=True`,但调用点写的是 `layer.w13_weight.data = shuffle_weight_a16w4(...)`;对 `.data` 赋值不复制自定义 Python 属性,`layer.w13_weight.is_shuffled` 始终未设置。`aiter.fused_moe` 经 `getattr(w1, "is_shuffled", False)` 读取,看到 False 便派发 `preshuffle_off` codegen——该内核假定权重字节未 shuffle,拿到实际已 shuffle 的字节即输出乱码。

两个 bug 相互作用:即使布局正确,缺失的 `is_shuffled` 仍会把派发引向错误的 codegen 分支。

## 修复方案
1. `mxfp4.py` `Mxfp4MoEMethod.process_weights_after_loading`(_use_aiter 分支):先反交错(de-interleave)w13 权重/scale/bias,再用 `shuffle_weight(..., is_guinterleave=False, gate_up=...)`/`shuffle_scale(...)` 产出 ATOM 对齐的 tile 布局,并补上标记:

```diff
+            layer.w13_weight.is_shuffled = True
+            layer.w2_weight.is_shuffled = True
```

2. `apply`(_use_aiter 分支):`.view(torch.float4_e2m1fn_x2)` 会产生丢弃 Python 属性的新张量,按原标记重新附上 `is_shuffled=True`;并向前传 `swiglu_limit=self.moe_runner_config.swiglu_limit or 0.0` 使 OAI swiglu clamp 生效。
3. `moe_runner/aiter.py` `AiterRunnerCore.run`:`gate_mode` 改由新环境变量 `SGLANG_USE_AITER_MOE_GU_ITLV` 门控(env 为 True 选 INTERLEAVE,否则 SEPARATED);`environ.py` 注册该变量(AMD & ROCm 段),`server_args.py` 在 ROCm+MXFP4 的 GPT-OSS 路径显式 `set(False)`,MI355X 精度测试同步加 `SGLANG_USE_AITER_MOE_GU_ITLV=0`。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:ROCm AITER 的 GPT-OSS MXFP4 加载与推理路径(权重布局、is_shuffled 标记、swiglu_limit 透传);`SGLANG_USE_AITER_MOE_GU_ITLV` 默认保留 INTERLEAVE,其他 AITER MXFP4 调用方行为不变,GPT-OSS 路径显式切到 SEPARATED。验证:lm_eval gsm8k 3-shot flexible-extract 0.8810±0.0089(ATOM 基线 0.8825±0.0089,同一标准误内),strict-match 0.3541±0.0132(ATOM 0.3487±0.0131);并经 byte-level 签名(前两层 w1/w2/scale/bias 的 bytesum/head8 及 is_shuffled)确认与 ATOM 位级一致。作者声明无性能回归(内核家族不变,仅输入布局正确化)。CI:Base/Extra 红(原帖标注)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
