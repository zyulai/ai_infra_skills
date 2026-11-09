# PR #60816: [ROCm][Qwen4Exp] Remove the Qwen3.8-Flash-Next TEP8 MXFP4 emulation guard

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#60816](https://github.com/vllm-project/vllm/pull/60816) |
| 作者 | mjkvaak-amd |
| 合并日期 | 2026-10-19 |
| labels | bug, rocm, ready, qwen |
| 热度 | 👍 0 · 💬 26 |
| 分类 | Quantization |
| bug 类型 | general |

## 一句话结论
Qwen3.8 Flash Next TEP8 的 MXFP4 仿真 guard 因 #55104 去掉 padding 而过时,移除后原生 AITER W4A4 kernel 精度持平且吞吐约 2 倍。

## 问题现象
并非新崩溃,而是过时 guard 造成的性能损失:在 gfx950 上以 TEP8 运行 `amd/Qwen3.8-Flash-Next-Quark-MXFP4` 且 `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0` 时,`_requires_qwen38_tep8_emulation` 强制走 OCP MX 仿真后端(并打日志「Performance will be lower」),输出吞吐仅为原生 kernel 的一部分(158 vs 362 tok/s @8k/1k 并发 4;741 vs 1399 @并发 32)。

## 根因
正文披露:该 guard 当初加入是因为 AITER 对该 shape 将 MoE 宽度从 640 padding 到 768,原生 W4A4 kernel 掉精度。而 #55104 将 W4A4 intermediate 对齐到 128 后,640 不再被 pad(原样存储),原生 kernel 精度与仿真持平(GSM8K 0.9651 vs 0.9629),CK-Tile 的 `inter_dim % 256` scale 问题在该 shape 上不再出现——guard 已过时。guard 仅精确匹配 `E=512, H=2560, N=640`、EP8、gfx950,其余 shape 与 checkpoint 卡配方(融合 shared expert 时每 rank 65 个 local expert,不满足恰好 64)均不受影响。

## 修复方案
1. 删除 `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` 中的 `_requires_qwen38_tep8_emulation` 及其在 `select_mxfp4_moe_backend` 里的强制 EMULATION 分支,该 shape 现在按常规优先级选择原生 `AITER_MXFP4_MXFP4`;
2. 顺带把 `rocm_aiter_moe.py` 中内联的 pad 舍入重构为 `aiter_moe_pads()`:W4A4 SiLU 路径在 TP1 时向 AITER 传未加倍 pad(AITER 内部会自行加倍,双重加倍会 mask 掉有效 intermediate 行);
3. 更新/新增单测(`test_qwen38_tep8_selects_native_aiter` 等)。

关键 diff 摘录(删除的核心逻辑):
```python
if _requires_qwen38_tep8_emulation(config, requested_activation_key):
    backend = Mxfp4MoeBackend.EMULATION
    ...
    return _return_or_raise(backend, config, kMxfp4Static, ...)
```

## 影响与回归风险
- 影响窗口:#55104 之前 pad 存在时不适用(正文明确基于 #55104 已合入)。
- 回归面:仅 ROCm MXFP4 oracle 与 AITER pad 计算路径,gfx950 专用,CUDA 无任何变化;gfx942 不运行该 W4A4 MXFP4 MoE 路径。精度与吞吐验证均在 MI355X 上完成(单测在 AMD kernels CI,精度/吞吐在 CI 外测得)。

## 关联
- [PR #55104](https://github.com/vllm-project/vllm/pull/55104)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
