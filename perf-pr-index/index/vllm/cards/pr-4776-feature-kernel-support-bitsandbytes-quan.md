# PR #4776: [Feature][Kernel] Support bitsandbytes quantization and QLoRA

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @chenqianfzh |
| 合并 | 2024-06-01 |
| 分类 | Quantization(tier: major) |
| Label | 无 |
| Diff | diffstat 缺失(素材未提供 additions/deletions 与文件数);可见 diff 未标注截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/4776 |

## 一句话结论

接入 bitsandbytes NF4 4-bit 量化(加载时量化)与 QLoRA 适配器推理,首步支持 HuggingFace QLoRA 模型(当前仅 Llama 基座、无 TP/PP)。

## 动机

QLoRA(arXiv:2305.14314)把基座权重 4-bit 量化,再配低秩高精度 Low-Rank 矩阵生成输出,可在不损失性能的前提下大幅降低权重加载内存。当时 vLLM 无法加载 QLoRA 作者发布的开源模型(如 timdettmers/qlora-flan-7b,基座 huggyllama/llama-7b),也缺少 bitsandbytes 在线(加载时)量化路径。本 PR 作为 vLLM 支持 QLoRA 的第一步:用户可带或不带 QLoRA adapter 运行;限制为仅支持 llama 基座、不支持 TP/PP(声明为下一步工作)。

## 优化点

机制上:(1) 新增 `BitsAndBytesConfig`/`BitsAndBytesLinearMethod`:权重打包为 `[n*m/pack_ratio, 1]` 的 uint8,`apply` 时把输入转 bfloat16 后按 shard 分段调用 `bitsandbytes.matmul_4bit`,各段结果拼回输出(bnb 的 `out=` kwarg 因上游 bug #1235 暂不可用);最小算力要求 7.0。(2) 新增 `BitsAndBytesModelLoader`:加载时对 target_modules(gate/down/up/q/k/v/o_proj 默认集,或从 adapter 的 `adapter_config.json` 读取)用 `quantize_4bit(nf4, compress_statistics=True)` 量化,并通过 `bitsandbytes_stacked_params_mapping` 把 q/k/v、gate/up 分片归属回堆叠参数,把 quant_state 与 shard offsets 挂到参数属性上。(3) `QKVParallelLinear`/`MergedColumnParallelLinear` 的 `weight_loader` 增加 bitsandbytes 分支,按量化后总行数等比换算 shard offset/size。(4) CLI 层:新增 `--qlora-adapter-name-or-path` 与 `load_format="bitsandbytes"`,两者强制配对,且显式拒绝 TP/PP 组合。

## 关键改动

```text
vllm/model_executor/layers/quantization/bitsandbytes.py: 新增 BitsAndBytesConfig/BitsAndBytesLinearMethod(matmul_4bit 分段 apply)
vllm/model_executor/model_loader/loader.py: 新增 BitsAndBytesModelLoader,加载时 NF4 量化并组装 quant_state/offsets
vllm/model_executor/layers/linear.py: weight_loader 的 bitsandbytes shard 调整(QKV/MergedColumn)
vllm/config.py / engine/arg_utils.py: qlora_adapter_name_or_path、bitsandbytes load_format、TP/PP 拒绝
vllm/model_executor/models/llama.py: bitsandbytes_stacked_params_mapping
```

关键片段(apply 的分段 matmul):

```python
for i in range(len(quant_states)):
    output_size = quant_states[i].shape[0]
    out[:, current_index:current_index + output_size] = matmul_4bit(
        bf_x, qweight[offsets[i]:offsets[i + 1]].t(), quant_states[i])
    current_index += output_size
```

## 性能声明

PR 正文未提供 benchmark 数字与测试条件,perf claims 检测为 none。动机仅引用 QLoRA 论文口径("cuts memory consumption in LLM weight loading without degrading performance"),未在 vLLM 内量化验证。

## 验证与回归风险

新增 `tests/quantization/test_bitsandbytes.py`:验证 MLP/attn 权重 dtype 为 uint8、lm_head/embed_tokens/layernorm 不被量化,并断言两条提示的贪心输出与预期文本一致。新增 `examples/lora_with_quantization_inference.py` 演示 bitsandbytes/AWQ/GPTQ 三种量化 + LoRA。回归风险:`weight_loader` 对所有模型共用路径插入 bitsandbytes 分支(靠 `use_bitsandbytes` 属性守卫);仅 llama 提供 `bitsandbytes_stacked_params_mapping`,其他模型加载会抛 AttributeError;`matmul_4bit` 逐 shard 循环每次 forward 都执行,decode 延迟受影响(eager 模式下使用);依赖 bitsandbytes==0.42.0。

## 关联

QLoRA 后续扩展(TP/PP、更多基座模型)PR 编号素材未注明。索引内同区域(Quantization)卡片:#1032(AWQ 支持)、#2497(GPTQ Marlin kernel)、#14447(WNA16 Marlin)、#23777(LoRA 启用时启动加速)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
