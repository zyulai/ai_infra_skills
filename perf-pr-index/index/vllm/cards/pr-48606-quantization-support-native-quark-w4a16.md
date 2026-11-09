# PR #48606: [Quantization] Support native Quark W4A16 INT4/UINT4 exports in vLLM

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @limitmhw |
| 合并 | 2026-09-18 |
| 分类 | Quantization \| tier: major |
| Label | ready, qwen, quantization, verified |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明 4 file(s) dropped from diff for size (see PR for full diff) |
| 链接 | https://github.com/vllm-project/vllm/pull/48606 |

## 一句话结论

新增原生 Quark W4A16 INT4/UINT4 加载路径（含 MoE），GSM8K 约保留 98.5% BF16 精度。

## 动机

为 Quark 导出的 `real_quantized` INT4/UINT4 checkpoint（`export.pack_method` 为 `order` / `reorder`）补上 vLLM 原生加载路径。正文强调：MoE router（`mlp.gate`）等 FP16 模块必须出现在 `quantization_config.exclude`，vLLM 不会从张量 dtype 推断，Quark 导出应保留全部 `exclude_layers_name` 模式（已有 Quark issue 记录 `*mlp.gate` 未写入 config）；并称没有现存 PR 加入带 MoE + Qwen3.5 加载的原生 Quark `real_quantized` W4A16 INT4/UINT4 支持。

## 优化点

新增 `QuarkW4A16Int4`（dense 层）与 `QuarkW4A16Int4MoEMethod`（routed experts）；把 Quark pack order 规范化到现有 `awq_*` ops 使用的布局（仅 compute kernel layout）；对称 INT4 与非对称 UINT4 的 MoE experts 都复用共享的 WNA16 oracle 后端 `TritonWNA16Experts`。`quark.py` 新增 `_is_w4a16_int4()`：当 weight dtype 为 `int4`/`uint4` 且 `pack_method` 为 `order`/`reorder` 时，按 `parse_w4a16_int4_weight_config` 的 group_size 与对称性映射到 `kInt4Static` / `kInt4Static32` / `kInt4Static32Asym` / `kInt4StaticAsym`；`get_quant_method` 把 `weight_config` 与 `layer_name` 传给 scheme/method；`apply_vllm_mapper` 对非纯字符串 list 原样保留。`quark_moe.py` 的 `get_moe_method_target` 增加 `_is_w4a16_int4` 分支返回 `QuarkW4A16Int4MoEMethod`，并接入 `int_wna16` 的 `make_wna16_moe_kernel` 等；`auto_awq.py` 允许 `w_zp_name` 为 None（对称量化只转 weight），`triton_moe.py` 与 `kernels/linear/__init__.py` 做配套调整。`quark.py` 的 `apply_vllm_mapper` 原先只对纯字符串 list 调 `apply_list`，本 PR 让含非字符串元素的 list 原样保留，以适配 Quark config 的复杂结构；`get_quant_method` 还把匹配到的 `weight_config` 与 `layer_name=prefix` 下传给 scheme/method。

## 关键改动

```text
vllm/model_executor/layers/quantization/quark/quark.py: 注册 QuarkW4A16Int4、新增 _is_w4a16_int4 判定与 weight_config/layer_name 传递。
vllm/model_executor/layers/quantization/quark/quark_moe.py: 新增 QuarkW4A16Int4MoEMethod，接 WNA16 oracle 后端。
vllm/model_executor/layers/quantization/auto_awq.py: _convert_awq_to_standard_format 支持 w_zp_name=None（对称量化）。
vllm/model_executor/layers/fused_moe/experts/triton_moe.py: SUPPORTED_W 调整以覆盖 INT4/UINT4 键。
tests/quantization/test_quark.py: 原生 Quark W4A16 加载测试。
```

```python
def _is_w4a16_int4(self, weight_quant, input_quant) -> QuantKeyMatch:
    if weight_quant is None or input_quant is not None:
        return QuantKeyMatch(False, None, None)
    is_int4 = weight_quant.get("dtype") in ("int4", "uint4")
    is_packed = self.pack_method in ("order", "reorder")
    if not (is_int4 and is_packed):
        return QuantKeyMatch(False, None, None)
    group_size, is_symmetric = parse_w4a16_int4_weight_config(weight_quant)
    if is_symmetric:
        weight_key = kInt4Static32 if group_size == 32 else kInt4Static
    else:
        weight_key = kInt4Static32Asym if group_size == 32 else kInt4StaticAsym
    return QuantKeyMatch(True, None, weight_key)
```

## 性能声明

PR 正文未给出定量性能数据（无吞吐/延迟/显存数字）。正文给出 GSM8K 精度（Qwen3.6-35B-A3B、5-shot、注入 `<think>\n\n</think>\n\n` 前缀抑制 thinking）：BF16 基线 flexible-extract 0.9158±0.0076 / strict-match 0.9090±0.0079；RTN-INT4-W4A16 0.9045±0.0081 / 0.8961±0.0084；RTN-UINT4-W4A16 0.9022±0.0082 / 0.8976±0.0083，量化模型保留约 98.5% BF16 精度，INT4 与 UINT4 在彼此误差范围内。测试条件：`CUDA_VISIBLE_DEVICES=0 lm_eval run --model vllm --tasks gsm8k_nothink --include_path . --num_fewshot 5 --batch_size auto`，`model_args` 含 `tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.95,enforce_eager=True`；硬件型号 PR 未注明。

## 验证与回归风险

Test plan：`PYTHONPATH=. python -m pytest tests/quantization/test_quark.py -v` → passed（正文未给通过数）。风险/约束：MoE router（`mlp.gate`）及其他 FP16 层必须列入 `quantization_config.exclude`，vLLM 不从张量 dtype 推断；Quark 导出需持久化所有 `exclude_layers_name` 模式；pack order 规范化仅作用于 compute kernel layout。pack 末尾注明 4 file(s) dropped from diff for size (see PR for full diff)。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
