# PR #20268: Enable ZP Support for Machete

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @czhu-cohere |
| 合并 | 2025-07-01 |
| 分类 | Quantization(major) |
| Label | performance, ready |
| Diff | diffstat 缺失(diff 全文已提供,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/20268 |

## 一句话结论

在 vLLM 前端启用 Machete kernel 已支持的 zero-point(asymmetric)量化路径,使 w4a16 非对称 checkpoint 可走 Machete,带 zp 的 kernel 延迟平均仅高约 1.5%。

## 动机

Machete kernel 本身已支持 zero-point,但 vLLM 的 `machete.py` 前端在 `can_implement` 中直接拒绝 `zero_points` 配置("Zero points currently not supported"),导致非对称(compressed-tensors w4a16 asym 等)checkpoint 只能回退到 Marlin;Marlin 在部分场景并非最优,把 ZP 路径打开可让这类模型用上 Machete。

## 优化点

`process_weights_after_loading` 新增 `transform_w_zp`:把 `weight_zp` 的参数布局重排为 `{input_dim=0, output_dim=1, packed_dim=1}`,解包 packed 值到 int32,然后按 Machete Readme 的做法把 scale 预乘进 zero-point:`x.data = -1.0 * w_s * x_unpacked`(预先施加 scale 并取负,免去 kernel 内的 per-group 反量化开销);`apply_weights` 把 `b_group_zeros` 从硬编码 `None` 改为传入 `w_zp`。`can_implement` 删除对 zero_points 的拒绝(仍要求 weight_type 在 Machete 支持的量化类型内)。另修正测试中 `group_size_valid` 的整除方向(`shape[2] % group_size == 0`),并更新 `benchmark_machete.py` 对 marlin 接口的调用签名(`c=None`、`global_scale=None`)。

## 关键改动

```text
vllm/model_executor/layers/quantization/kernels/mixed_precision/machete.py: 解禁 zero_points;新增 transform_w_zp(scale 预乘);apply_weights 传 b_group_zeros
tests/kernels/quantization/test_machete_mm.py: 修正 group_size 整除判断方向
benchmarks/kernels/benchmark_machete.py: marlin gemm 调用签名适配
```

scale 预乘节选:

```python
def transform_w_zp(x):
    permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=1)
    x_unpacked = unpack_quantized_values_into_int32(x.data, c.weight_type,
                                                    packed_dim=1)
    w_s = getattr(layer, self.w_s_name).data
    # pre-apply scales to zero-points
    x.data = (-1.0 * w_s * (x_unpacked.to(w_s.dtype))).contiguous()
```

## 性能声明

kernel 延迟(`benchmark_machete.py`,act/group-scale/out/group-zero 均 bf16,具体形状列表在截图里):跨不同 shapes,带 zp 平均比不带 zp 高约 1.5% 延迟(PR 未注明 GPU 型号)。这是启用 ZP 的代价而非加速——收益在于让非对称 checkpoint 能用 Machete 替代 Marlin。精度对比(w4a16 asym gs=128 compressed-tensors,基于 CohereLabs/c4ai-command-a03-2025):lm-eval mmlu_pro exact_match Machete 0.6990±0.0041 vs Marlin 0.6967±0.0041;sanity 查询两边输出相同。

## 验证与回归风险

依赖 `test_machete_mm.py` 中既有的 AWQ 风格 zp 正确性用例;e2e 用 lm-eval(mmlu_pro)与 sanity 查询对比 Marlin 基线。风险面:`transform_w_zp` 在加载时额外做 unpack 与 scale 预乘,增加权重加载时间与内存(scale 预乘把 zp 展成浮点);symmetric 配置路径不变;`group_size_valid` 方向修正可能影响此前侥幸通过的测试配置。

## 关联

同区域:Machete/Marlin w4a16 kernel 选型与 compressed-tensors 量化前端;相关编号素材未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
