# PR #20922: :sparkles: [diffusion][npu][quant] Add MXFP8 quantization support for Wan2.2 Diffusion on Ascend NPU

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @TallMessiWu |
| 合并 | 2026-05-07 |
| 分类 | Quantization(classify.py 给出,tier: major) |
| Label | documentation, quant, npu, run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 片段约 12 个文件,另有 4 个文件因超长被裁剪 |
| 链接 | https://github.com/sgl-project/sglang/pull/20922 |

## 一句话结论

为 Ascend NPU 上的 Wan2.2 扩散模型接入 MXFP8(W8A8)量化,单卡生成 E2E 延迟降低约 11%。

## 动机

Ascend NPU 的 MXFP8 支持在 issue #14424 中被跟踪为缺口,扩散模型(Wan2.2)此前只能跑 FP16/BF16 或其他 ModelSlim 方案。本 PR 补齐 diffusion 子系统的 MXFP8 在线/离线两条路径,闭合 #14424 的部分内容。硬件要求 Ascend A5 及更新(`npu_dynamic_mx_quant` 在 A2/A3 上不可用),CANN ≥ 8.0.RC3。

## 优化点

- 在线量化:新增 `MXFP8Config` + `NPUMXFP8DiffusionLinearMethod`,load 时用 `torch_npu.npu_dynamic_mx_quant` 把 FP16/BF16 权重在线量化为 MXFP8(float8_e4m3fn 权重 + float8_e8m0fnu scale);推理时激活按 token 动态量化,由 `npu_quant_matmul(..., group_sizes=[1,1,32])` 以 block_size=32 执行 MXFP8 matmul。3D 输入先 flatten 成 2D 以满足 NPU kernel 要求,输出再恢复形状。
- 离线量化:新增 `ModelSlimMXFP8Scheme`,加载 msmodelslim 预量化权重(float8_e4m3fn 权重 + uint8 scale),从 `quant_model_description.json` 的 `W8A8_MXFP8` 类型自动识别 scheme。
- CLI/loader:multimodal_gen 的 server_args 新增 `--quantization` 覆盖项,loader 优先级变为显式 flag > config.json 自动识别;fsdp_load 的 unused-key 列表加入 `weight_scale` 防止离线权重加载崩溃。
- 工具链:把 `wan_repack.py` 重构为一步式 repack(复制原模型、转换权重、恢复 config.json),修复 4 个使原脚本完全不可用的 bug(glob 串当字面路径、`RENAME_DICT` 未定义的 NameError、无条件 `dict.pop` 的 KeyError 等),支持 TI2V-5B 与 T2V/I2V-A14B 双 transformer。

## 关键改动

```text
multimodal_gen/runtime/layers/quantization/mxfp8_npu.py: 新增,在线 MXFP8 config 与 linear method
multimodal_gen/runtime/layers/quantization/modelslim_mxfp8_scheme.py: 新增,离线 ModelSlim MXFP8 scheme
multimodal_gen/runtime/layers/quantization/modelslim.py: W8A8_MXFP8 分支分发到 ModelSlimMXFP8Scheme
multimodal_gen/runtime/server_args.py / transformer_load_utils.py: --quantization 显式覆盖
multimodal_gen/tools/wan_repack.py: 重构为一步式 repack 工具并修 bug
```

```python
qx, input_scale = torch_npu.npu_dynamic_mx_quant(
    x_2d, dst_type=torch_npu.float8_e4m3fn
)
output = torch_npu.npu_quant_matmul(
    qx,
    layer.weight.transpose(0, 1),
    layer.weight_scale_inv.transpose(0, 1),
    scale_dtype=torch_npu.float8_e8m0fnu,
    pertoken_scale=input_scale,
    pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
    bias=bias.to(torch.float32) if bias is not None else None,
    output_dtype=original_dtype,
    group_sizes=[1, 1, MXFP8_BLOCK_SIZE],
)
```

## 性能声明

Wan2.2-TI2V-5B,单 NPU GPU,704×1280、81 帧、40 步推理,开启 CacheDiT(TaylorSeer 等缓存参数两组一致):E2E latency 基线 102922.86 ms,离线 MXFP8 91619.75 ms(-11.0%),在线 MXFP8 91413.05 ms(-11.2%)。分阶段:DenoisingStage 83386.55→72910.67/72868.55 ms(约 -12.6%),DecodingStage 约 -13.0%;TextEncodingStage 基本持平。三个臂均基于同一 commit `ef874c0a`,PR 未注明 NPU 具体型号。

## 验证与回归风险

单元测试仅更新 `test_transformer_quant.py` 的 fixture(`quantization=None` 字段);PR 正文未给出量化前后视觉质量/精度的数值对比。风险集中在 NPU diffusion 路径:A5-only(`npu_dynamic_mx_quant` 不支持 A2/A3),A5 表格中 Linear 仍标 WIP、MoE MXFP8 未支持;`--quantization` 改变 loader 优先级,显式 flag 会压过 checkpoint 自动识别;`fsdp_load` 的 unused-keys 变化影响离线权重加载路径;`srt/layers/quantization/fp8.py` 的小重构(import 清理、process_weights_after_loading 结构调整)触及 LLM 侧 FP8 代码。

## 关联

同区域:ModelSlim on Ascend 支持(PR #14504)、mix-bits MoE 高级用法(PR #17361,进行中);闭合 issue #14424 的部分内容(MXFP8/MXFP4 on Ascend)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
