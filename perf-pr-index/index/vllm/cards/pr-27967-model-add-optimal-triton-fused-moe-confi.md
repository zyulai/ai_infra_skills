# PR #27967: [Model] add optimal triton fused moe configs for NemotronH MoE

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tomeras91 |
| 合并 | 2025-11-04 |
| 分类 | MoE(major) |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 与文件数);diff 内容可见 5 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/27967 |

## 一句话结论

为 NemotronH MoE 新增 4 份 Triton fused MoE 调优配置(H100/L40S,BF16),使该形状不再使用默认 tile 参数。

## 动机

vLLM 的 Triton fused MoE kernel 按 `E,N,device,dtype` 命名的 JSON 配置选择 tile 参数,缺配置时回退默认值,该 MoE 形状下 kernel 效率受损。NemotronH(E=128,两种 intermediate size)此前没有对应配置条目,`benchmarks/kernels/benchmark_moe.py` 的架构分支也不认识 `NemotronHForCausalLM`,无法直接生成配置。

## 优化点

纯配置型 PR,不改 kernel 代码。用 `benchmark_moe.py` 网格搜索出每档 batch size(1-4096)的 `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps`、`num_stages`,新增 4 个 JSON:`E=128,N=1856` 与 `E=128,N=928` 各配 NVIDIA H100 80GB HBM3 与 NVIDIA L40S,对应 TP=1、2 两种切分;配置标注 `triton_version: 3.5.0`。`benchmark_moe.py` 的架构识别分支加入 `NemotronHForCausalLM`,以便后续继续生成。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/configs/E=128,N=1856,device_name=NVIDIA_H100_80GB_HBM3.json: 新增调优配置
vllm/model_executor/layers/fused_moe/configs/E=128,N=1856,device_name=NVIDIA_L40S.json: 新增调优配置
vllm/model_executor/layers/fused_moe/configs/E=128,N=928,device_name=NVIDIA_H100_80GB_HBM3.json: 新增调优配置
vllm/model_executor/layers/fused_moe/configs/E=128,N=928,device_name=NVIDIA_L40S.json: 新增调优配置
benchmarks/kernels/benchmark_moe.py: 架构识别加入 NemotronHForCausalLM
```

## 性能声明

PR 正文未给出具体性能数字,仅说明配置由 `benchmarks/kernels/benchmark_moe.py` 生成;生成条件为 TP=1,2、H100 HBM3 80GB 与 L40S、dtype=BF16。

## 验证与回归风险

无新增测试;风险面小:配置按 `E,N,device,dtype` 精确匹配才生效,只影响 E=128、N=1856/928 在这两款 GPU 上的 fused MoE tile 选择,其他模型/设备不受影响;`triton_version` 标记随 Triton 版本匹配策略生效。

## 关联

同分类 tuned config 类 PR:#37340(Qwen3.5 H200)。素材未列出其他前序/后续 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
