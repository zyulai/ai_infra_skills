# PR #43062: [Diffusion] Add ModelOpt FP8 checkpoint quantization for Pi0.5

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @trebladev |
| 合并 | 2026-10-11 |
| 分类 | Quantization (tier: major) |
| Label | quant, run-ci, diffusion |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files;diff 含 5 个文件,另有 2 个文件因超长被省略) |
| 链接 | https://github.com/sgl-project/sglang/pull/43062 |

## 一句话结论

为 Pi0.5 VLA 增加 ModelOpt FP8 checkpoint 校准与原生 FP8 推理:CUDA Graph 下延迟 -21.8%、峰值显存 -37.2%。

## 动机

Pi0.5 此前只能以 BF16 常驻推理。本 PR 复用 SGLang 已有的 ModelOpt FP8 linear backend,为 Pi0.5 提供 checkpoint 校准与原生 FP8 推理能力,量化覆盖与合成校准配方沿用 jetson-ai-lab 的 openpi_on_thor 教程,目标是降低机器人 VLA 模型的推理延迟与显存占用。

## 优化点

新增校准/导出工具,支持观测数据集或显式选择的合成输入;用 per-tensor E4M3 权重与静态 activation scale 量化 PaliGemma 与 action-expert 的融合投影(qkv_proj/o_proj/gate_up_proj/down_proj)、SigLIP encoder 投影、multimodal projector 及 action 输入/输出头。替换层 `Pi05Fp8Linear` 保持原有 tensor/tuple 接口与输出 dtype(`x.to(torch.bfloat16)` 进、按原 dtype 出);embeddings、convolutions、norms、AdaRMS dense、time MLP 保持不量化。加载侧读取 checkpoint `config.json` 的 `quantization_config`,校验 fused-projection ModelOpt FP8 契约(单分量 scale、fused 权重),再做投影替换与 `finalize_fp8_weights` 校验。约束:仅支持单张 SM89+ NVIDIA GPU 上的常驻 BF16 推理,拒绝 TP/SP 与 component offload。

## 关键改动

```text
python/sglang/multimodal_gen/tools/quantize_pi05_modelopt_fp8.py: 校准与导出工具(mtq.quantize、export_state、校验数据)
python/sglang/multimodal_gen/runtime/vla/pi05_quantization.py: 投影选取/替换/FP8 权重 finalization 与契约校验
python/sglang/multimodal_gen/runtime/models/vlas/pi05_policy.py: 加载期 quantization_config 处理与单 GPU/约束检查
python/sglang/multimodal_gen/runtime/models/vlas/pi05_core.py: projection_dtype/linear_compute_dtype 适配 FP8 权重下的计算 dtype
python/sglang/multimodal_gen/test/unit/test_pi05_modelopt_fp8.py: CPU/CUDA 契约测试(替换、接口保持、校准、graph replay)
```

## 性能声明

RTX 4090 D,PyTorch 2.13.0+cu130 / ModelOpt 0.47.0 / Transformers 5.12.1,完整 Pi0.5 Libero 模型,batch size 1、三个 224x224 相机输入、200 tokens、10 denoising steps、torch_sdpa,5 次预热 + 30 次计时(含 prefix encoding 与 action 采样,不含预处理/加载/首次建图):

- CUDA Graph:BF16 median 70.97 ms → FP8 55.53 ms(1.28x,延迟 -21.8%);峰值分配显存 6.72 GiB → 4.22 GiB(-37.2%)。
- Eager:BF16 248.44 ms → FP8 360.67 ms(0.69x,延迟 +45.2%),即提速仅适用于 CUDA Graph 执行。

精度:10 预测步 x 7 动作维度上,BF16 vs native FP8 平均绝对差 0.008364、最大 0.100873、cosine 0.999526(per-observation 0.999384-0.999659);fake quant vs native FP8 0.006177/0.059110。真实机器人任务成功率未验证。

## 验证与回归风险

新增单测覆盖:融合替换前后 forward 逐位一致、bias/tuple 接口保持、排除项(conv/time MLP/AdaRMS dense)不量化、ModelOpt amax 导出校验、校准全流程不改动固定 noise、config 校验与非法输入拒绝;另含 CUDA 校准/checkpoint 加载/graph-replay 测试。风险:仅覆盖 Pi0.5 单 GPU BF16 场景,TP/SP/offload 直接报错;FP8 路径要求 fused E4M3 checkpoint 权重;eager 模式反而变慢 45.2%,需 CUDA Graph 才有收益;真实任务精度未评估。

## 关联

配方来源 jetson-ai-lab openpi_on_thor 教程;复用 SGLang ModelOpt FP8 linear backend(`modelopt_fp8`)。正文未列出关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
