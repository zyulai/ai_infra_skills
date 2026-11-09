# PR #50411: [Model] Fused mm preprocess normalisation on the Device

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @noooop |
| 合并 | 2026-08-06 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | documentation, new-model, ready, multi-modality, qwen, build-docs |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数) |
| 链接 | https://github.com/vllm-project/vllm/pull/50411 |

## 一句话结论

把视觉模型的 rescale+normalize 融合为 GPU 上的单 kernel(`FusedInputNorm`),传输链路全程保持 uint8,PCIe 传输量减半,Qwen2-VL 预处理速度接近翻倍。

## 动机

多模态图像数据从 entrypoint 经 engine core 到 GPU 的传输使用 uint8,但 HF processor 在 CPU 上先做 rescale(/255)与 normalize(减均值除方差),数据在进入 ViT 前才转 bf16——normalize 的浮点算术全部压在 CPU 上,而 CPU 侧的 Qwen2-VL processor 本就以慢著称。GPU 上做 normalize 几乎不耗时且可被后续算子融合,把这个"CV 常用技巧"搬到设备侧可同时省 CPU 算力与一半 PCIe 带宽。

## 优化点

新增 `FusedInputNorm` 模块:把 rescale factor(通常 1/255)烘焙进有效 mean/std(有效 mean = `image_mean / rescale_factor`),实现 `output = (input * rescale_factor - mean) / std` 的单次映射;forward 内部 view 成 `(patches, channel, patch_size)` 后调用 `F.batch_norm`(frozen,training=False,eps=0),以 fp32 计算保证精度再 cast 到 ViT dtype,identity 情形(weight 全 1、bias 全 0)直接跳过。参数从 `processor_config.json`/`preprocessor_config.json` 读取,缺失时回退 image_processor 对象;`mm_device_do_normalize` 开启时 `merge_mm_processor_kwargs` 自动注入 `do_normalize=False, do_rescale=False`,让 CPU 侧跳过这两步、pixel_values 保持 uint8。整条路径变为 Entry point(uint8)→ Engine Core(uint8)→ GPU(uint8)→ GPU 内 fp32 归一 → bf16。默认对支持的架构开启(当前 `Qwen2VLForConditionalGeneration` 与 `Qwen2_5_VLForConditionalGeneration`),经 `supports_mm_device_do_normalize` ClassVar 与 `_ModelInfo` 注册;Rust frontend 尚不支持,`VLLM_USE_RUST_FRONTEND=1` 时强制关闭并告警。设计文档写入 `docs/design/mm_processing.md`。

## 关键改动

```text
vllm/model_executor/models/vision.py: 新增 FusedInputNorm(fused rescale+normalize,batch_norm 实现)
vllm/model_executor/models/qwen2_vl.py / qwen2_5_vl.py: input_norm 接入 _process_image_input/_process_video_input
vllm/config/multimodal.py + vllm/config/model.py: mm_device_do_normalize 配置、默认值解析与 Rust frontend 强制关闭
vllm/transformers_utils/processor.py: get_processor_config() 读归一化参数
```

## 性能声明

PR 正文数据(Qwen/Qwen2-VL-2B-Instruct):

- 预处理微基准(1024x1024 图像,snippet benchmark):Qwen2VLImageProcessor preprocess 12.22ms → 6.27ms,接近减半;e2e 0.1541s → 0.1439s。
- `vllm bench serve`(openai-chat,random-mm,100 prompts,random-output-len 1,总输入 168,350 token):Mean TTFT 3809.70ms(关)→ 3459.52ms(开);Median TTFT 3612.82ms → 3057.37ms;P99 基本不变(~7992ms);吞吐 12.27→12.27 req/s。同条件下 `VLLM_USE_RUST_FRONTEND=1` 的 Mean/Median TTFT 为 3078.26ms / 2021.02ms(Peak output TPS 78),PR 自述两者正交、可叠加。
- PPL 对比(dtypes bf16/bf16):开关两种取值下 vllm PPL 完全相同(Qwen2-VL-2B 41096328.0,Qwen2.5-VL-3B 18406568.0),与 Transformers 常数差异分别 0.036% 与 0.418%。

PR 未注明 bench 用的 GPU 型号与并发参数(仅脚本参数)。

## 验证与回归风险

新增 `tests/models/multimodal/processing/test_qwen2_vl.py::test_mm_device_do_normalize`(开关两种路径的 pixel_values 逐元素一致)与 `test_qwen.py` 的 PPL parametrize(True/False,固化 HF 参考值);`ppl_utils.py` 打印开关状态。风险:默认开启改变 Qwen2-VL/Qwen2.5-VL 的预处理路径,依赖 pixel_values dtype/值域的下游(如自定义 processor)需注意 HF processor 被注入 `do_normalize=False`;非支持的架构 flag 会被强制降级为 False 并告警;`MultiModalConfig.compute_hash` 因新字段变化(配置哈希改变)。

## 关联

同区域:多模态预处理加速卡(pr-30200 renderer、pr-32104 MMTensorIPC、pr-22742/22763 ViT 并行与 graph);文档位于 `docs/design/mm_processing.md` 的 "Speeding Up Multi-Modal Data Processing" 一节。PR 正文未列出具体前序/后续 PR。
