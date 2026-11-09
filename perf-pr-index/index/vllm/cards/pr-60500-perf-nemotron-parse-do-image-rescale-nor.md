# PR #60500: [Perf][Nemotron-Parse] Do image rescale/normalize on the device

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @HangruiCao2000 |
| 合并 | 2026-10-14 |
| 分类 | Other |
| Label | documentation, ready, multi-modality |
| Diff | diffstat 缺失(pack 未提供 +/− 统计,仅含源码 diff) |
| 链接 | https://github.com/vllm-project/vllm/pull/60500 |

## 一句话结论

Nemotron-Parse 图像 rescale/normalize 移到 GPU(uint8 传输),TTFT 近减半、c=8 聚合吞吐 +21%。

## 动机

`NemotronParseForConditionalGeneration` 的图像 rescale/normalize 在 engine 进程内以单线程 CPU 完成,固定 2048x1664 canvas 以 fp32 传给模型(每请求 40.89 MB),直接拖高 TTFT。vLLM 已有 `mm_device_do_normalize` 设备侧归一化路径(`FusedMMInputNorm`),该模型此前未 opt-in。

## 优化点

模型声明 `supports_mm_device_do_normalize = True`,由 `build_mm_input_norm()` 构建 norm 并在 `_process_image_input` 中应用;feature 关闭时为 `IdentityInputNorm`(plain cast),行为与改动前一致。核心难点是 checkpoint 的 image processor 以 remote code 发布,其 `preprocess` 忽略 `merge_mm_processor_kwargs` 注入的 `do_rescale`/`do_normalize`,只设 flag 会导致 normalize 两次。因此 `_call_hf_processor` override 在 feature 开启时重放 processor 自身 pipeline 至 `ToTensor` 之前(`_build_uint8_canvas`),直接产出 uint8 canvas,同时兼容 v1.2 的 albumentations `transform` 分支与 2.0 的 `_pad_to_size` 分支;若 checkpoint 不暴露该 pipeline 则抛错并指向 `--no-mm-device-do-normalize`,使"feature on 必得 uint8"由构造保证。每请求 payload 从 40.89 MB fp32 降至 10.22 MB uint8。

## 关键改动

```text
vllm/model_executor/models/nemotron_parse.py: opt-in 支持 + _call_hf_processor uint8 canvas 构建 + _process_image_input 接 input_norm
tests/models/multimodal/processing/test_nemotron_parse.py: 新增 device 归一化路径测试
docs/design/mm_processing.md: 支持 mm_device_do_normalize 的模型表加入 nemotron-parse
```

```python
def _process_image_input(self, image_input):
    ...
    flat = self.input_norm(pixel_values.flatten(1), dtype)
    return self.encoder(flat.view(pixel_values.shape))
```

## 性能声明

一台 H100,`vllm/vllm-openai:nightly`(0.31.1rc1.dev23),`--dtype bfloat16 --attention-backend TRITON_ATTN --gpu-memory-utilization 0.25 --max-num-seqs 16`:encoder-only p50(max_tokens=1)263.1→**132.5 ms**,c=1 p50 261.5→131.8 ms(**1.99x**,六次重复区间不重叠);c=8 聚合吞吐 772.7→937.9 tok/s(**+21%**)。归因:单线程 CPU microbenchmark 中 processor 的 float 工作 160.3→5.9 ms。input_norm 无条件化后复验:TTFT p50 226.8→118.6 ms。数值上 bf16/fp16 的 device path 与被替换的 host path 在 10,223,616 个元素上 0 mismatch,fp32 max abs 2.4e-07。

## 验证与回归风险

新增 `test_nemotron_parse.py`(2 passed):覆盖 flag 开关下 dtype/shape、`FusedMMInputNorm` 输出一致性(rtol=1.6e-2)、多种图像输入布局、以及 remote processor 忽略 kwargs 的行为。四种 flag/engine 组合贪心输出 sha256 一致(`b37da04a78014080`,180 tokens);feature off 时无条件调用与原 `pixel_values.to(dtype)` 逐位一致。风险:依赖 checkpoint remote code 的三个非稳定属性(`_resize_with_aspect_ratio`、`_pad_to_size`、`transform`),变更时报 RuntimeError 并提示 `--no-mm-device-do-normalize`(显式失败而非静默变慢);未覆盖 CUDA 以外平台与多图请求;2.0 checkpoint 的复现还依赖 #60272(tied lm_head 修复),否则输出头随机初始化。

## 关联

依赖 #60272(#53020 的 re-land)。同一 `mm_device_do_normalize` 路径已有 llama-nemotron-vl、mistral3、qwen2-vl 等模型接入(见文档表)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
