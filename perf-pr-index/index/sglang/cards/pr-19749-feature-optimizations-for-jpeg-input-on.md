# PR #19749: [Feature] Optimizations for JPEG input on NVIDIA GPU

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @wili-65535 |
| 合并 | 2026-03-29 |
| 分类 | Model-specific Fast Path |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/19749 |

## 一句话结论

用 nvJPEG 在 GPU 上直接解码 JPEG 并输出 torch tensor, eliminating PIL/CPU 中转,H100 上 QwenVLProcessor 图像处理平均提速 1.5x、极端场景可达 3.8x。

## 动机

issue #18784 与 PR #18559 的第一部分。此前 JPEG 输入经 PIL 解码为 CPU tensor,再搬到 GPU,存在 PIL Image、CPU tensor 等中间格式与多余的 CPU-GPU 传输;大分辨率图片下 QwenVLProcessor 的 process_time 高达数百 ms,是 VLM prefill 的显著瓶颈。

## 优化点

`load_image` 增加 `gpu_image_decode` 参数与内部 `_load_image`:当 CUDA 可用、字节流符合 JPEG 魔数(`\xff\xd8 ... \xff\xd9`)且处理器未显式关闭时,经 `torchvision.io.decode_jpeg(device="cuda")`(对应 `torch.ops.image.decode_jpegs_cuda`)调用 nvJPEG 硬件解码,直接产出 GPU 上的 torch tensor;解码失败或非 JPEG 时回退 PIL CPU 路径(nvJPEG 对不严格合规的 JPEG 更苛刻,PIL 更宽容)。`BaseMultimodalProcessor` 增加 `gpu_image_decode = True` 类属性作默认开关,不支持 tensor 输入的处理器显式置 False(InternVL、KimiVL/K2.5、LLaVA、MiniCPM、Pixtral、NanoNemotronVL 等);`_load_single_item` 从 staticmethod 改为 classmethod 以读取该开关,RGB 转换仅在 PIL 图像上执行;`encode_server` 分离式编码路径同步适配。

## 关键改动

```text
python/sglang/srt/utils/common.py: _load_image/is_jpeg_with_cuda 与 load_image 重构
python/sglang/srt/multimodal/processors/base_processor.py: gpu_image_decode 开关与 classmethod 化
python/sglang/srt/multimodal/processors/{internvl,llava,minicpm,...}.py: 不支持 tensor 输入的处理器置 False
python/sglang/srt/disaggregation/encode_server.py: 分离式编码路径适配
```

```python
if is_jpeg_with_cuda(image_bytes, gpu_image_decode):
    try:
        encoded_image = torch.frombuffer(image_bytes, dtype=torch.uint8)
        image_tensor = decode_jpeg(encoded_image, device="cuda")
        return image_tensor
    except Exception as e:
        logger.warning(f"Failed to decode JPEG on GPU, falling back to CPU. Error: {e}")
return Image.open(BytesIO(image_bytes))
```

## 性能声明

H100,Qwen3-VL-8B,单张 JPEG 图片请求,观察 QwenVLProcessor 日志(load/process/total time):平均 1.5x 加速,图片越大收益越大,极端场景最高 3.8x。逐尺寸表:1024x1024 total 14.60→7.24 ms(2.02x),992x992 14.34→7.36(1.95x),512x512 4.77→3.29(1.45x),小图 32x32 持平、64x64 反而 0.74x(2.16→2.91 ms)。E2E 示例(两张 2048x2048 图):request total_time 300.03→270.79 ms。精度:lm_eval GPQA 六个任务与 lmms_eval mmmu_val 前后分数一致(mmmu 0.5022,GPQA 各任务无变化)。

## 验证与回归风险

验证为 lm_eval/lmms_eval 前后对比;无新增单测(checklist 该项未勾选)。风险面:默认开启 gpu_image_decode,不支持 tensor 输入的处理器必须正确置 False,否则后续 HF processor 会收到 GPU tensor;nvJPEG 对非标准 JPEG 有失败率,依赖 warning 日志 + PIL 回退保证正确性;CPU 解码返回类型从恒为 PIL 变为"PIL 或 torch.Tensor",所有下游 `_load_single_item` 调用方需兼容两种类型。

## 关联

系列第一部分,对应 issue #18784 与 PR #18559(性能数据亦在彼处)。同区域(VLM 预处理优化)索引中另有 pr-12240、pr-13487、pr-17420 等卡片。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
