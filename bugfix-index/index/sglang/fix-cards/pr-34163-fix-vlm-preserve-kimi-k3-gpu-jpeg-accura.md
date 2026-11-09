# PR #34163: fix(vlm): preserve Kimi-K3 GPU JPEG accuracy

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34163](https://github.com/sgl-project/sglang/pull/34163) |
| 作者 | mickqian |
| 合并日期 | 2026-08-10 |
| labels | run-ci, run-ci-extra |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

GPU nvJPEG 对 4:2:0 JPEG 用最近邻色度上采样致 Kimi-K3 精度偏离 PIL;改用 nvImageCodec fancy upsampling + DLPack 零拷贝修复。

## 问题现象

Kimi-K3 走 GPU JPEG 解码路径(`torchvision.io.decode_jpeg(..., device="cuda")`)时,常见 4:2:0 JPEG 的解码结果与 PIL 及 Kimi 参考处理器不一致,造成精度损失。对比实验隔离了损失来源:CPU torchvision 与 4:4:4 JPEG 均与 PIL 一致,说明损失只出在 JPEG 色度重建而非 K3 归一化。全部 1,000 张 OCRBench JPEG 上,原 GPU 路径相对 PIL 的平均原始像素 MAE 为 0.476081(p99 3.061401,最大 93)。

## 根因

按 PR 正文(Root cause 一节):`torchvision.io.decode_jpeg(..., device="cuda")` 以 nvJPEG 默认 flags 创建解码器,对常见 4:2:0 JPEG 使用最近邻(nearest-neighbor)色度上采样;而 PIL 与 Kimi 参考处理器使用插值式(interpolated)色度重建。两者不一致导致 K3 输入像素差异,进而影响精度。

## 修复方案

- 新增 `python/sglang/srt/utils/nvjpeg_decoder.py`:通过 NVIDIA nvImageCodec 启用 nvJPEG 插值色度上采样(`options=":num_cuda_streams=1 :fancy_upsampling=1"`、`SampleFormat.P_RGB`、不做 EXIF 旋转),经 DLPack 零拷贝交给 PyTorch;每 GPU 解码器池上限 2 个实例以约束 HBM 开销(每个解码器约占 15–25 MiB)。
- `python/sglang/srt/utils/common.py`:`gpu_image_decode` 类型扩展为 `Union[bool, Literal["nvjpeg_fancy"]]`,`_load_image` 在该模式下调用新解码器,依赖或 codec 不可用时回退 PIL 并做去重告警。
- `python/sglang/srt/multimodal/processors/kimi_k3.py`:`gpu_image_decode` 从 `True` 改为 `"nvjpeg_fancy"`;`python/sglang/srt/disaggregation/encode_server.py` 的 `_load_single_item` 对 kimi_k3 + GPU processor 传同一模式,普通 serving 与 GPU encoder 分离(encoder disaggregation)走同一路径。
- 两个 Kimi-K3 Dockerfile(cu12/cu13)安装 pinned `nvidia-nvimgcodec-cu12/cu13[all]==0.9.0.20`。

真实权重验证(TP8 双 4×GB300,OCRBench 1,000 图):GPU fancy nvJPEG 与官方 PIL 配对确定性评测同为 0.895,每方向各 6 例翻转恰好相消,零 prompt token 数不匹配;解码吞吐还快 2.44%(1,815 vs 1,772 img/s),完整 GPU 预处理约 18.6 倍于 PIL。

## 影响与回归风险

- 影响窗口:正文未给出版本范围;影响开启 GPU 图像解码的 Kimi-K3 serving(含 encoder disaggregation 模式)。
- 回归面:Kimi-K3 图像预处理路径与镜像依赖。其他模型/处理器的 `gpu_image_decode` 默认行为不变;依赖缺失时回退 PIL 保证可用性但性能回落。一个精确复刻 Pillow 定点 bicubic 的 GPU kernel 原型因慢 11.4% 且无配对精度增益被有意排除;双解码器池约保留 30–50 MiB HBM/进程。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
