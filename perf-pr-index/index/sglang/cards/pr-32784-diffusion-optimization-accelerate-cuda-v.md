# PR #32784: [diffusion] optimization: accelerate CUDA video output finalization

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-07-29 |
| 分类 | Other (major) |
| Label | run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 含 2 个文件:输出保存 helper 与其单测 |
| 链接 | https://github.com/sgl-project/sglang/pull/32784 |

## 一句话结论

加速扩散模型视频输出的落盘尾段:单遍 FFmpeg 编码 + CUDA 注册 memfd 直通,124 帧输出 1.591s→0.621s(2.56x)。

## 动机

视频生成请求的输出保存(finalization)此前走 imageio 路径:先把 CUDA 张量经 pageable CPU 张量物化成帧列表,再写一个无声视频、最后 remux 音频,存在多次拷贝与两遍编码开销。VAE 解码之后的这段 CPU/IO 尾巴在短 denoise 时长的 diffusion 服务里占比可观,值得优化。

## 优化点

四层改动:(1) 有音频时用 imageio 的一次 FFmpeg pass 直接编码 H.264/AAC,替代"先写无声视频再 remux"两遍流程;(2) 既有物化路径改用 pinned CPU 内存做 D2H 拷贝(non_blocking + stream 同步);(3) 对受支持的 Linux/CUDA MP4 请求(不要求帧插值、超分、不要求返回解码帧),新增直通路径:CUDA 张量 quantize/permute 后 `copy_` 进一个 `cudaHostRegister` 注册的 memfd 映射缓冲,FFmpeg 直接以 `/proc/self/fd/N` 作为 rawvideo 输入单遍编码,跳过中间 pageable CPU 张量与 Python pipe;(4) 形状感知的注册缓冲按 shape 复用(上限 1 GiB,atexit 清理),x264 线程数按高度与 CPU 亲和度有界选择。任何不支持的配置或运行时失败都回退既有 materialized/imageio 路径;音频保持 AAC,无音频请求行为不变。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/entrypoints/utils.py: _CudaMemfdVideoBuffer(memfd+cudaHostRegister)、_acquire_cuda_video_buffer 形状复用、_try_save_cuda_video_direct 直通编码、_try_save_video_with_audio 单遍编码、pinned 内存物化
python/sglang/multimodal_gen/test/unit/test_output_saving.py: 单遍编码、回退、线程数、直通短路物化四组单测
```

直通路径核心:

```python
with _acquire_cuda_video_buffer(shape) as buffer:
    buffer.tensor.copy_(frames, non_blocking=True)
    torch.cuda.current_stream(frames.device).synchronize()
    command = [ffmpeg_exe, "-f", "rawvideo", "-s", f"{width}x{height}",
               "-pix_fmt", "rgb24", "-i", f"/proc/self/fd/{buffer.fd}", ...]
    subprocess.run(command, check=True, pass_fds=(buffer.fd,), ...)
```

## 性能声明

代表性 124 帧 1344x768 H.264/AAC 输出:

- 旧路径:median **1.591 s**
- registered-memfd 直通路径:median **0.621 s**
- speedup:**2.56x**

产物经完整解码验证。PR 未注明测试硬件(GPU 型号/CPU)与更多 workload 条件;除上述单一场景外无其他 benchmark。

## 验证与回归风险

输出保存与输出物化两个单测套件共 **19 passed**;两个改动文件 pre-commit 通过;公开 diff 限于输出 helper 与其单测。新增单测覆盖:有音频走单遍编码(且断言旧两遍 mux 不再运行)、单遍失败回退两遍、`_x264_auto_thread_count` 参数化(高度×CPU 数)、直通保存成功时跳过帧物化。CI:PR Test (Base) 通过、PR Test (Extra) 失败。风险面:直通路径条件严格(Linux memfd、MP4 后缀、3 通道 CUDA 张量、quality 1-10),失败静默回退旧路径并 warning;注册缓冲为进程级单例、按 shape 匹配复用,shape 变化时关闭重建;memfd/注册失败时缓冲构造即抛错并关闭。

## 关联

同区域:multimodal_gen 输出保存/物化基础设施、Wan/其他 diffusion 模型的输出后处理(pack 未列具体编号)。与 VAE/推理 kernel 类优化互补——本卡优化的是生成完成后的落盘段,不影响 denoise/decode 计时。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
