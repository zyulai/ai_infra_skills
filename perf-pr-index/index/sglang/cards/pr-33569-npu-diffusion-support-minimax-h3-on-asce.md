# PR #33569: [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @OrangeRedeng |
| 合并 | 2026-08-25 |
| 分类 | Model-specific Fast Path |
| Label | documentation, quant, amd, npu, run-ci, diffusion, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用(9 个文件从 diff 中截断) |
| 链接 | https://github.com/sgl-project/sglang/pull/33569 |

## 一句话结论

把 MiniMax-H3 视频生成管线移植到 Ascend NPU,并新增 Laser Attention 变长打包路径,T2VA 生成最高提速 1.45x。

## 动机

MiniMax-H3 将真实多模态 token 与序列并行对齐 padding 打包在一起,NPU attention 路径必须隔离真实段与 padding 段;同时需要设备无关的 RNG/autocast/显存驻留处理,以及 CUDA/ROCm 专用融合 kernel 缺失时的安全回退。本 PR 把原生 MiniMax-H3 管线(#33275)扩展到 Ascend NPU。

## 优化点

`laser_attn` 后端新增 `forward_varlen`:识别 MiniMax-H3 的 `[0, used, padded]` 三段 cu_seqlens 约定,只对真实 token 段调用 dense Laser attention,padding 段输出置零,避免真实 token attend 对齐 padding 且不在 padding 上浪费算力;无 padding 时跳过分配/拷贝。数值上:NPU Laser kernel 对 BF16 输入内部用 FP16 存 QK/V,新增 256 倍 Q/K 与 value/output scale 补偿(`_BF16_LASER_SCALE`)保住 BF16 激活范围。其他:MiniMax-H3 使用 laser_attn 时,Qwen3-VL 文本编码器自动改用 `torch_sdpa`(较 Ascend FA 更快);keyframe RNG fork、decode autocast、组件显存释放检查推广到平台无关(支持 NPU 设备类型);VAE gated FFN 复用 NPU `SiluAndMul` 路径;ffmpeg/ffprobe 缺失时报可操作错误;新增 Ascend cookbook 与 CI 用例 `minimax_h3_t2va_2npu`(TP2+SP1,性能与视觉一致性对 pinned GT 校验)。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/attention/backends/laser_attn.py: forward_varlen 打包段隔离 + BF16 scale 补偿
python/sglang/multimodal_gen/runtime/server_args/server_args.py: 文本编码器自动切 torch_sdpa
python/sglang/multimodal_gen/runtime/pipelines_core/.../minimax_h3/*: RNG/autocast/denoise 平台适配
python/sglang/multimodal_gen/test/server/ascend/perf_baselines_npu.json: NPU 扩散基线刷新
```

```python
padding_start = (bounds[1]
    if len(bounds) == 3 and bounds[0] == 0 and bounds[-1] == query.shape[0]
    else None)
# 真实段单独跑 dense attention,padding 段置零
segment = self._forward_dense(query[:padding_start].unsqueeze(0), ..., preserve_bf16_range=True)[0]
output[:padding_start].copy_(segment)
output[padding_start:].zero_()
```

## 性能声明

手动测量:同一 5 秒 T2VA 请求(1344x768、124 帧、24 FPS、50 步、seed 1101)。Laser vs Ascend FA:8 NPU(TP2+SP4)69.51 s → 55.07 s(1.26x);4 NPU(TP2+SP2)150.44 s → 103.57 s(1.45x);2 NPU(TP2+SP1)274.44 s → 190.20 s(1.44x)。Cache-DiT 叠加:8 NPU 上 Laser 158.71 s → Laser+Cache-DiT 55.07 s(2.88x)。自动化 2-NPU CI 基线:e2e 155.11 s,平均 denoise 步 2749.56 ms,中位 613.17 ms(用于回归检测,与手动拓扑对比分开)。

## 验证与回归风险

e2e 验证 5 秒 T2VA 生成(8 NPU 上 Ascend Flash Attention 与 Laser Attention 两种后端)与图像条件 FL2VA(Laser);CI 用例含性能与视觉一致性检查(pinned Ascend GT `ci-data-diffusion@7df858e`);刷新 NPU 扩散各模型性能基线(flux/wan2.1 等,基线生成脚本改指平台专属基线文件)。风险面:laser_attn 的 varlen 语义、BF16 scale 补偿只针对当前 NPU kernel 的 FP16 行为,换 kernel 版本需重验;文本编码器自动切 torch_sdpa 只在 MiniMax-H3 + laser_attn 组合触发;组件显存释放对 NPU 不做 empty_cache。pack 显示 CI Extra 与 AMD ROCm 7.2 失败、Base 通过。

## 关联

前序 #33275(原生 MiniMax-H3 管线);相关 #33827(Cache-DiT 支持)、#33927(FFmpeg 输出 issue 跟踪)、#33366(XPU 使能)、#34855(CI 基线修复)。同分类其他 PR 素材未提供。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
