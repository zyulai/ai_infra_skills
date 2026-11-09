# PR #42785: [MM][CG] Enable encoder CUDA Graph for MiniCPM-V

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @YunzhuLu |
| 合并 | 2026-09-10 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | documentation, ready, v1, multi-modality, nvidia, kimi, k3, glm, cohere, llama, qwen, deepseek |
| Diff | 素材未提供 diffstat(pack 记录为 +None/-None across None files;diff 摘录另有 12 个文件因超长被丢弃) |
| 链接 | https://github.com/vllm-project/vllm/pull/42785 |

## 一句话结论

为 MiniCPM-V 2.5/2.6/4.0 启用多模态 encoder CUDA Graph(VPM+resampler),并给 encoder graph 缓存引入第二捕获轴避免同 budget 不同 shape 的多余 padding。

## 动机

vLLM 的多模态 encoder CUDA Graph 机制(#38061,tracker #38175)此前只按 `token_budget` 单轴做图缓存键。MiniCPM-V 的 ViT 编码器(VPM)捕获 shape 还取决于离散的 patch-grid / pixel-layout 档位:相同 token budget 的输入可能需要不同捕获 shape。若强行共用一个 capture shape,会引入不必要的 padding,削弱 CUDA Graph 收益;而 MiniCPM-V 走 eager 路径则享受不到 encoder graph replay。MiniCPM-V 2.0 因早于 slice-based 视觉架构被排除;4.5 因 dynamic frame fusion 产生输入相关的 encoder shape、与静态 shape 捕获不兼容,继续走 eager(并注明可能损失性能)。

## 优化点

两条线:(1) 新增 `_MiniCPMVEncoderCudaGraphMixin`(`SupportsEncoderCudaGraph`),捕获范围覆盖 ViT encoder(VPM)与 resampler,定义 patch-grid 捕获轴(`_MINICPMV_BASE_PATCH_BUCKETS` (32,32)/(32,64)/(64,32) + 满分辨率档)、`get_encoder_cudagraph_config/budget_range/item_specs`、pixel 打包(`_mcpmv_pack_flat_pixels` 用 `F.fold` 重组 slice)与按档位解析的 capture/replay buffer;graph cache 键从 `token_budget` 扩展为 `(token_budget, secondary_capture_axis_key)`,`interfaces.py` 的 `prepare_encoder_cudagraph_capture_inputs` 全模型(cohere_compass、deepseek_ocr、ernie45_vl、gemma3/4_mm、glm4_1v、idefics3、internvl、kimi_k25/k_vl、lfm2_vl 等)统一加 `axis_keys` 参数,模型经 `ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG` 回传解析出的 per-axis key。(2) 为可捕获性重写 MiniCPM-V 热路径:capturing 期间跳过 `_adjust_pos_cache`;`tgt_sizes` 用 pinned async H2D;`pos_embed` 由 per-batch Python 循环 + `pad_sequence` 改为向量化 gather + `torch.where` 掩码;共享的 `idefics2_vision_model.get_position_ids` 也从 CPU 逐 batch 循环改为全 GPU 向量化(bucketize 批量化),并去掉 forward 中的 `torch.any(~flat_patch_mask)` 同步检查。

## 关键改动

```text
vllm/model_executor/models/minicpmv.py: 新增 _MiniCPMVEncoderCudaGraphMixin(patch-grid 捕获轴、打包与 buffer),pos_embed 向量化
vllm/model_executor/models/idefics2_vision_model.py: get_position_ids 全 GPU 向量化,消除 CPU 循环与同步
vllm/model_executor/models/interfaces.py: capture_axes/axis_keys 协议与 ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG
vllm/model_executor/models/*.py(约 11 个模型): prepare_encoder_cudagraph_capture_inputs 增加 axis_keys 形参
tests/models/multimodal/generation/test_vit_cudagraph.py: 新增 minicpmv_25/26/40 测试配置
```

关键摘录(patch-grid 捕获轴定义):

```python
return EncoderCudaGraphConfig(
    modalities=modalities,
    buffer_keys=buffer_keys,
    out_hidden_size=int(self.embed_dim),
    max_frames_per_video=max_frames,
    capture_axes=(self._mcpmv_patch_grid_keys(),),
)
```

## 性能声明

benchmark:`vllm bench mm-processor`,RTX 5090,random-mm,500 prompts / 100 warmups,`(448,448,1)` bucket。正文结论:"no measurable regression",部分 run 有小幅改善:

| 模型 | Backend | Mean | P99 |
| --- | --- | --- | --- |
| MiniCPM-V-2_6 | FLASH_ATTN2 | +0.60%(44555.14→44286.37 ms) | +0.10%(67712.77→67644.05 ms) |
| MiniCPM-V-4 | FLASH_ATTN2 | +2.01%(44723.41→43822.66 ms) | +2.69%(84580.66→82305.93 ms) |

详细分阶段数据中 `encoder_forward_ms` mean 64.14→62.67(注意:正文 MiniCPM-V-2_6 与 4_0 的详细输出块数字完全相同,疑为粘贴重复,以汇总表为准)。精度对照:功能测试(RTX 5090,MiniCPM-V-4)image/video 描述文本与 no-CUDA-Graph 基本一致。

## 验证与回归风险

单元测试 `tests/v1/cudagraph/test_encoder_cudagraph.py` 36 passed / 16 warnings;`test_vit_cudagraph.py` 新增 MiniCPM-V 2.5/2.6/4.0 配置(2.5 用 64/1024 小 budget 以适配 CI 35GB MIG 显存)。回归风险面较宽:`prepare_encoder_cudagraph_capture_inputs` 签名变化触及所有支持 encoder CUDA Graph 的多模态模型;`idefics2_vision_model.get_position_ids` 的向量化重写影响共享该 VPM 的模型(Aria、Phi4 等注明不 apply mask 的路径);去掉 `torch.any` 检查改变了 "无 padding 则跳过 mask" 的快捷路径;capture_axes 机制改变 graph cache 键,需确认旧配置(单 token_budget)行为兼容。MiniCPM-V 2.0/4.5 明确继续走 eager。

## 关联

Tracker #38175;工作流沿用 #38061。同区域为多模态 encoder CUDA Graph 支持的各模型接入。本卡所在批次素材中未列出其他同区域 PR 编号,暂无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
