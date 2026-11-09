# PR #14422: [VLM] Support VLM ViT Piecewise CUDA Graph

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-12-13 |
| 分类 | MoE |
| Label | performance, Multi-modal, run-ci, vlm, piecewise-cuda-graph |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/14422 |

## 一句话结论

把 Piecewise CUDA Graph 扩展进 VLM 的 ViT 编码器,消除 kernel launch 间隙,8xH20 TP2 下 TTFT 降 3.8%。

## 动机

此前 VLM 只在 LLM 部分支持 Piecewise CUDA Graph;ViT 因内部部分张量地址不固定而未被纳入,profiling 显示 ViT 的 kernel launch 存在间隙。本 PR 让 ViT 也进图,使 kernel 连续发射;首发支持 Qwen2.5-VL,Triton Attention 与 FA3 两种 MM-Attention 后端。TP>1 现在可用,因为默认禁用了 custom all-reduce。

## 优化点

新增 `ViTCudaGraphRunner`:以 seq_len 为 graph key,每个长度 capture 一次图;为图回放预分配地址稳定的 input/attn output workspace/output buffer 与 cu_seqlens 常量缓冲,rotary 的 cos/sin 使用跨图共享、按 max_context_len 预留的工作区(replay 时 copy_ 更新内容)。`VisionAttention` 的 Triton 与 FA3 后端各加 cuda-graph 分支:不再在 forward 里分配输出或对 cu_seqlens 做 `.cuda()`/`.max().item()` 同步,改用调用方传入的 `output_ws` 与预算好的 cu_seqlens 列表(Triton 传 `[cu, cu_kk, max_len]`,FA3 传 `[cu, max_len]`);Qwen2.5-VL 的 `Qwen2_5_VisionTransformerPretrainedModel` 新增 `forward_with_cuda_graph`,在 `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` 且 attention tp_size==1 时把 patchify/rotary/window index 之后的主干交给 runner。非图路径行为不变。

## 关键改动

```text
python/sglang/srt/multimodal/vit_cuda_graph_runner.py: 新增 ViTCudaGraphRunner(capture/replay/稳定地址缓冲)
python/sglang/srt/layers/attention/vision.py: Triton/FA3 后端 cuda-graph 分支与 output_ws 传递
python/sglang/srt/models/qwen2_5_vl.py: forward_with_cuda_graph 接入
test/manual/nightly/test_vlms_vit_cuda_graph.py: MMMU 门槛测试
```

```python
if get_bool_env_var("SGLANG_VIT_ENABLE_CUDA_GRAPH") and self.tp_size == 1:
    return self.forward_with_cuda_graph(x, grid_thw)
...
return self.cuda_graph_runner.run(
    x=x, position_embeddings=position_embeddings,
    cu_seqlens=cu_seqlens, cu_window_seqlens=cu_window_seqlens,
    output_indices=reverse_indices)
```

## 性能声明

8 x H20,TP=2,bench_serving image 数据集(256 prompts,560x560 JPEG,cc32):TTFT 下降 3.8%。torch profiling 对比图显示 ViT kernel 从带间隙发射变为连续发射。单请求日志示例(Qwen2.5-VL-7B):处理 970-token 图像请求约 0.65~0.76 s 返回。PR 未给出 before/after 的完整吞吐表,其余负载条件(具体 GPU 单卡型号)未注明。

## 验证与回归风险

新增 nightly 手动测试 `test_vlms_vit_cuda_graph.py`:Qwen2.5-VL-7B-Instruct 跑 lmms_eval mmmu_val,断言 accuracy >= 0.60,开启 `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` 与 piecewise cuda graph。风险面:仅支持 Triton Attention 与 FA3(其他 mm_attention_backend 直接 RuntimeError);仅 tp_size==1(attention 侧)或 use_data_parallel 时启用;按 seq_len 逐长度 capture,首见长度会有 capture 开销,图缓冲常驻显存;窗口注意力与 full attention 分别用不同的 cu_seqlens 缓冲。pack 未显示 CI 状态。

## 关联

同区域:同批 #17255(ViT CUDA Graph 在 TP 下补 ca_comm.capture() 的修复,即建立在本 PR 引入的 runner 之上)。索引中另有后续各模型的 ViT piecewise/cuda graph 卡(如 pr-16732、pr-15320)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
