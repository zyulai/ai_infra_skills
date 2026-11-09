# PR #15320: [VLM] Support ViT Piecewise CUDA Graph for Qwen3-VL

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-12-20 |
| 分类 | CUDA Graph / Compiler |
| Label | performance, Multi-modal, run-ci, vlm, piecewise-cuda-graph |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与文件数统计);diff 本身可用,涉及 6 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/15320 |

## 一句话结论

为 Qwen3-VL 启用 ViT Piecewise CUDA Graph,8xH20 TP=4 下 TTFT 1384.53ms → 1120.68ms(约 -19%)。

## 动机

ViT prefill 在每个 rank 计算量较小时(TP=4、图像不大、prefill 非 compute bound)kernel launch 占据大头,把 ViT 捕获进 CUDA Graph 可省掉这部分时间。此前 `ViTCudaGraphRunner` 只支持 Qwen2.5-VL,本 PR 将其泛化以同时支持 Qwen2.5-VL 与 Qwen3-VL,并放开 TP>1。另顺带修复一个 bug:outplace allreduce 未把 torch-symm-mem 列为可用后端,导致其不生效(对比 NCCL 约有 4% e2e 提升)。

## 优化点

将 `ViTCudaGraphRunner` 泛化为 generic runner:按 `graph_key = seq_len` 捕获 "blocks + merger + deepstack merger(可选)" 为 CUDA graph 并 replay。对 Qwen2.5-VL 保留 windowed attention 分支(`fullatt_block_indexes`,同时提供 `cu_seqlens` 与 `cu_window_seqlens`);为 Qwen3-VL 新增 deepstack 支持(`deepstack_visual_indexes` + `deepstack_merger_list`),graph 内每经过一个 deepstack 层就跑一次对应 merger,输出改为 `torch.cat([main_out] + deepstack_outs, dim=1)`。position embedding 支持两种形态:Qwen2.5 的 `(cos, sin)` 元组与 Qwen3-VL 的 `rotary_pos_emb_cos/sin`,两者互斥,均先拷入常驻 workspace 再 capture/replay。`qwen3_vl.py` 新增 `forward_with_cuda_graph` 入口,patchify/pos-embed/rotary/cu_seqlens 仍在 graph 外完成。同时移除 vision attention 与模型 forward 中 `tp_size == 1` 的限制,使 ViT CUDA graph 在 TP>1 下生效。

## 关键改动

```text
python/sglang/srt/multimodal/vit_cuda_graph_runner.py: 泛化 runner,支持 deepstack 与 rotary_pos_emb 输入
python/sglang/srt/models/qwen3_vl.py: 接入 ViTCudaGraphRunner,新增 forward_with_cuda_graph
python/sglang/srt/models/qwen2_5_vl.py: 适配新 runner 接口
python/sglang/srt/layers/attention/vision.py: 去掉 ViT CUDA graph 的 tp_size==1 限制
python/sglang/srt/distributed/parallel_state.py: outplace allreduce 允许 torch_symm_mem_comm
test/manual/nightly/test_vlms_vit_cuda_graph.py: 增加 Qwen3-VL-8B-Instruct 精度 case
```

```python
if deepstack_outs:
    self.block_output[graph_key] = torch.cat([main_out] + deepstack_outs, dim=1)
else:
    self.block_output[graph_key] = main_out
```

## 性能声明

- 8xH20,Qwen3-VL-8B-Instruct,TP=4:TTFT 1384.53ms → 1120.68ms(两侧均开 torch-symm-mem,唯一变量为是否启用 ViT Piecewise CUDA Graph)
- torch-symm-mem 相比 legacy NCCL TP 额外约 4% e2e 提升(需 `--disable-custom-all-reduce`,因 custom all-reduce 尚不支持 CUDA Graph)
- benchmark 条件:bench_serving image dataset,256 prompts,input 128 / output 1 tokens,560x560 单图 jpeg,concurrency 32;PR 未注明精度/量化配置

## 验证与回归风险

`test/manual/nightly/test_vlms_vit_cuda_graph.py` 覆盖精度(Qwen2.5-VL-7B 与 Qwen3-VL-8B,mmmu_accuracy 0.60 阈值)。风险:ViT CUDA graph 路径从 TP=1 扩展到 TP>1,影响 Qwen2.5-VL/Qwen3-VL 的 prefill;graph 按 seq_len 缓存,新图像尺寸会触发 capture;依赖 `output_ws` workspace 传入约定(cuda-graph 模式下缺失会报错)。

## 关联

前序:ViTCudaGraphRunner 最初为 Qwen2.5-VL 引入(编号素材未注明);同区域:CUDA IPC 多模态传输优化 PR #21418 亦在 multimodal 服务路径。同分类其他 PR 编号素材未提供。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
