# PR #35963: [Feature] ViT Full CUDA Graph

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @b-mu |
| 合并 | 2026-03-23 |
| 分类 | CUDA Graph / Compiler |
| Label | performance, ready, v1, multi-modality, qwen, nvidia |
| Diff | diffstat 缺失(pack 未提供 +N/-N 与文件数;diff 可用但注明 4 个文件因篇幅被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/35963 |

## 一句话结论

为视觉编码器(ViT)引入按 token 预算的 full CUDA graph,消除内核启动开销,GB200 上时延最多降约 44%。

## 动机

V1 中 LLM decoder 已有 CUDA graph,但多模态视觉编码器每次前向都按 eager 方式逐内核启动;ViT 层数多、内核小,launch overhead 占比高,批量图片输入时尤为明显。需要一套能把整个 encoder forward 捕获为 CUDA graph、并在不同图片数量/尺寸间复用同一 graph 的机制。

## 优化点

按"token 预算"维度而非 batch size 维度建 graph:在可配置的 token budget 列表(如 `[512, 1024, ..., 4864]`,缺省按模型推断为 2 的幂)各捕获一张 full CUDA graph;运行时把 batch 内图片按输出 token 数升序做贪心 bin-packing,装进尽量小的预算档,通过把 `cu_seqlens` 等元数据 zero + slice-copy 进固定 buffer,使同一张 graph 能重放任意图片组合(填充位置被 cu_seqlens 掩掉)。`EncoderCudaGraphManager` 完全模型无关:模型通过实现 9 个方法的 `SupportsEncoderCudaGraph` protocol 提供自己的 grid 配置、dummy 输入、metadata 计算与 forward 分派(`prepare_encoder_cudagraph_capture_inputs` 造合成网格与 dummy pixel_values,`prepare_encoder_cudagraph_replay_buffers` 复用 `prepare_encoder_metadata` 计算重放元数据);超出最大预算的单图回退 eager。支持 `mm_encoder_tp_mode=data` 时各 TP rank 以数据并行独立跑 ViT,按输入 patch 数做负载均衡分配、pad 后 `all_gather` 汇聚;并支持 FlashInfer cuDNN attention(在 graph 路径覆盖其 bucket)。qwen3_vl.py 将原 forward 内联的 metadata 计算抽成 `prepare_encoder_metadata`,eager/capture/replay 三路共享;`max_seqlen` 特意留在 CPU(attention wrapper 会 `.item()`,放 GPU 会把无意义的 D2H 拷贝烧进 graph)。

## 关键改动

```text
vllm/v1/worker/gpu/mm/encoder_cudagraph.py: EncoderCudaGraphManager(capture/replay/贪心装箱/DP)
vllm/model_executor/models/interfaces.py: SupportsEncoderCudaGraph protocol(9 方法)+ type guard
vllm/model_executor/models/qwen3_vl.py: Qwen3VL 实现协议,抽出 prepare_encoder_metadata
vllm/config/compilation.py: cudagraph_mm_encoder / token_budgets / max_images_per_batch 配置
vllm/v1/worker/gpu_model_runner.py: 接入 V1 model runner
```

```python
# 重放模式:copy 输入 → copy 元数据 → replay → 读输出
graph_meta.input_buffer.zero_()
graph_meta.input_buffer[:n].copy_(src)
for key in self.config.buffer_keys:
    ...
    buf.zero_()
    buf[:n].copy_(src)
graph_meta.graph.replay()
```

## 性能声明

(`vllm bench mm-processor`,均值/时延,越低越好;PR 以"+X%"表述改善幅度)

- 单卡:Qwen3-VL-30B-A3B-Instruct,1x GB200,VisionArena-Chat 3000 prompts(+300 warmup),FLASH_ATTN:Mean +11.8%(5.13→4.52ms)、P99 +31.6%(9.16→6.26ms);另一组 FLASH_ATTN 行:Mean +19.6%(5.42→4.36ms)、P99 +40.3%(10.87→6.49ms)(两行配置差异 PR 未注明)。
- 多卡:Qwen3-VL-32B-Instruct,4x GB200 TP=4 + ViT DP=4,random-mm(每请求 20 图,336x336)1000 prompts:FLASH_ATTN Mean +18.4%(28.39→23.16ms)、P99 +14.0%(238.78→205.28ms);FLASHINFER Mean +44.4%(23.24→12.91ms)、P99 +84.9%(172.41→26.05ms)。

## 验证与回归风险

单测/GPU 测试:`pytest tests/v1/cudagraph/test_encoder_cudagraph.py`(本 PR 新增,含 unit 与 GPU 测试)。风险面:默认关闭(`cudagraph_mm_encoder: false`),需显式开启;目前仅 Qwen3-VL 实现 protocol,其他模型不受影响;贪心装箱超预算、max_seqlen 被 capture 固化等场景依赖 eager 回退与 override 逻辑正确;DP 路径的 all_gather pad/unpad 与 eager 的 `run_dp_sharded_mrope_vision_model` 采用同一算法。

## 关联

后续同区域工作:#65778(GLM 系列视觉编码器接入 FlashInfer ViT 后端的兼容性修复,其素材引用了 encoder CUDA-graph 路径 #49852 与本特性的测试体系);#65453(encoder graph 场景下 FlashInfer cu_seqlens 的 capture/replay 对齐)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
