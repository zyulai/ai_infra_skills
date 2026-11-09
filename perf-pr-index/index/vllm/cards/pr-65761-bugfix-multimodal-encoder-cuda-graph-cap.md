# PR #65761: [Bugfix][Multimodal] Encoder CUDA graph: cap each graph's frame slots at its token budget

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Pb314314 |
| 合并 | 2026-11-05 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | bug, ready, nvidia |
| Diff | diffstat 缺失(pack 中为 +None/-None,None files;diff 完整可用,共 2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/65761 |

## 一句话结论

修复 encoder CUDA graph 按全局最大帧数捕获导致的巨型 graph replay:每个 graph 的 frame slots 按自身 token budget 封顶,GLM-4.1V 图像请求中位延迟 470ms→31ms。

## 动机

开启 encoder CUDA graph 且模型支持视频时,`EncoderCudaGraphManager` 给每个 token budget 的捕获都传同一个 `max_frames_per_batch = max_batch_size * max_frames_per_video`(#40445 引入的默认)。对 GLM-4.1V 这是 64 x 600 = 38,400 帧,于是最小的 graph 也至少计算 38,400 token,任何图像/视频请求都会 replay 一个远大于自身输入的 graph,graph 路径反而比 eager 慢约 8x。

## 优化点

核心观察:每帧至少产出一个输出 token,能装进 budget B 的 batch 至多只有 B 帧。因此捕获每个 budget 的 graph 时用 `min(max_frames_per_batch, token_budget)` 作为 frame slots 数,该值存入 `BudgetGraphMetadata`;replay 时按 graph 自身存的 `max_frames_per_batch` 对 `cu_seqlens` 做 padding。模型侧捕获布局代码不变,是随传入值自然缩小。所有 auto/validated 配置保证 budget >= max_batch_size,graph 不会比 max_batch_size 少帧,image replay 不受影响。该 manager 由 Model Runner V1/V2 共享(经 `vllm/v1/worker/gpu/mm/encoder_runner.py`),两者同时受益;改动不引入 host sync。注意 video 关闭或 `max_frames_per_batch` 不超过 budget 时无效果。

## 关键改动

```text
vllm/v1/worker/encoder_cudagraph.py: 核心修复。_capture_budget_graph 里 max_frames_per_batch = min(self.max_frames_per_batch, token_budget),存入 BudgetGraphMetadata;_run_budget_graph replay 改用 graph_meta.max_frames_per_batch
tests/v1/cudagraph/test_encoder_cudagraph.py: 新增 test_frame_slots_capped_by_budget(max_frames_per_batch=1000 捕获,断言每个 graph 为 min(1000, budget) 帧槽且视频 batch 仍可 replay)
```

关键代码:

```python
# Every frame yields at least one token, so a batch that fits the
# budget has at most token_budget frames.
max_frames_per_batch = min(self.max_frames_per_batch, token_budget)
```

## 性能声明

GB200、bf16、默认 ViT attention backend,基线为 main@2a777ce26e;每 run 单引擎、24 image + 24 video 请求逐条发送(max_tokens=1),输入为随机新鲜数据(无 encoder-cache 命中),两轮取中位:GLM-4.1V-9B(max_model_len=8192)图像 469.8/480.6ms → 31.9/31.2ms,视频 475.4/484.2ms → 35.9/35.2ms(eager 为 56.8/60.7ms);Qwen2.5-VL-7B(128000)图像 32.6/30.3ms → 27.2/27.1ms,视频 32.1/31.9ms → 30.8/29.7ms。正确性:GLM-4.1V graph 输出与 eager 逐 bit 一致;Qwen2.5-VL 与 main 的 graph 输出逐 token 一致(12/12;两者与 eager 各有 2 条在 eager top-2 logprobs 完全打平处分歧,属 main 已有 bf16 漂移,与本 PR 无关)。

## 验证与回归风险

`pytest tests/v1/cudagraph/test_encoder_cudagraph.py` 79 passed,pre-commit 通过。风险面:改动位于 V1/V2 共享的 encoder cudagraph 捕获/replay 路径,影响所有启用 `cudagraph_mm_encoder` 的多模态模型;依赖"每帧至少一个 token"的假设,若某模型布局不满足会拿到偏小的捕获尺寸(PR 称模型捕获布局由传入值派生,未改)。PR 同时指出 main 上 GLM-4.1V 图像 replay 数值错误(graph vs eager 相对 L2 误差最高 169%,576-token 图像)不属本修复,由 #65453 注册 cu_seqlens padding 解决。

## 关联

同区域:#40445(引入当前 max_frames_per_batch 默认值)、#65453(cu_seqlens padding 修复,同 padding 路径但无共同代码行)、#60112(replay 超捕获 buffer 时回退 eager,open)、#43403(closed draft)。分类同属 CUDA Graph / Compiler 的其他 PR 见索引。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
