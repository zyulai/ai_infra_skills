# PR #13055: [VLM] Support Piecewise CUDA Graph for Qwen2.5-VL

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-11-20 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | high priority, performance, Multi-modal, ready-to-merge, run-ci, vlm, piecewise-cuda-graph |
| Diff | 素材未提供 diffstat(diff 正文含 11 个文件,另有 1 个文件因体积被省略) |
| 链接 | https://github.com/sgl-project/sglang/pull/13055 |

## 一句话结论

让 Qwen2.5-VL 走 piecewise CUDA graph,prefill TTFT 降 23.8%、输入吞吐升 21%。

## 动机

SGLang 与 vLLM 在 VLM 上对比时,一个关键差距是 piecewise CUDA graph 支持:它能显著减少 prefill 模式下小 kernel 的 launch 时间,从而降低 TTFT(对应 issue #12838)。此前 SGLang 的 `general_mm_embed_routine()` 每次都调用 `embed_mm_inputs` 分配全新的 `inputs_embeds` 张量,设备地址不固定,导致无法对整个 VLM 做 CUDA graph 捕获。Torch Profiling 显示无该特性时出现 "CUDA Command Buffer Full",GPU 不能持续高效工作。

## 优化点

参照 vLLM 的做法(在 model runner 的 `_preprocess` 中执行 encode、跳过 ViT 的 CUDA graph),本 PR 把 `general_mm_embed_routine()` 拆成两部分:多模态 embedding 部分(`external_embed_mm_inputs`/`external_mm_preprocess_routine`)移到 VL 模型之外,在 `model_runner.forward_extend` 中预先执行;语言模型部分留在 VLM 模型 forward 内。这样 piecewise CUDA graph 只作用于 LLM 部分、避开 ViT 逻辑。相应地,piecewise CUDA graph runner 增加了固定地址的 `input_embeds` 与 `mrope_positions` graph 输入张量,replay 时用 `input_embeds` 替代 `input_ids` 作为捕获输入。`should_use_external_mm_preprocess` 目前仅对 Qwen2-VL/Qwen2.5-VL 类启用外部预处理,Llava 系列仍走原路径。

## 关键改动

```text
python/sglang/srt/managers/mm_utils.py: 新增 external_embed_mm_inputs、
  external_mm_preprocess_routine、should_use_external_mm_preprocess、
  resolve_language_model,把 mm embedding 移出模型 forward
python/sglang/srt/model_executor/model_runner.py: forward_extend 中先做外部 mm 预处理
python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py:
  增加 input_embeds、mrope_positions 静态输入与拷贝逻辑
python/sglang/srt/models/qwen2_5_vl.py / qwen2_vl.py:
  forward 改为直接调 self.model 并接收 input_embeds
python/sglang/srt/layers/rotary_embedding.py: 去掉 dynamo/compile 标记以兼容 capture
```

model_runner 中的接入点:

```python
if self.is_multimodal and should_use_external_mm_preprocess(self.model):
    forward_batch = external_mm_preprocess_routine(
        forward_batch=forward_batch,
        multimodal_model=self.model,
    )
```

## 性能声明

bench_serving,image 数据集,256 prompts,random-input-len 128 / output-len 32,image 560x560 x1,concurrency 32,attention-backend fa3,chunked-prefill-size 8192:

- Prefill TTFT:1135.70ms → 865.97ms,23.8% speedup
- Input throughput:3932.23 → 4760.27 token/s,21% speedup
- Request throughput 7.97 → 9.65 req/s;Mean E2E 3974.76 → 3276.21ms

注意:PR 正文两个 server 配置不一致——"Without PCG" 写的是 Qwen2.5-VL-72B-Instruct / tp8,"With PCG" 写的是 Qwen2.5-VL-7B-Instruct / tp2,PR 未注明哪组为准;精度测试(lmms_eval mmmu_val)用 Qwen2.5-VL-7B-Instruct,tp2,mmmu_acc 0.5044。

## 验证与回归风险

新增 `test/srt/nightly/test_vlms_piecewise_cuda_graph.py`:Qwen2.5-VL-7B-Instruct,tp8 + piecewise CUDA graph,跑 lmms_eval mmmu_val,阈值 mmmu_accuracy 0.60。`test/srt/test_piecewise_cuda_graph.py` 新增 `TestPiecewiseCudaGraphQwen25VL` 用 GSM8K/MGSM 验证精度;`run_suite.py` 将 test_piecewise_cuda_graph.py 超时从 600s 提到 750s 并注册 nightly 测试。风险面:rotary_embedding.py 移除了 `@torch._dynamo.disable()` 与 `@torch.compile(dynamic=True)` 标记,影响所有走 triton mrope 路径的模型;Qwen2-VL/Qwen2.5-VL 的 forward 签名与 embedding 流程改变,其他 VLM(Llava 系)依赖 `should_use_external_mm_preprocess` 返回 False 保持原逻辑。

## 关联

解决 issue #12838(SGLang vs vLLM VLM 性能对比),与 @yhyang201 协作;正文还提到若配合 CUDA IPC 特性协调,性能可进一步提升。同分类(piecewise-cuda-graph)其他 PR 编号素材未列出,写"无"。
