# PR #65227: [Perf][Multimodal] Encoder CUDA graphs: one replay per batch, finer grid up to one max budget

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Pb314314 |
| 合并 | 2026-11-08 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | documentation, rocm, ready, llama, qwen, deepseek, nvidia, kimi, glm, minimax, cohere, DSv4.1 |
| Diff | pack diffstat 缺失(+/− 未知);按 diff 文本计数可见 9 个文件,另有 1 个文件因体积被丢弃,diff 末尾截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/65227 |

## 一句话结论

多模态 encoder(ViT)CUDA graph 改为"每批至多一次 replay + 细网格预算(全局上限默认 5120)":GB200 上 6 个真实权重模型 encoder 延迟 geomean 1.07–1.13×,graph 显存约降 3 倍,并修复 main 上 GLM-4.1V 单图崩溃。

## 动机

Encoder CUDA graph 只在 eager 受 host launch 时间限制时有收益。main 上 graph 捕获到 `max_num_batched_tokens`(默认 16384),运行时把一个 batch 贪心拆成多个 padded replay——encoder 在 16384 token 处多数已 GPU-bound,padding 与额外 replay 都是纯开销,大 batch 以 graph 跑反而常比 eager 慢。此外 #65084 的细预算让 GLM-4.1V 的向下取整 capture 尺寸不足行数,开启 encoder graph(默认)时一张 252×252 图像即以 `RuntimeError` 终止引擎(main 上已复现:256 vs 324 行)。

## 优化点

沿 LLM decoder CUDA graph 的风格定两条规则。其一,one replay per batch:`EncoderCudaGraphManager.execute()` 收到整批后,若 item 数 ≤ max_batch_size 且存在能容纳全部 token 的捕获预算,就 replay 最小的那个图;否则整批跑一次 eager forward,删除了原来的贪心分组、budget-cliff deferral 与多 replay 逻辑(路径模型每 path 决策一次);item 按 largest first 排序以利 varlen attention 调度。其二,预算网格细化并加全局上限:quarter-octave 到 1024、每 128 到 2560、再 1/8-octave(rounded 到 128),上限是新配置 `encoder_cudagraph_max_token_budget`(默认 5120,None 捕获到模型最大预算;`encoder_cudagraph_token_budgets` 用户覆盖仍优先生效),图数 25→36,得益于 #65084 的共享 capture buffer,细化网格不增加 buffer 显存——且最大预算从 16384 降到 5120,共享 buffer 反而缩小约 3 倍。5120 的选取有 24 模型扫描支撑:2560 会让 7/24 模型出现比 main 慢最多 +42% 的 eager batch,4096 剩 Gemma 4(+10%),5120 为 0/24。同时修复三处 eager 路径(DeepSeek-V4.1 从逐图 forward 改为单次 varlen 批、Ernie-4.5-VL resampler 跑两次且崩溃、DeepSeek-VL2 缺 `encoder_eager_forward`),并修 GLM-4.1V capture 尺寸向上取整。PR 自身在测试/文档外仅 −10 行。

## 关键改动

```text
vllm/v1/worker/encoder_cudagraph.py: _generate_budgets 加 cap;_execute_local 改单 replay 分派
vllm/config/compilation.py: 新增 encoder_cudagraph_max_token_budget(默认 5120,拒绝 ≤0)
vllm/models/deepseek_v41/common/vl_cudagraph.py: eager 改单次 varlen 批 forward
vllm/model_executor/models/glm4_1v.py: capture 每 item 输出尺寸向上取整(修崩溃)
vllm/model_executor/models/{deepseek_vl2,ernie45_vl}.py: eager fallback 修复
docs/design/cuda_graphs_multimodal.md: 设计文档同步(删贪心 packing 章节)
```

```python
# One replay per path: the smallest budget that holds the whole batch.
fits = num_items <= self.max_batch_size
batch_indices = sorted(range(num_items),
                       key=lambda i: per_item_out_tokens[i], reverse=True)
for path, tokens in per_item_path_tokens.items():
    if sum(tokens) == 0:
        continue
    token_budget = (self._path_fitting_budget(path, sum(tokens))
                    if fits else None)
    if token_budget is None:
        with torch.inference_mode():
            output = self.model.encoder_eager_forward(batch_mm_kwargs, path=path)
    else:
        graph_output = self._run_budget_graph(batch_mm_kwargs, token_budget,
                                              path=path, axis_keys=axis_keys)
```

## 性能声明

全部为 GB200、BF16、TP1、encoder-only,48 个 1–40 图随机尺寸 batch(token 45–16,128;InternVL3 34 个),每 cell 取 10 次计时中位数,顺序 Baseline/After/After/Baseline、各 batch 取两次较快者。vs 当前 main(含 #65084,`ebc97ad8d0`):InternVL3-8B 1.133×(>2560 token 批 1.172×)、Qwen2.5-VL-7B 1.113×/1.214×、MiniMax-M3 1.086×/1.119×、DeepSeek-V4.1-Flash 1.072×/1.137×、Kimi-K2.5 1.071×/1.099×(9/48 批慢 ≤6.6%,均为 ≤384 token 单图)、Qwen3.8-27B 1.067×/1.121×(5/48 ≤3.3%);GLM-4.1V 无法对比因 main 崩溃(本 PR 修复)。graph 显存:InternVL3 2.89→0.98 GiB、Qwen2.5-VL 3.59→1.34、MiniMax-M3 4.99→1.79、DeepSeek-V4.1 6.29→2.25、Kimi-K2.5 3.03→1.12、Qwen3.8 3.31→1.16。vs #65084 之前的 main:24 模型 geomean 1.03×–1.30×(LFM2-VL 1.266×、Qwen2.5-VL 1.300×、Kimi-K3 1.033×);已知例外:Gemma 4 恰好 32 图(8960 token)时 eager 比正好匹配的 graph 慢 9%(0.91×),作者接受此例并文档化(`encoder_cudagraph_max_token_budget` 可调高)。捕获时间多数 0–3 s 增加,除 GLM-4.1V(8.8→15.8 s)。

## 验证与回归风险

`tests/v1/cudagraph/test_encoder_cudagraph.py` 96 passed(main 上 16 failed:单 replay 分派、预算网格与 max_token_budget 测试);graph vs eager 输出对 7 个真实权重 + 17 个 dummy 权重模型逐一比对,无任何 batch 比 main 更偏离 eager;rebase 到 #65084 后 encoder 输出与 pre-rebase head bitwise 一致(0/3992 item outputs differ)。新增配置校验回归测试(非法 cap 必须在模型加载前报错,23 passed)。风险与限制:仅在 GB200 测量、无 AMD 数据;默认视频限制下单条长低分辨率视频会溢出 `cu_seqlens` 崩引擎(main 同样存在,#60112 修复且需小 rebase);GLM-4.1V 族开 video 时 image replay 输出错误(main 亦存在,#65453 修复);Gemma 4 32 图已知退化;图数 36 捕获略慢。

## 关联

前序:#65084(共享 capture buffer,已合并,本 PR 基于其上)。后续:#66067(stack 在本 PR 上,把 Qwen2/2.5/3-VL、GLM-4.1V、Cohere Compass 的 per-model encoder graph 代码收进共享 mixin,无行为变化);#65351(类似 buffer-sharing);#60112 与 #65453(修上文两处 main 既有缺陷,需 rebase)。pack 内无其他同分类 PR 信息。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
