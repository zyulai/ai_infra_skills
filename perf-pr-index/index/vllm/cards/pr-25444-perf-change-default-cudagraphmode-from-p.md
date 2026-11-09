# PR #25444: [Perf] Change default CUDAGraphMode from PIECEWISE to FULL_AND_PIECEWISE

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2025-09-23 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | performance, ready, v1 |
| Diff | 素材未提供 diffstat 数字;pack diff 收录 config 与 gpu_model_runner 三个文件改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/25444 |

## 一句话结论

把 vLLM V1 默认 cudagraph 模式从 PIECEWISE 切到 FULL_AND_PIECEWISE(decode 捕获全图),Qwen3-Coder-30B-A3B-FP8 单 H100 吞吐 4.07→5.98 req/s(+47%)。

## 动机

超出 piecewise 的 full cudagraph 支持在数月内逐步落地(尤其 #20059),此前默认仍保守地用 PIECEWISE(把 attention 等不兼容 op 留在图外)。虽然 full cudagraph 会增加启动时间(需要捕获更多图),但对小模型或 MoE 的低延迟收益明确,作者判断值得作为默认。

## 优化点

核心是一行默认值切换:V1 + PIECEWISE compilation level 下 `cudagraph_mode` 默认从 `PIECEWISE` 改为 `FULL_AND_PIECEWISE` —— decode batch 捕获 full cudagraph(消除 decode 的 CPU launch 开销),prefill 与混合 batch 仍走 piecewise。配套两个保护:pooling 模型(有 pooler_config)回退 PIECEWISE,因为其不支持 full cudagraph;`initialize_cudagraph_capture` 新增检查,当 decode full-graph 与 attention backend 的 `AttentionCGSupport.NEVER` 冲突时自动降级为 PIECEWISE(attention 编译为 piecewise 时)或 NONE,并输出 warning。另外 `_dummy_run` 在 capture 时用 `max_query_len` 而非 `max_model_len` 填 seq_lens。

## 关键改动

```text
vllm/config/__init__.py: V1 默认 cudagraph_mode 改 FULL_AND_PIECEWISE,pooling 模型回退 PIECEWISE
vllm/config/compilation.py: 文档同步,FULL_AND_PIECEWISE 标注为默认且"对多数模型最高效"
vllm/v1/worker/gpu_model_runner.py: attention backend 不支持 decode full graph 时自动降级;dummy run seq_lens 改用 max_query_len
```

关键片段(config/__init__.py):

```python
if envs.VLLM_USE_V1 and self.compilation_config.level \
    == CompilationLevel.PIECEWISE:
    # default to full and piecewise for most models
    self.compilation_config.cudagraph_mode = \
        CUDAGraphMode.FULL_AND_PIECEWISE
```

## 性能声明

Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8,1×H100,10 条请求(1024 in / 128 out),`vllm bench serve --num-prompts 10`:
- default(PIECEWISE):4.07 req/s;--async-scheduling:4.29 req/s。
- -O.cudagraph_mode=FULL_AND_PIECEWISE(本 PR 新默认):5.98 req/s;加 --async-scheduling:6.20 req/s。
- FULL 模式:5.83 / 6.03 req/s(低于 FULL_AND_PIECEWISE)。
- 启动时间代价:`python -c "from vllm import LLM;LLM('Qwen/Qwen3-0.6B')"` 总耗时 21.684s → 25.005s(约 +3.3s)。

## 验证与回归风险

验证为上述吞吐与启动时间对比,diff 中未新增单元测试。风险面:这是所有 V1 用户的默认行为变化;依赖各 attention backend 的 full-graph 支持度,不满足时靠新增的自动降级逻辑兜底(PIECEWISE 或 NONE + warning);pooling 模型显式回退;`_dummy_run` 的 seq_lens 语义变化影响 capture 时形状假设。

## 关联

前序:#20059(full cudagraph 与独立 sampling)。相关:#21146(gc.freeze 缓解 full 模式更多 capture 带来的启动开销)、#22763(Cutlass full CUDA graph)、#24123(WideEP piecewise CUDA graph)、#54985(跨 record 复用 CUDA graphs)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
