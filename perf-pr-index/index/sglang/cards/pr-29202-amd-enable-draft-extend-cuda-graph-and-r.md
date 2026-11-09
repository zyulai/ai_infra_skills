# PR #29202: [AMD] Enable draft-extend CUDA graph and reduce bubble for MTP

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @RolaoDenthu |
| 合并 | 2026-08-13 |
| 分类 | Speculative Decoding (tier: major) |
| Label | amd, deepseek, run-ci, bypass-fastfail |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 含 3 个文件的修改 |
| 链接 | https://github.com/sgl-project/sglang/pull/29202 |

## 一句话结论

为 HIP/ROCm 上 DeepSeek-V4 MTP(EAGLE 投机解码)打通 draft_extend 阶段的 CUDA graph 捕获并消除一处隐式 D2H 同步,TTT 提升 5.7%–11.7%、ITL 提升 9.7%–13.5%。

## 动机

针对 AMD HIP/ROCm 路径上 DeepSeek-V4 MTP 的三个独立问题:(1) `draft_extend` 阶段因 HIP 的 draft-extend CUDA-graph 门控不识别 DSV4 后端而回落到 eager 执行,本可被 graph 捕获的阶段没被捕获;(2) 对 `_attach_unified_kv_prefill_meta` 做修改后,带 `--profile` 跑 server 会在垃圾回收时可靠地崩溃,完全阻塞了 MTP 路径的 profiling 与验证;(3) unified-kv prefill metadata 构建中的 `repeat_interleave` 触发隐式 device→host 同步,拖慢整个 `draft_extend` metadata 构建。

## 优化点

三处机制性修复:一是扩展 `eagle_worker_v2.py` 的 graph 门控,`supports_hip_draft_extend_graph` 除 `AiterMultiStepDraftBackend` 外新增接受 `DeepseekV4HipRadixBackend`(DSV4 把 draft-extend 支持暴露在 draft-extend 后端本身,而非 aiter 的 decode multi-step 后端),使 draft_extend 在 CUDA graph 下运行;二是 `profiler_manager.py` 在受控的 stop 点先置空再 `gc.collect()`,避免 torch.profiler pybind11 对象的引用环在循环 GC 中延迟回收时 C++ 析构 double-free;三是 `deepseek_v4_backend_hip_radix.py` 的 `_attach_unified_kv_prefill_meta` 给 `repeat_interleave` 传入 `output_size=num_tokens`,免去把 `extend_seq_lens.sum()` 读回 host 的隐式同步(调用方已知该值);target-extend 路径仍保留原行为(去掉它会触发 GPU 内存访问故障)。

## 关键改动

```text
python/sglang/srt/speculative/eagle_worker_v2.py: HIP graph 门控新增 DeepseekV4HipRadixBackend,启用 draft-extend CUDA graph
python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py: _attach_unified_kv_prefill_meta 新增 num_tokens/need_compress 参数,repeat_interleave 传 output_size 避免隐式 D2H
python/sglang/srt/managers/scheduler_components/profiler_manager.py: _stop_profile 中置空 profiler 后立即 gc.collect(),修复 double-free 崩溃
```

```python
supports_hip_draft_extend_graph = isinstance(
    self.draft_attn_backend, AiterMultiStepDraftBackend
) or isinstance(self.draft_extend_attn_backend, DeepseekV4HipRadixBackend)
```

## 性能声明

- **TTT(time to token)提升 5.7%–11.7%,ITL(inter-token latency)提升 9.7%–13.5%**(作者正文;前后 profiler 截图佐证,截图内具体数字未在文本中给出)。
- 精度:gsm8k acc **0.95**。
- PR 未注明具体模型/硬件配置细节(仅可推断为 DeepSeek-V4 MTP + HIP/ROCm),也未给出绝对 tokens/s。

## 验证与回归风险

作者给出 gsm8k 精度结果(0.95)。回归风险:`repeat_interleave` 的 `output_size` 只在 `need_compress=False` 分支传入,注释明确 target-extend 路径去掉 sum 读取会触发 GPU 内存访问故障,两分支行为需与上游 unified-kv kernel 保持一致;profiler 的 gc.collect() 影响所有使用 profiler 的路径(不仅 AMD);CUDA graph 门控扩展仅 HIP 路径生效,导入保持局部以免非 HIP 环境依赖。CI 状态:Base run 通过,Extra run 显示失败(:x:)。

## 关联

同区域:DeepSeek MTP/EAGLE 投机解码系列、AMD HIP attention 后端(aiter)优化;pack 未列出具体关联 PR 编号,关联暂记"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
