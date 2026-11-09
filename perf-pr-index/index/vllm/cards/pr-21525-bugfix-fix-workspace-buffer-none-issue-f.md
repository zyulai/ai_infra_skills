# PR #21525: [Bugfix] Fix workspace buffer None issue for Flashinfer TRTLLM Backend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @elvischenv |
| 合并 | 2025-07-29 |
| 分类 | Attention / Kernel(major) |
| Label | performance, ready, v1 |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,涉及 4 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/21525 |

## 一句话结论

修复 FlashInfer TRTLLM decode 首轮 build 时 workspace_buffer 为 None 的问题,并改为直接写预分配输出、去掉重复的 workspace 属性。

## 动机

PR #19825 在构造 `FlashInferMetadata` 时传入 `workspace_buffer=self._workspace_buffer`,但该字段在第一轮 `self._plan()` 之前是 None,只有第二轮 `build()` 才拿到正确值——即 TRTLLM decode 后端启动后的第一次运行会拿到空 workspace。PR #21137 曾用 `self._get_workspace_buffer()` 缓解,但 `decode_wrapper._float_workspace_buffer` 本身始终有效,`attn_metadata.workspace_buffer` 是完全重复的属性,应直接移除。

## 优化点

三层改动:(1) 从 `FlashInferMetadata` 删除 `workspace_buffer` 字段及其在 `build()` 中的赋值,forward 时直接取 `decode_wrapper._float_workspace_buffer`;(2) 调用 `trtllm_batch_decode_with_kv_cache` / `decode_wrapper.run` 时传入 `out=`(v0 路径直接写 `output[:num_decode_tokens]`,v1 路径预分配 `decode_output`),替代旧的"返回值再赋给 output 切片"写法,消除一次输出拷贝,并把非 TRTLLM 分支同样改为 `out=` 形式;(3) `benchmarks/kernels/benchmark_trtllm_attention.py` 与单测同步适配 FlashInfer 新 API(#21485 引入的破坏性变更:头数/page_size/kv_cache_dtype 等位置参数改为 `bmm1_scale`/`bmm2_scale`/`out` 关键字)。

## 关键改动

```text
vllm/v1/attention/backends/flashinfer.py: 删除重复 workspace_buffer 字段;out= 直写输出
vllm/attention/backends/flashinfer.py: v0 路径同样直取 _float_workspace_buffer 并 out= 直写
tests/kernels/attention/test_flashinfer_trtllm_decode_attention.py: 适配新 FlashInfer API
benchmarks/kernels/benchmark_trtllm_attention.py: 适配新 FlashInfer API
```

关键片段(v1 forward):

```python
workspace_buffer = decode_wrapper._float_workspace_buffer
trtllm_batch_decode_with_kv_cache(
    query=decode_query, kv_cache=kv_cache_permute,
    workspace_buffer=workspace_buffer, ..., out=output[:num_decode_tokens])
```

## 性能声明

PR 正文未给出性能对比数字(perf claims 检测为 none);收益是修复首轮 decode 的 None workspace 崩溃/错误路径并省一次输出拷贝。正确性验证:单测 `tests/kernels/attention/test_flashinfer_trtllm_decode_attention.py` 96 passed(121.20s);lm-eval gsm8k 5-shot(limit 500、batch 200、temperature=0)在 `nvidia/Llama-4-Scout-17B-16E-Instruct-FP8`(modelopt 量化、TP1、max_model_len 2048)上 exact_match flexible-extract 0.932±0.0113、strict-match 0.914±0.0126。

## 验证与回归风险

依赖上述单测与 lm-eval。风险面:TRTLLM decode 的 workspace 来源从 metadata 持有改为运行时从 wrapper 读取,若 wrapper 生命周期与 metadata 不同步(plan 重建)需要保证引用有效;`out=` 语义要求传入 buffer 形状/dtype 精确匹配;FlashInfer API 关键字参数(k_scale→bmm1_scale 组合)在 flashinfer 版本变化时需再适配。

## 关联

前序:PR #19825(引入问题)、PR #21137(临时修复)、PR #21485(API 破坏性变更,本 PR 修单测)。同区域为 FlashInfer/TRTLLM attention 后端;PR 未注明其他后续编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
