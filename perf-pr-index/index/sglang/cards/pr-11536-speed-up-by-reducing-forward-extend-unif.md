# PR #11536: Speed up by reducing _forward_extend_unified's gpu->cpu sync bubble by ~2x faster

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @byjiang1996 |
| 合并 | 2025-10-13 |
| 分类 | Model-specific Fast Path(major) |
| Label | high priority |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 显示仅改动 triton_backend.py 的 _forward_extend_unified |
| 链接 | https://github.com/sgl-project/sglang/pull/11536 |

## 一句话结论

triton 后端 `_forward_extend_unified` 把逐请求 `.item()` GPU->CPU 同步改为一次性批量同步,该函数平均耗时 3370us→1634us(约 2x faster)。

## 动机

`_forward_extend_unified` 在拼接 prefix 与 extend 的 KV indices 时,对每个请求循环调用 `prefix_kv_indptr[i].item()`、`forward_batch.extend_start_loc[i].item()`、`forward_batch.extend_seq_lens[i].item()` 等——每次 `.item()` 都是一次 GPU->CPU 同步,bs 大时形成同步气泡,拖慢整个 extend 路径。PR 正文 Motivation 仅写 "As titled"。

## 优化点

把 CPU-GPU 同步批量前置:循环外一次性把 `prefix_kv_indptr[:bs+1]`、`forward_batch.extend_start_loc[:bs]`、`forward_batch.extend_seq_lens[:bs]` 三个张量各做一次 `.cpu()`,在 CPU 上向量化算出 `prefix_lens` 与 unified lens;`unified_kv_indptr` 在 GPU 上用 `torch.cat + cumsum` 重建。per-请求循环改为纯 CPU numpy 数组访问(`prefix_kv_indptr_np[i]` 等),循环内的张量切片用整数索引、不触发同步;循环中仅做 GPU 张量切片与 cat。整个循环不再有 per-iteration `.item()` 同步。

## 关键改动

```text
python/sglang/srt/layers/attention/triton_backend.py: _forward_extend_unified 重构——
  批量 .cpu() 同步一次,CPU numpy 上循环算索引,GPU 上重建 indptr,循环内零同步
```

关键摘录(批量同步前置):

```python
prefix_kv_indptr_cpu = prefix_kv_indptr[: bs + 1].cpu()
extend_start_loc_cpu = forward_batch.extend_start_loc[:bs].cpu()
extend_seq_lens_cpu = forward_batch.extend_seq_lens[:bs].cpu()
prefix_lens = (prefix_kv_indptr_cpu[1:] - prefix_kv_indptr_cpu[:-1]).to(torch.int32)
```

## 性能声明

TP4 Qwen3-8B,`bench_serving --dataset-name random --num-prompts 32 --random-input-len 256 --random-output-len 256 --random-range-ratio 1 --profile`:`_forward_extend_unified`(triton_backend.py)平均耗时从 3370us 降到 1634us(约 2x faster,数据来自 profile 截图)。PR 未注明所用 GPU 型号。精度(同一 benchmark 输出):Qwen3-8B Accuracy 0.945 / 输出吞吐 1513.949 tok/s;Qwen3-30B-A3B Accuracy 0.920 / 1222.439 tok/s(仅 After 值,无 Before 对照)。

## 验证与回归风险

给出 After 的 GSM8K 精度数值(Qwen3-8B 0.945、Qwen3-30B-A3B 0.920),但没有 Before 对照,无法据此断言精度不变;checklist 四项均未勾选,无新增测试。回归风险:逻辑重构涉及空 prefix/空 extend 的边界分支(两分支都可能产出空 list,已加 `if unified_kv_indices_list else torch.empty(...)` 兜底);依赖 `extend_start_loc`/`extend_seq_lens` 存在于 forward_batch(原实现有 hasattr 回退路径,新实现直接取用);影响所有走 triton 后端 unified extend attention 的模型。

## 关联

同作者同路径前序:#10680(causal_conv1d_fn kernel 消除 CPU<>GPU copy sync)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
