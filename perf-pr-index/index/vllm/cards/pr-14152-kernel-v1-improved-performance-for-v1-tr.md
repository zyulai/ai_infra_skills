# PR #14152: [Kernel] [V1] Improved performance for V1 Triton (ROCm) backend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tdoublep |
| 合并 | 2025-03-06 |
| 分类 | Attention / Kernel（tier: major） |
| Label | rocm, ready, v1 |
| Diff | diffstat 缺失（素材记录为 +None/-None）；diff 含 4 个文件（约 +880/-120 行） |
| 链接 | https://github.com/vllm-project/vllm/pull/14152 |

## 一句话结论

V1 Triton 注意力后端把 decode 序列从 `prefix_prefill` kernel 中拆出，改用专用 paged decode kernel，serving 吞吐提升 3.65 倍。

## 动机

V1 `ROCmAttentionBackend`（后随 #14071 改名 `TritonAttentionBackend`）此前对所有 prefill、chunked prefill 与 decode 统一使用单个 `prefix_prefill` Triton kernel。作者发现该 kernel 对 decode 阶段（query_len=1）的序列优化很差：H100 上 Llama-3.1-8B-Instruct serving 基准中总吞吐比 FlashAttention 基线差 5 倍以上。把所有场景塞进一个 Triton kernel 意味着 grid、tile 尺寸与 auto-tuning 必须对所有情形折衷，限制很大。

## 优化点

核心机制是 prefill 与 decode 分道：(1) 新增 `chunked_prefill_paged_decode` 入口，batch 中 query_len > 1 的序列仍走 `prefix_prefill`（新增 `skip_decode=True` 参数，kernel 内 `SKIP_DECODE` 编译期分支使 decode 序列直接 return），其余序列走新写的 `kernel_paged_attention_2d` 优化 paged decode Triton kernel；(2) 新 decode kernel 以 `(num_seqs, num_query_heads)` 为 grid，逐 KV block 做 online softmax（running max M / running sum L / alpha 重缩放），支持 fp8 KV cache 反量化、alibi slopes 与 sliding window；(3) 无需改动 V1 attention metadata。PR 初版依赖外部 `ibm_triton_lib`，后更新为自包含实现。

## 关键改动

```text
vllm/attention/ops/chunked_prefill_paged_decode.py: 新增，prefill/decode 分派入口 + kernel_paged_attention_2d 专用 decode kernel
vllm/attention/ops/prefix_prefill.py: 增加 SKIP_DECODE 编译期参数，decode 序列在 prefill kernel 中提前退出
vllm/v1/attention/backends/rocm_attn.py: forward 改调 chunked_prefill_paged_decode
tests/kernels/test_prefix_prefill.py: 全部用例参数化到两个 kernel，并强制 batch 含一条 decode 序列
```

decode kernel 的分派过滤逻辑：

```python
if filter_by_query_len:
    ...
    cur_batch_query_len = cur_batch_in_all_stop_index \
        - cur_batch_in_all_start_index
    if cur_batch_query_len > 1:
        return
```

## 性能声明

测试条件注明：`meta-llama/Llama-3.1-8B-Instruct`、NVIDIA H100 GPU（为让 ROCm 后端在 NVIDIA 上运行做了 hack）、`benchmark_serving.py` sharegpt 数据集 1000 请求。结果：Triton 后端 main 总吞吐 3842.98 tok/s → 本卡优化后 14037.37 tok/s（**3.65x** 吞吐提升），与 FlashAttention 基线（20187.58 tok/s）相比仍差约 30%。Mean TPOT 408.87 ms → 100.91 ms，Mean ITL 210.67 ms → 53.15 ms，Mean TTFT 与 FlashAttention 基本持平（3711 vs 3473 ms）。正确性：GSM8K 5-shot（limit 500，H100）Triton 后端 0.798/0.780（flexible/strict）vs FlashAttention 0.792/0.770。作者注明该 Triton kernel 尚未 auto-tune，还有提升空间；AMD GPU 性能当时评估中。

## 验证与回归风险

`tests/kernels/test_prefix_prefill.py` 中 context attention、alibi、f32 三组用例均参数化到新旧两个 kernel 对比验证（atol 对非 fp8 从 1e-6 放宽到 1e-4），并新增"batch 中必有一条 decode 序列"的覆盖。风险面：`prefix_prefill` 新增 `SKIP_DECODE` 参数改变共享 kernel 签名，其他直接调用 `context_attention_fwd` 的路径需确认兼容；decode kernel 支持 fp8（e4m3/e5m2）KV cache，但精度容差放宽；此路径原生面向 ROCm/Triton 后端，本次基准在 NVIDIA H100 上完成。

## 关联

前向关联：#14071（后端改名 TritonAttentionBackend）；#13305（kernel auto-tuning 方向）；RFC issue #13319（外部 Triton kernel 项目讨论）。同后端后续优化可参考 rocm/Triton attention 相关 PR（素材未列出更多编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
