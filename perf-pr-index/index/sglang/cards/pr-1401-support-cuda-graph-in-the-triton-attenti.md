# PR #1401: Support cuda graph in the triton attention backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @merrymercy |
| 合并 | 2024-09-12 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | (none) |
| Diff | pack 未提供 diffstat(diff 完整可见,改 7 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/1401 |

## 一句话结论

让 triton attention 后端支持 cuda graph:预分配 attn_logits 静态 buffer,bs=1 decode 提速 Llama-3-8B 1.3x、DeepSeek-Coder-V2-Lite 4x。

## 动机

此前 cuda graph 只对 flashinfer attention 后端启用(model_runner 中显式拦截其他后端),triton 后端无法用图捕获。阻塞点在于 triton decode 路径的中间 logits buffer `att_m` 是每次 forward 按 `total_num_tokens` 动态 `torch.empty` 分配的,形状依赖运行时数据,无法作为静态 buffer 进图。本 PR 把 attention 后端的 cuda graph 能力抽象成统一接口,并让 triton 后端实现之。

## 优化点

三层改动。其一,`AttentionBackend` ABC 新增统一 cuda graph API:`init_cuda_graph_state`、`init_forward_metadata_capture_cuda_graph`、`init_forward_metadata_replay_cuda_graph`、`get_cuda_graph_seq_len_fill_value`,并把 `forward` 拆为 `forward_decode`/`forward_extend`;flashinfer 后端的 `capture_cuda_graph_init`/`replay_cuda_graph_init` 更名对齐。其二,TritonAttnBackend 实现该接口:`init_cuda_graph_state` 预分配 `cuda_graph_start_loc` 与 `(num_head, max_bs * context_len)` 的 `cuda_graph_attn_logits` 静态 buffer;capture 时 metadata 直接指向静态 buffer 且 `max_seq_len` 取 context_len;replay 时用 `cumsum` 原地重算 start_loc。其三,`decode_attention_fwd` 签名从传入 `total_num_tokens` 内部分配 `att_m` 改为外部传入预分配的 `attn_logits` buffer(消除图捕获不友好的动态分配与 `.item()` 同步)。配套:`cuda_graph_runner` 的 `seq_lens` 初始化为后端提供的 fill value(非 0 时避免除零/非法索引),replay padding 时按 fill value 填充;移除"仅 flashinfer 支持 cuda graph"的限制。

## 关键改动

```text
python/sglang/srt/layers/attention_backend.py: ABC 增加 cuda graph API 与 decode/extend 拆分;triton 后端实现静态 buffer 方案
python/sglang/srt/layers/triton_attention/decode_attention.py: decode_attention_fwd 改收预分配 attn_logits buffer
python/sglang/srt/model_executor/cuda_graph_runner.py: seq_lens fill value 机制接入后端接口
python/sglang/srt/model_executor/model_runner.py: 移除非 flashinfer 后端禁用 cuda graph 的拦截
test/srt/test_serving_throughput.py: 新增 triton 后端吞吐回归测试
```

```python
def init_cuda_graph_state(self, max_bs: int):
    self.cuda_graph_max_total_num_tokens = max_bs * self.cuda_graph_max_seq_len
    self.cuda_graph_attn_logits = torch.empty(
        (self.num_head, self.cuda_graph_max_total_num_tokens),
        dtype=self.REDUCE_TORCH_TYPE, device="cuda",
    )

def get_cuda_graph_seq_len_fill_value(self):
    return 1
```

## 性能声明

测试条件:`sglang.bench_latency`,batch-size 1,input 128,output 8(pack 未注明 GPU 型号)。

Llama 3 8B(PR 标注 1.3x faster,`--attention-backend triton`):

| 配置 | decode median latency | throughput |
|---|---|---|
| triton w/ cuda graph | 0.00706 s | 141.63 token/s |
| triton w/o cuda graph | 0.00928 s | 107.79 token/s |
| flashinfer w/ cuda graph | 0.00735 s | 135.98 token/s |
| flashinfer w/o cuda graph | 0.00823 s | 121.46 token/s |

DeepSeek-Coder-V2-Lite(PR 标注 4x faster,`--enable-mla --trust-remote`):triton w/ cuda graph 0.00622 s / 160.82 token/s,vs w/o cuda graph 0.02453 s / 40.77 token/s。

## 验证与回归风险

新增 `test_srt/test_serving_throughput.py::test_default_with_triton_attention_backend`(CI 中断言 output_throughput > 2400)。回归面:所有 attention 后端的 forward 接口签名(ABC 强制实现 decode/extend 拆分与新接口);flashinfer 后端方法更名;triton decode kernel 的 logits buffer 生命周期从函数内分配改为调用方持有(`# TODO: reuse the buffer across layers` 注明尚未跨层复用);`attn_logits` 静态 buffer 大小为 max_bs x context_len x num_head,显存占用随 context_len 线性增长。

## 关联

同区域:flashinfer 后端已有的 cuda graph 支持(本 PR 将其模式推广为后端接口);pack 正文未列出关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
