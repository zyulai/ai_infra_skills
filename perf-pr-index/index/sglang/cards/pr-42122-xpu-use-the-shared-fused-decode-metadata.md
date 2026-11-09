# PR #42122: [XPU] Use the shared fused decode-metadata kernel in the intel_xpu ba…

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @xinchen9 |
| 合并 | 2026-10-14 |
| 分类 | Overlap / Scheduler | tier: major |
| Label | intel, xpu, run-ci |
| Diff | diffstat 缺失(pack 未提供 +N/-N);diff 含 2 个文件:`xpu_backend.py`、`test_normal_decode_set_metadata.py` |
| 链接 | https://github.com/sgl-project/sglang/pull/42122 |

## 一句话结论

把 XPU decode 元数据构建换成 FlashAttention 后端共享的 fused Triton kernel,元数据准备提速 3-5x,每步少 5-7 次 launch。

## 动机

PR 之前,`XPUAttentionBackend.init_forward_metadata_out_graph()` 每个 decode step 用一串独立 PyTorch ops 构建 page table:从 `req_to_token` strided gather、按 `page_size` floor-divide、cast 到 int32、copy 进预分配 buffer、再 zero 尾部列;`cache_seqlens_int32`/`cu_seqlens_k` 与 SWA page-table 还各有独立操作。NVIDIA FlashAttention backend 早已用一个 fused Triton kernel `normal_decode_set_metadata()`(位于 `python/sglang/kernels/ops/attention/metadata.py`)完成全部工作。该 kernel 是纯 Triton、无 CUDA 特有 intrinsics,可在 Triton 的 XPU backend 上原样运行,但 XPU backend 一直维护自己的 PyTorch 版本,每步付出 6-8 次 op 的 launch/host 开销。

## 优化点

核心机制是复用而非新写:新增私有方法 `_set_decode_page_metadata()`,用一次 fused Triton launch 同时写 `cache_seqlens_int32`、`cu_seqlens_k`、`page_table`,hybrid-SWA 模型还写 `swa_page_table`(通过 `full_to_swa_index_mapping`,等价于原 `SWAKVPool.translate_loc_from_full_to_swa()` 路径)。转发 `seq_len_delta`(= `kv_len_offset`)保留 target-verify / draft-decode 的 KV 扩展语义;传入紧界 `max_seq_pages = cdiv(max_seq_len_k, page_size)`。encoder-decoder 分支保留原 PyTorch 实现,seq-len cast 与 cumsum 移入该分支;SWA 的 gather/translate/copy 整块删除。行为差异:旧代码每步清零 page-table 尾列,新 kernel 只写每个请求的 live prefix(`cdiv(seq_len + delta, page_size)` 页),尾部保持旧值——这是 kernel 与 FlashAttention 后端共享的既有契约:attention kernel 以 `cache_seqlens_int32` 为读取边界,陈旧尾部条目始终是早期步骤写入的 in-range page id(或分配时的 0),chunked-attention 的 whole-table gather 不会越界。净改动约 -30/+40 行,一次 Triton launch 替代每步 6-8 个 PyTorch ops。

## 关键改动

```text
python/sglang/srt/layers/attention/xpu_backend.py: 新增 _set_decode_page_metadata(),在 init_forward_metadata_out_graph() 非 encoder-decoder 分支改调 fused kernel
test/registered/attention/test_normal_decode_set_metadata.py: 内核单测扩展到 XPU 并注册 XPU CI (stage-b-test-1-gpu-xpu)
benchmark/kernels/bench_xpu_decode_metadata.py: 新增;旧 PyTorch 块 vs fused kernel 的正确性+延迟对比,不需 repo 外依赖
```

```python
normal_decode_set_metadata(
    cache_seqlens_int32=metadata.cache_seqlens_int32,
    cu_seqlens_k=metadata.cu_seqlens_k,
    page_table=metadata.page_table,
    req_to_token=self.req_to_token,
    req_pool_indices=req_pool_indices,
    max_seq_pages=max_seq_pages,
    seq_lens=seq_lens,
    seq_len_delta=seq_len_delta,
    page_size=self.page_size,
    swa_page_table=metadata.swa_page_table if use_swa else None,
)
```

## 性能声明

- 元数据准备 standalone(host-inclusive latency,`bench_xpu_decode_metadata.py --markdown`,Intel Arc Pro B70):bs1/ctx8192/page64:97.4 us → 31.5 us(3.1x-3.9x);bs4:103.8 → 31.3 us;bs16:104.0 → 32.5 us;bs32/ctx1024:103.5 → 32.3 us;bs32/ctx8192:104.8 → 32.5 us,整体 3.1x-4.2x。两侧延迟对 batch/context 基本平坦,瓶颈是 launch overhead 而非带宽。
- 端到端 serving(`bench_serving` 300 prompts,`--backend sglang --num-prompts 300 --max-concurrency {1,32}`,Qwen2.5-1.5B-Instruct BF16,`--attention-backend intel_xpu --page-size 64 --cuda-graph-backend-decode full --mem-fraction-static 0.8`,每格单次运行):并发 1 中性——median ITL 6.33→6.31 ms(-0.3%)、median TPOT 6.32→6.30 ms(-0.3%)、output throughput 155.1→155.4 tok/s(+0.2%)。并发 32:output throughput 2494→2724 tok/s(+9.2%)、median TPOT 10.88→9.95 ms(-8.5%),但作者标注 baseline 该次含约 1 s stall(max ITL 886 ms vs 133 ms),这两格是单次观察、非主张增益;median ITL 8.27→8.29 ms(0)。

## 验证与回归风险

测试:`test_normal_decode_set_metadata.py` 在 XPU 上 16 passed + 8 subtests;bench 脚本 36 组配置(bs 1-32,ctx 1K-8K,page 1/16/64)下 `cache_seqlens`/`cu_seqlens_k`/live `page_table` 条目与旧 PyTorch 块精确一致;GSM8K 5-shot 200 题、temperature 0、Qwen2.5-1.5B-Instruct BF16、`--page-size 64 --cuda-graph-backend-decode full`:accuracy 0.625,0 invalid;server log 确认 decode graph 捕获(3.4 s)且每个 decode batch 走图,即 fused 路径每步被行使。风险:XPU 上 decode graph 是 opt-in(`--cuda-graph-backend-decode full`),eager 路径 `init_forward_metadata()` 不经过本改动;page-table 尾列不再清零依赖"读取以 `cache_seqlens_int32` 为界"的下游契约,若有新消费方做无界全表读取会受影响;encoder-decoder 分支仍走旧路径;`intel_xpu` 后端仅接受 `--page-size` 64/128,其他值被强制为 128。

## 关联

无正文列出的前序/后续 PR。共享 kernel 来自 FlashAttention backend 的 `normal_decode_set_metadata()`(NVIDIA 路径已存在);CUDA/ROCm 同名单测此前已注册,本 PR 补 XPU。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
