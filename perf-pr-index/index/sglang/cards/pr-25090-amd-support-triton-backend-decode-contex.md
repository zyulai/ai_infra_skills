# PR #25090: [AMD] Support triton backend decode context parallel for Qwen3.5

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @At1a8(co-author @IzacharyI) |
| 合并 | 2026-06-24 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, quant, amd, run-ci |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 超长,素材保留部分文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/25090 |

## 一句话结论

为 AMD HIP 上的 Triton attention 引入 decode context parallel(DCP):KV 按 `pos % dcp_size` 分片到多卡,1M 上下文 ITL 提升 45.31%、吞吐提升 22.82%。

## 动机

Qwen3.5 这类超长上下文(hybrid full/linear attention,yarn 到 1M)模型在纯 TP 下 decode 时 KV cache 与注意力计算都受单卡限制。DCP 把 KV 沿序列维切到 TP 组内多卡(每个 rank 只持有所属 token 的 KV),与 #14982 及 vllm issue #34018 的方向一致,可让长上下文 decode 随 DCP 规模扩展。此前 Triton backend 无此能力。

## 优化点

(1) 新增 `--decode-context-parallel-size`(`--dcp-size`),TP 组内按 dcp_size 切出 `_DCP` 通信组并纳入 CUDA graph capture;要求 dcp_size 整除 tp_size 且仅支持 AMD HIP。(2) KV 所有权规则 `owner(abs_pos) = abs_pos % dcp_size`:`MHATokenToKVPool.set_kv_buffer` 用 `masked_set_kv_buffer_kernel` 只写本 rank 拥有的 token,`HybridLinearKVPool` 转发 mask 以兼容 Qwen3.5 hybrid 路径,fused KV write 在有 mask 时被禁用。(3) Triton decode:rank 内 Q 沿 head 维 all-gather 得到全部 Q heads,各自用本 rank 的 KV shard(`create_triton_kv_indices_for_dcp_triton`)跑原 decode kernel 得部分输出与 split 维 LSE,`torch.logsumexp` 归并 split 维后由 `cp_lse_ag_out_rs()`(LSE all-gather → 输出 rescale → all-reduce → 本 rank head 切片)合成,输出形状与普通 TP decode 一致;`get_num_kv_splits` 用本地 shard 长度。(4) Triton extend kernel 增加 `STORE_LSE`/`SKIP_PREFIX`/`SKIP_EXTEND` 开关与 `extend_attention_fwd_with_lse()` 包装:current chunk 用本地 heads(skip_prefix),prefix shard 先 all-gather heads 再各算各的(skip_extend),prefix 部分经 `cp_lse_ag_out_rs(return_lse=True)` 跨 rank 合并,再与 current 用 `logaddexp` 合成,prefix 为空时跳过通信。(5) DCP extend 绕开 dense-attention Python fallback,避免 `[extend_tokens, heads, context_tokens]` 稠密分数张量导致的长上下文 OOM。

## 关键改动

```text
python/sglang/srt/distributed/parallel_state.py: _DCP 组构建/销毁 + graph capture 纳入
python/sglang/srt/layers/attention/triton_backend.py: DCP decode/extend 元数据、_kv_cache_loc/_set_kv_buffer、cp_lse_ag_out_rs
python/sglang/srt/layers/attention/triton_ops/extend_attention.py: LSE 输出 + SKIP_PREFIX/SKIP_EXTEND 开关
python/sglang/srt/mem_cache/memory_pool.py: masked_set_kv_buffer_kernel 只写本 rank owned token
python/sglang/srt/server_args.py 等: dcp_size 参数与校验
```

## 性能声明

吞吐/延迟(bench_serving,Qwen3.5-397B-A17B-FP8,tp8 + triton attention + yarn 1M context,DCP2 vs base,random dataset output 1k):吞吐在 input 1k-16k 时为 -12.36% ~ -3.30%,32k 起 +3.62%,64k +14.05%,128k +24.90%,256k +34.11%,512k +35.20%,1M +22.82%;ITL 在 1M 处 113.84 → 62.26(+45.31%),512k +42.14%,256k +36.32%;TTFT 基本持平或略负(1M 处 +3.60% 例外为正)。

精度:GSM8K 1319 题,FP8 模型 3 次、MXFP4 模型 5 次;MXFP4 汇总:main 0.9096±0.002,dcp 分支 dcp=1 基线 0.9096±0.006,dcp=2 0.9102±0.006。

## 验证与回归风险

非 DCP 路径全部走原代码分支(`_kv_cache_loc` 原样返回、`STORE_LSE=False`、原 `extend_attention_fwd`),风险集中在 DCP 使能路径:仅支持 HIP;dcp_size>1 非 HIP 平台直接 raise;DCP extend 显式拒绝 sinks、custom mask、sliding window;MLA 路径保持原逻辑("当前 DCP 改动主要面向非 MLA 模型");物理页放大分配与 invariant checker 的 slack 处理涉及 allocator 页大小匹配。CI Base 通过,Extra 为红。

## 关联

前序:sglang #14982、vllm issue #34018。同区域:同分类 Attention / Kernel 其他 PR;Qwen3.5 长上下文部署相关改动(rope yarn override)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
