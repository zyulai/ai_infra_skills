# PR #32219: [MTP] Cut spec-v2 host-seam overhead in hybrid-linear MTP decode

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-07-28 |
| 分类 | Speculative Decoding (tier: major) |
| Label | speculative-decoding, run-ci, jit-kernel, run-ci-extra, linear-attention, release-highlight, user-tps |
| Diff | diffstat 缺失(pack 未提供 +/− 行数与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/32219 |

## 一句话结论

消除 hybrid-linear MTP decode 的四处 host seam(阻塞同步/逐步分配/多余 dispatch),低并发下 GPU 空泡近零。

## 动机

spec-v2 overlap 调度下,每个 MTP decode 步由三个 CUDA graph 组成(draft → target_verify → draft_extend),graph 之间是 eager 的 host "接缝":accept sampling、cache-slot 分配、attention metadata 重建、下一步 draft 输入准备。低并发(bs=1 到小 batch)时这些接缝、host 侧 Python dispatch 与逐步同步无法被 GPU overlap,直接变成 GPU idle。`event_loop_overlap` 的设计意图是 host 跑在 GPU 前面一步(run-ahead),但四个具体的 per-step host 成本把它拉回 lockstep:KDA 后端未声明 `needs_cpu_seq_lens=False` 导致每步阻塞式 seq_lens D2H 流同步;hybrid wrapper 丢失 verify-buffer hooks 导致每步重新分配 `bs * max_context_len` 的 custom_mask;verify prep 里 eager 的 end-offset 相加;replay-prep 的 5-7 次独立 dispatch 链。作者强调 host seam 是串行链——单点修复不叠加,必须成 bundle 才能把 host 藏到 GPU 后面。

## 优化点

四个独立可 revert 的修复。其一,`KDAAttnBackend` 声明 `needs_cpu_seq_lens = False`(与 GDN/Mamba2/ShortConv 对齐):KDA 的共享 metadata 从不读 spec-v2 的 CPU seq_lens 镜像(replay padding 来自 `forward_batch.num_padding`),于是每步 `fwd_prepare_d2h_stream.synchronize()` 消失——实测 bs=1 MTP trace 中该 stall 中位数约 0.5 ms/步。其二,`HybridLinearAttnBackend` 转发 `get/update_verify_buffers_to_fill_after_draft`:draft 阶段直接写进 full-attn 子后端的已捕获 verify buffer,逐步 mask 分配消失;plan-stream 下 tree verify(topk>1)在 stream join 后 re-copy tree links,无实现的子后端保持 fail-loud。其三,新增 `assign_extend_cache_locs_uniform` Triton kernel:end offset 在 kernel 内算(`start + draft_token_num`),输出偏移直接 `pid * draft_token_num`,免去 host 的 `seq_lens + N` 张量与 O(bs) 跨行前缀和 load;NPU/CPU 回退参考路径。其四,`_fused_replay_state_indices_kernel` 把 mamba replay-prep 的 gather + sentinel + store 链(含零化 padded 行的 side effect)融成一次 launch;门槛为 CUDA + static hybrid pool(identity v2p)+ replayssm off。全部为调度/融合改动,无数值路径变化。

## 关键改动

```text
python/sglang/srt/layers/attention/linear/kda_backend.py:
  needs_cpu_seq_lens = False(去掉每步阻塞 D2H 同步)
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:
  verify-buffer hooks 转发;_replay_metadata 走 fused kernel fast path
python/sglang/kernels/ops/speculative/cache_locs.py:
  assign_extend_cache_locs_uniform(kernel 内算 end offset)
python/sglang/kernels/ops/mamba/mamba_state_indices_triton.py(新):
  _fused_replay_state_indices_kernel(单 launch,bit-identical)
```

```python
# replay-prep:5-7 次 aten dispatch 融合为一次 triton launch
req = tl.load(req_pool_indices_ptr + offs, mask=valid, other=0)
idx = tl.load(mamba_map_ptr + req, mask=valid, other=0)
out_val = tl.where(valid, idx.to(tl.int32), -1)
tl.store(out_ptr + offs, out_val, mask=in_range)
tl.store(req_pool_indices_ptr + offs, zeros, mask=in_range & (~valid))  # 保 side effect
```

## 性能声明

PR 无 bundle 级 TPOT A/B 数字(注明 Blackwell 上的 bundle-level TPOT A/B 属后续系列)。单项声明:item 1 去除实测约 0.5 ms/步中位阻塞流同步(bs=1 MTP trace,KDA-hybrid);item 2 去除每步 `bs * max_context_len` bool-mask 分配;items 3-4 把 6-8 次 per-step host dispatch 换成 2 次 kernel launch。背景:本 PR 是 hybrid-linear MTP 优化系列的一部分,该系列使 MTP 的 GPU 利用率从 67% 提到 95%;来源 campaign(KDA-hybrid MoE,4×B200,TP4,NEXTN)终态约 90% GPU busy、host 完全被 GPU 遮蔽。trace 前后对比图显示 draft_extend/draft/target_verify 之间的 GPU bubble 接近消失、verify graph launch 可被 overlap 进下一个 draft_extend。

## 验证与回归风险

新增两个测试:verify-buffer fixup hook 单测(CPU 纯逻辑,refresh/chain no-op/eager no-op/dummy no-op);`test_fused_replay_state_indices.py` 在 guard-padded buffer、bs ∈ {1,2,7,32,33} × num_padding sweep × 3 seeds + 非 injective mapping case 上断言 fused 与参考 aten 链 bit-identical(含 -1 sentinel、padded 行零化 side effect、guard tail 不被越界写);H20-3e 上 2 tests 45 subtests 全过。e2e:H20-3e tp=4 上 Qwen3.5-35B-A3B + NEXTN MTP(GDN-hybrid spec-v2)与 Kimi-Linear-48B-A3B-Instruct 均 8 prompts × 256 tokens 贪婪输出 byte-identical,`spec_verify_ct` 逐 prompt 一致(接受行为不变)。风险面:item 1 依赖"KDA metadata 不读 CPU seq_lens 镜像"这一前提,后续改 metadata 代码时需保持;fused kernel 的门控(identity v2p、replayssm off)之外路径仍走参考链;plan-stream 的 tree-link re-copy 只覆盖 EagleVerifyInput。

## 关联

作者注明为 hybrid-linear MTP 优化系列之一(bundle-level TPOT A/B 在后续系列报告),pack 未给出系列其他 PR 编号;同区域相关:本批次 #14792(mamba radix cache + overlap scheduler)、#29792(Mamba overlap bookkeeping 修复)、#26312(MTP 拒绝采样)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
