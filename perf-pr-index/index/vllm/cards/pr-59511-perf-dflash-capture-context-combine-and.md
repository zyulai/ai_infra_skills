# PR #59511: [Perf][DFlash] Capture context combine and anchor in the draft CUDA graph

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Juntian777 |
| 合并 | 2026-10-14 |
| 分类 | Speculative Decoding (tier: major) |
| Label | speculative-decoding, ready, nvidia, mrv2, dflash |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 本身可用,含 3 个文件(dflash speculator、lilicorr speculator、测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/59511 |

## 一句话结论

把 DFlash/DSpark draft 的上下文准备(aux hidden 合并、LiLiCorr anchor、context K/V 预计算)纳入 FULL CUDA graph,decode 每 step 省 0.28-0.35 ms CPU,batch-1 ITL 提升约 2-6%。

## 动机

DFlash/DSpark/LiLiCorr 共享的上下文准备分三步:合并 target 的 auxiliary hidden states、发布 LiLiCorr context anchor、预计算 context K/V。前作 #57632 只把最后一步收进了 draft 的 FULL CUDA graph,前两步仍在每次 replay 前以 eager 方式执行,batch-1 decode 为 host-bound 时这段 CPU 开销直接进入 ITL。

## 优化点

把完整的 context prologue 录进 draft FULL graph 中 query forward 之前:`_prepare_graph_forward` 依次执行 staged aux hidden 的 `combine_hidden_states`、`prepare_context_anchor` 与 `_precompute_context_kv`。为使 graph 可 replay,做了配套改造:target aux states 在 replay 前由 `_store_hidden_states` 拷入 capture 时分配的固定地址 `_aux_staging` buffer(仅在 draft 捕获 FULL graph 时分配,内存 profiling 可计入,eager fallback 不占);`LiLiCorrSpeculator` 自持 `_num_rejected` 的固定地址副本供 anchor 读取(DFlash/DFlash2/DSpark 传 None 跳过拷贝);`prepare_context_anchor` 签名从 `(input_batch, num_rejected)` 改为 `(num_reqs, query_start_loc, num_rejected)`,保证在 graph 内运行时不与 host 同步。超出 graph 容量的 context 行与非 FULL 路径仍走 eager(`_prepare_graph_replay` 中 `_precompute_context_kv(num_context, num_target_tokens)`);capture 前的 eager warmup 保留,含 #58814 的 Kimi-K3 cache-pointer 刷新。

## 关键改动

```text
vllm/v1/worker/gpu/spec_decode/dflash/speculator.py: _aux_staging 固定地址 buffer;_prepare_graph_forward/_prepare_graph_replay/_store_hidden_states;prologue 进 graph
vllm/v1/worker/gpu/spec_decode/lilicorr/speculator.py: _num_rejected 固定地址副本;prepare_context_anchor 改签名以支持 graph 内运行
tests/v1/spec_decode/test_lilicorr.py: 新增 capture/replay 回读固定输入的 CUDA graph 测试
```

graph 内 context step(节选):

```python
def _prepare_graph_forward(self, num_reqs, attn_metadata):
    num_context = self._num_graph_context_tokens(num_reqs)
    if self._aux_staging is not None:
        self.hidden_states[:num_context].copy_(
            self.model.combine_hidden_states(self._aux_staging[:num_context])
        )
    self.prepare_context_anchor(
        num_reqs, self.target_input_buffers.query_start_loc, self._num_rejected
    )
    self._precompute_context_kv(0, num_context)
```

## 性能声明

E2E decode ITL(TP4 + DSpark,4x GB200 单节点,DeepSeek-V4.1-Flash,FLASHINFER_MLA_SPARSE_DSV41,fp8 KV cache,关闭 prefix caching 与 FlashInfer autotune,5 spec tokens + synthetic acceptance 6.0,p50 of 3 repeats):128 in/1024 out、c=1 时 5.66/5.91 ms(base 两次)对 5.55 ms(PR),即 -2%/-6%(base 自身波动约 5%,故对两次基线分别引用);8192 in 时同样 -2%/-6%;c=8 与 c=64 无可测差异(GPU-bound)。Microbenchmark(单 GB200,context prologue 单独测):decode 每步省 0.28-0.35 ms CPU(如 req=1 CPU 392.6 对 48.8 µs),GPU 时间基本不变;prefill 因投影拆分多付约 37 µs。GSM8K(DEP4 MegaMoE adaptive verification):base 92.57% 对 PR 92.34%/92.49%,acceptance length 匹配(3.365 对 3.367/3.330)。

## 验证与回归风险

7 个 spec_decode 测试文件共 92 passed;新增 `test_captured_context_anchor_rereads_fixed_inputs` 用 `torch.cuda.CUDAGraph` 验证 replay 会重读 anchor 的固定地址输入、忽略 padded 请求的 stale rejection count、并对超出 graph 容量而 eager 存储的 prefill 行正确 anchor。风险面:仅影响 DFlash/DFlash2/DSpark/LiLiCorr 共享路径的 FULL graph 模式;`_aux_staging` 增加常驻显存(大小为 max graph context rows x aux 层数 x hidden size,捕获时分配);非 FULL 模式与超容量 prefill 回退 eager,行为保留;anchor 语义依赖 `num_rejected` 在 replay 前拷贝到位。

## 关联

前序 #57632(首次捕获最后一步 context K/V)、#58814(Kimi-K3 cache-pointer 刷新)、#50658(Kimi-K3 aux projection 前移到 SP all-gather 之前);正文注明检索未发现其他捕获该 prologue 的 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
