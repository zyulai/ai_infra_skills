# PR #65132: [ROCm][Perf] Build AITER MLA full-graph decode metadata in one kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zengleixin-amd |
| 合并 | 2026-11-13 |
| 分类 | Speculative Decoding(tier: major) |
| Label | rocm, ready |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/65132 |

## 一句话结论

Full CUDA graphs 下 `AiterMLAMetadataBuilder._build_decode` 原用 10–25 个小 eager op 填 persistent decode metadata buffers,本 PR 改为单个 Triton kernel `_build_decode_indptr_kernel`:uniform MTP GPU 时间 −33%、adaptive verification −64%,端到端 decode step 在并发 1–8 快 0.2–0.56%,metadata 与 eager 构建 bit-identical。

## 动机

Full CUDA graphs 下,`_build_decode` 每步用 10–25 个小 eager op 填 persistent decode metadata buffers(`graph_seq_lens`、`paged_kv_indptr`、`qo_indptr`,adaptive verification 还有 varlen q/o row maps),launch 与 host 开销在低并发和 adaptive verification 下占比明显。

## 优化点

- 新增 `_build_decode_indptr_kernel`(单 program,大小 `next_power_of_2(max_num_reqs + 1)`)一次性写出:每请求 kernel KV 长度及其零前缀 cumsum、query indptr(uniform 时按 pad)、两段 indptr 的尾部(至 buffer 长度,重复最后活值),varlen 时还填 q/o row maps。
- 融合路径门控:full CUDA graphs、`self.dcp_world_size == 1` 且无 DCP total lengths;其余(含 DCP)保持既有 eager 路径;超过 graph buffer 的 batch 由 assert 拒绝(eager op 本来也拒绝)。
- `num_reqs` 通过 `do_not_specialize` 排除出 Triton specialization,serving 中新 batch size 不触发 JIT 编译。
- 输出与 eager 构建 bit-identical(唯一差异是 varlen o-row map 中从不被读的 dump slot);不引入新的 host/device 同步。
- `_expand_page_indices_kernel` 改从 persistent buffer 读 `kv_indptr`(值与原 temporary 相同)。

## 关键改动

```text
vllm/v1/attention/backends/mla/rocm_aiter_mla.py: 新增 _build_decode_indptr_kernel;_build_decode 增 fused_indptr 门控与单次 launch;varlen/paged_kv_indptr/qo_indptr 分支相应改写
tests/v1/attention/test_aiter_mla_decode_indptr_kernel.py(新增): kernel 直接对照 eager 构建(任意 GPU,无需 AITER)
tests/v1/attention/test_rocm_aiter_mla_mtp_split.py: 新增 full-graph builder fused vs eager、DCP 不launch、oversize 拒绝等测试
```

```python
# Full graphs read the kernel lengths and both indptrs from persistent
# buffers. Fill all three in one launch rather than ~20 eager ops.
fused_indptr = (
    self._graph_seq_lens is not None
    and self.dcp_world_size == 1
    and g_tot_seq_lens is None
    and num_kernel_reqs > 0
)
if fused_indptr:
    assert num_kernel_reqs <= self._graph_seq_lens.shape[0], (
        f"{num_kernel_reqs} decode requests exceed the "
        f"{self._graph_seq_lens.shape[0]}-request graph buffers"
    )
    varlen_rows = (
        (self._varlen_q_rows, self._varlen_o_rows)
        if varlen_decode
        else (self.qo_indptr, self.qo_indptr)
    )
    _build_decode_indptr_kernel[(1,)](
```

## 性能声明

- Kimi-K3 official recipe(TP8,DSpark MTP 7 tokens,fp8 KV,`FULL_AND_PIECEWISE`),MI355X,每 decode metadata build:uniform MTP(recipe 默认)kernels 11 → 4,GPU 60.2 → 40.6 µs(−33%),host p50 211 → 158 µs(−25%);adaptive verification(varlen)kernels 24.5 → 4.7,GPU 134.7 → 48.4 µs(−64%),host 342 → 163 µs(−52%)。
- Kernel 开销 vs `max_num_seqs`(MI355X 单卡,每请求 8 query tokens,torch profiler 每 build kernel 时长和):uniform 在 max_num_seqs 8/64/256/512 为 21–27 → 2.0 µs,1024 为 24–27 → 2.3–2.8 µs;adaptive 在 8/64 为 56–64 → 2.0–2.1 µs,256 为 63–67 → 3.6–3.7 µs,512 为 64–71 → 5.0–5.2 µs,1024 为 66–78 → 8.1–8.5 µs(batch 1/half/full)。单 program 尺寸随 `max_num_seqs` 而非活跃 batch 增长,但处处比 eager op 便宜 8–30x;host launch 约 17 µs fused vs 77–213 µs eager,与 `max_num_seqs` 无关。
- 端到端:在每个 `max_num_seqs`(至 1024)与每个 batch size 上都比 eager op 快;decode step period(ABBA 交错,每 leg 重启 server;用 step period 替代 TPOT,因 DSpark acceptance length 跨 run 波动 ±8%,在该效应量级下淹没 TPOT 与 throughput):recipe(`--max-num-seqs 8`)C1 −0.20%、C4 −0.44%、C8 −0.56%(paired t −1.4/−2.7/−2.3);`--max-num-seqs 64`(graph sizes to 512)C16 −0.12%、C32 +0.14%(n.s.)、C64 0.00%。约 20–100 µs/步的节省在高并发下占比缩小。
- Serving 正确性:GSM8K 5-shot 全 1319 题 greedy:uniform 94.92% → 95.38%,adaptive 94.92% → 95.83%;DCP=8 / DCP=8+adaptive(前 400 题)98.5% / 98.25%(如预期未走 fused 路径)。NIAH(RULER `niah_single_2`,每长度前 100 题):32k/64k/128k 双侧均 100%。GPQA Diamond(198 题双侧两 run)pooled 91.2% vs 88.9%(−2.3 点,95% bootstrap [−5.1, +0.3];正文论证该差异属 serving stack 的 run 级波动而非本 PR——1,394,688 次 fused build 的 shadow 对照 0 mismatch)。

## 验证与回归风险

- 单测(8 GPU,main vs 本 PR):`tests/v1/attention/...` 套件 main 963 passed / 1 failed(`test_rocm_aiter_mla_prefill_fusions.py` 一例,main 上间歇性失败、与本 PR 无关),本 PR 989 passed / 0 failed。
- 新 builder 测试:真实 `_build_decode` 有/无 fused 路径对照 metadata 与 graph buffers(含 tails),8/64/1024-request buffer 在 batch 1、half、full;断言 DCP batch 永不 launch fused kernel、oversize batch 被拒绝;MTP 宽度 1、3、4、7 覆盖。`test_rocm_aiter_mla_mtp_split.py` 单独 107 passed;因该模块在非 ROCm 跳过,`test_aiter_mla_decode_indptr_kernel.py` 在任意 GPU(无需 AITER)直接测 kernel,MI355X 33 passed(NVIDIA V1 Attention step 也会跑)。
- Shadow check(三种分支各一个 server):fused 构建后重算 eager 参考并全对比——GPQA run 1,394,688 次 0 mismatch;adaptive verification server 331,776 次(221,184 varlen + 110,592 plain)0 mismatch;无投机解码 server 49,152 次 0 mismatch。
- 风险:单 program 尺寸由 `max_num_seqs` 决定,开销随 buffer 而非活跃 batch 增长;每 build 剩余最大开销是 AITER 自己的 `get_mla_metadata_v1`(约 28 µs),本 PR 未动;更早版本把 `paged_kv_indices.fill_(-1)` 全量替换为 tail reset 的部分被砍掉以保持 PR 小(该 `fill_` 每 build 花费 7.8 µs);GPQA 数字本身不构成等价性证据(正文以 bit-identical metadata 为准)。

## 关联

- #65022(split metadata lifetime)与 #64560(decode q absorb/quant fusion)动同一 backend 但不涉及此 metadata build;#61504 / #58352 针对 sparse MLA backend;#62990(无 DCP 时 varlen rows 恒左对齐,是 fused varlen 分支的前提)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
