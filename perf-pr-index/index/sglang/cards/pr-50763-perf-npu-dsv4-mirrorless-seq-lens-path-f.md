# PR #50763: [Perf] [NPU] dsv4: mirrorless seq_lens path for DFlash-family spec decode

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @iridiumine |
| 合并 | 2026-11-18 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, npu, run-ci, run-ci-npu-extra |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 摘录含 8 个文件,含 1 个测试文件修改) |
| 链接 | https://github.com/sgl-project/sglang/pull/50763 |

## 一句话结论

为 DeepSeek-V4 NPU attention backend 在 DSpark/dFlash verify 下增加无 CPU 镜像的 seq_lens 路径(`SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0`):全部 attention 元数据在设备端从 live `seq_lens` 推导,消除每步 pageable D2H(单步最多 ~52ms 的 StreamSynchronize);median ITL 降约 7.6%、每步墙钟降约 9%。

## 动机

DFlash-family 投机解码(DSpark/dFlash verify)在 Ascend NPU 上,attention backend 需要 `seq_lens` 的 CPU 镜像(`SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=1`);scheduler 每步发布该镜像,任何回退到设备侧 `seq_lens.cpu()` 的消费者都要付出 pageable D2H——排空整个 CANN async task queue,每 verify 步最多约 52ms 的 StreamSynchronize 而设备空闲。CUDA 的 DSPARK backend 已无条件 mirrorless,本 PR 是该思路的移植。

## 优化点

- `ascend_dsv4_backend.py`:graph replay 在设备端推导 live/final KV lengths 与 verify compress 元数据(where-gated `live + n_draft`;DP padding 的 idle/ghost 行经 where-gating 保持全零);kernel 元数据 `*_cpu` 参数按 `needs_cpu_seq_lens` 门控,使 warmup 与 replay 的 ABI 一致;eager verify 在设备端把 live prefix 展开为 final KV lengths,不再信任可能过期的 `fm.seq_lens_cpu_int`;replay-prep 宽度用 `max_seq_len_host` hint(+1 page slack 应对 overlap host-view lag),无 hint 时全宽回退;decode compress-loc 用 stable compact 替代布尔索引(动态 shape 输出会迫使每步 D2H sync)。
- `dsv4_common_hooks.py`:mirrorless verify 在设备端构建 DSV4 cache-loc bundle——c128 draft-interval locs 经 gather 与 stable compact(cumsum + scatter,无 NonZero)进入按消费者 padding bound 定宽的固定 buffer,消费者 `numel()` 不再 sync;以有效 `live_seq_lens_cpu` 为 None 触发。新增 `_flatten_interval_device` 作为 host `flatten_interval` 的设备孪生。
- `eagle_utils.py` / `dflash_info_v2.py` / `dflash_info.py` / `dspark_draft.py`:在 `eagle_prepare_for_verify`、`prepare_for_decode`、`prepare_for_verify` 等 DFlash prep 路径设置 `batch.max_seq_len_host`(host 侧 final KV 长度上界,纯 host),并带进手工构造的 draft `ForwardBatch`(绕过 init_new);`=1` 时消费者 gate 在 `seq_lens_cpu` 上,不读该 hint。
- `=1` 路径除两处数值等价替换(布尔行索引 `x[mask]` → mask-multiply / stable compact;`aclnnNonzeroV2` 在 NPU 上无法启动且隐含 D2H)外保持不变。

## 关键改动

```text
python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py:
-        assert seq_lens_cpu is not None, "V4 graph replay requires seq_lens_cpu."
+        # Mirrorless replay: no host mirror; live/final derivations stay on
+        # device and the ctx *_cpu fields may hold device tensors.
+        mirrorless = seq_lens_cpu is None
python/sglang/srt/hardware_backend/npu/dsv4/dsv4_common_hooks.py(新函数):
+def _flatten_interval_device(*, table, page_size, req_pool_indices, live_seq_lens, verify_lens, n_draft, upper):
+    """Device twin of the host ``flatten_interval`` loop (ratio=128): the c128
+    slots of the compressed positions ``[live // 128, (live + verify) // 128)`` ..."""
python/sglang/srt/speculative/eagle_utils.py:
+        batch.max_seq_len_host = (
+            max((req.seqlen for req in batch.reqs), default=0)
+            + verify_input.draft_token_num
+        )
test/registered/unit/npu/attention/test_npu_ascend_dsv4_backend.py: 新增 TestMirrorlessReplayWidthHint(宽度 hint 的封顶/全宽/不越界/SWA 切片)。
```

## 性能声明

(数字全部照抄 PR 正文,未独立核实。基准:ShareGPT gsp,64 prompts,concurrency 32,system prompt 57.6k tokens,seed 1,16P,各 3 个 steady runs,首个 warm-up run 剔除。)

| Metric | =1 | =0 |
|---|---|---|
| Output throughput (tok/s) | 481 / 482 / 518 | 545 / 504 / 519 |
| Accept length | 5.55 / 5.58 / 5.35 | 5.36 / 5.06 / 5.54 |
| Median ITL (ms) | 12.01 / 11.98 / 11.89 | 11.06 / 11.01 / 11.08 |
| Per-step wall time (ms) | 11.5 / 11.6 / 10.3 | 9.8 / 10.0 / 10.7 |

正文结论:"Accept length is content-level noise (it swings 5.0–5.6 across identical configs), so ITL and per-step wall time are the stable comparators: **mirrorless cuts median ITL by ~7.6%, output throughput by ~6%, and average per-step wall time by ~9%**." Profiling(元数据流,每 verify 步,steady decode):`aclnnIndex_StridedSlice` 0.52 → 0.52 ms;`aclnnIndex_Index` 0.20 → 0.46 ms;metadata stream busy 3.4 → 4.15 ms;ProfilerStep 83.1 → 81.3 ms。"With `with_stack` profiling, the per-step `Runtime@StreamSynchronize` (pageable D2H, up to 51.7ms) is fully eliminated; device kernel busy stays ~98% across the steady window"。精度:GSM8K 200 prompts,DeepSeek-V4-Flash + DSPARK,16P:`=1` 98.5%,`=0` 99.5%(正文附完整 env 与启动命令)。

## 验证与回归风险

新增单测 `TestMirrorlessReplayWidthHint`:hint 300 → compress 宽度封顶 4 页(512 tokens)、无 hint 用捕获全宽、hint 1e9 不越捕获宽度、SWA 刷新按 hint+1 页切片(100 → 2/8 页)与无 hint 全宽。CI 状态:Base 与 Extra 通过,AMD ROCm 10 失败(NPU 特定改动,AMD 平台不适用)。回归风险:(1) mirrorless 下各 prep 路径都必须设置 `max_seq_len_host`,漏设则全宽拷贝(正确但更慢);(2) `_flatten_interval_device` 的固定宽度与 trash-slot stable compact 依赖 `upper` 边界与 `_loc_pad_value` 哨兵正确;(3) `update_verify_buffers_to_fill_after_draft` 在 mirrorless 下跳过 stale fill(`fm.seq_lens_cpu_int` 由后续 refresh 重填),与 `=1` 行为不同;(4) `=1` 路径的两处布尔索引替换虽数值等价,但实现路径不同(aclnnNonzeroV2 不可用是 NPU 特有约束);(5) 无 hint 时 compress_seq_lens_max 用捕获 buffer 全宽,页表拷贝代价上升。

## 关联

#42224(本批次,NPU DSA indexer INT8,同为 Ascend 路径);CUDA DSPARK backend 已无条件 mirrorless(正文动机);`SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS` 文档(environment_variables.mdx)更新,说明 DSV4 backend 在无 kernel 开关时也遵循该开关。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
