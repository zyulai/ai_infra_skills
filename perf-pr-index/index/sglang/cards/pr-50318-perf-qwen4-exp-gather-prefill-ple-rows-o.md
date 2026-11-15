# PR #50318: [Perf] qwen4-exp: gather prefill PLE rows on the host for the file-backed table

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @N3mezis |
| 合并 | 2026-11-14 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/50318 |

## 一句话结论

file-backed PLE n-gram 表在经 HMM 逐页 fault 的离散 GPU(RTX 50 系)上,prefill 尺寸的 gather 改为 host 侧 `index_select` 进 pinned memory + 一次 H2D + 原 Triton kernel:首次 gather 9.3–13.2 s 降到 0.039–0.060 s(约 20x),~1.5k-token prompt 首 sight TTFT 17.10 s→7.42 s(−57%)。

## 动机

`--ple-offload-backend file`(#37068)下 Qwen3.8-Flash-Next 把 47.7 GiB PLE n-gram 表放进 file-backed mmap 由 gather kernel 直接读;#48689 起包括经 HMM 读 pageable 内存的离散 GPU(如开放内核模块上的 RTX 50 系),配合 paged experts(#42557、#48642、#49195)可在 16 GB 卡上服务完整模型。这类 GPU 上,每个其页未被 GPU 映射的表行都是驱动逐个处理的 page fault,约 0.5 ms/页;prefill 每 token gather 16 行、大多来自不同页,~1.5k-token prompt 即 17–21k 个 distinct rows,首次 gather kernel 花 9–13 s(页已映射后 0.4 s,RTX 5070 Ti)。file backend 已有的 WILLNEED hint 把页带进 page cache,但 GPU 仍要逐页 fault 进自己的映射——这占了 TTFT 的大头。经 host page tables 读 pageable 内存的 GPU(ATS,如 GB10 / DGX Spark)不会逐个 fault,故保持原路径。

## 优化点

- `Qwen4ExpPinnedHostEmbedding.gather`:满足 prefill 尺寸(≥ `PLE_FILE_PREFETCH_MIN_ROWS` = 2048 ids,即 prefetch hint 已用的阈值)、来自 file-backed 表、且 GPU 会 fault pageable 页时,在 host 上读出 distinct rows(mmap 表 `index_select` 进 pinned memory),一次传输拷到 GPU,再对 staged rows 跑同一个 Triton kernel(ids 重映射进 staged rows;TP shard 外的 id 仍保持 shard 外,得零行)。kernel 对 staged rows 的转换与对表内行完全一致,输出字节不变。
- host 拷贝前先对 staged rows 的页 `posix_fadvise(WILLNEED)`:经一次性打开的只读 fd,相邻页合并为一次调用(`advise_rows_willneed`)。server 以单线程跑 torch、映射为 `MADV_RANDOM`,否则拷贝会把每个冷行当作独立的 4 KiB 读逐一进行;advised 之后 reads 一起 in-flight。
- 保持 kernel 原路径:decode 尺寸 gather、CUDA graph capture 期间、pinned backend、经 host page tables 读或无法查询的 GPU。
- `qwen4_exp_ple_table.py`:新增 `device_faults_pageable_pages()` 查询 `cudaDevAttrPageableMemoryAccess`(=1)与 `cudaDevAttrPageableMemoryAccessUsesHostPageTables`(=0),与既有 `device_reads_pageable_memory()` 共享属性查询;建层时读一次。host 路径不依赖 `SGLANG_QWEN4_PLE_FILE_PREFETCH`。

## 关键改动

```text
python/sglang/srt/models/qwen4_exp.py: Qwen4ExpPinnedHostEmbedding 增加 _stages_on_host 门控与 _gathers_on_host/_stage_rows;gather 分派 host staging 或原 kernel 路径
python/sglang/srt/models/qwen4_exp_ple_table.py: device_faults_pageable_pages()(两条 CUDA 属性)与 advise_rows_willneed(posix_fadvise WILLNEED,相邻页合并)
test/registered/kernels/ops/embeddings/test_qwen4_ple_offload.py(新增 GPU 用例): host 路径字节一致 + host-page-table GPU 控制组
test/registered/unit/models/test_qwen4_exp_ple_table.py(新增 CPU 用例): 属性判定(HMM/GB10/不可查)与 WILLNEED 合并/失败/真实文件
```

```python
ids = flat_ids.cpu()  # a prefill chunk's ids: one sync, as the page hint takes
in_range = (ids >= vocab_start) & (ids < vocab_end)
rows, inverse = torch.unique(ids[in_range] - vocab_start, return_inverse=True)
...
advise_rows_willneed(self._table_fd, rows, table.shape[1])
staged = torch.empty(
    (rows.numel(), table.shape[1]), dtype=torch.uint8, pin_memory=True
)
torch.index_select(table, 0, rows, out=staged)
```

## 性能声明

(Qwen3.8-Flash-Next-FP8,一块 RTX 5070 Ti 16 GB、62 GB RAM、PCIe 5 NVMe,paged experts disk tier 按文档设置,file-backed PLE 表同一 NVMe;main = 6a72cc2918;每次启动前清 page cache;main / PR / main again,4 prompts 取中位数)

- TTFT:~1.5k-token prompts 首次 17.10 s → 7.42 s(main again 17.17 s),同 prompts 重访(rows cached)7.82→7.18 s;68-token 短 prompt(低于阈值)2.78→2.94 s;decode(320 tokens)14.93→14.79 tok/s。首 sight TTFT 降 57%(4 prompts 范围:main 16.2–20.6 s,PR 7.0–9.4 s)。decode 单独复核:8 短 prompts x 256 tokens,main/PR/main again 14.50/14.40/14.51 tok/s,0.1 tok/s 差距在任何长 prompt 之前就存在;两 main run 逐 prompt 相差最多 0.35 tok/s。
- gather 单独计时(前后 synchronize;~1.5k prompts,16.7k–21.4k distinct rows):kernel 直读表 first sight 9.3–13.2 s / 重访 0.38–0.47 s;host 拷贝无 WILLNEED 0.83–1.22 s / 0.002 s;host 拷贝带 WILLNEED(本 PR)first sight 0.039–0.060 s / 重访 0.010–0.012 s——带 hint 时首次 gather 约快 20x,缓存行上 hint 每 prompt 花约 9 ms(上表 TTFT 用的是早版 `madvise` hint,每 prompt 约多 10 ms;最终版 TTFT 差异在 0.03 s 内)。
- 全模型 GSM8K(RTX 5070 Ti,Qwen3.8-Flash-Next-FP8 + paged experts,200 题,thinking off)0.98,2/200 答案截断;无本 PR 的 disk tier 同设置 0.975。(GSM8K 在 WILLNEED hint 加入前测;hint 只调度读取,staged bytes 由上述测试覆盖。)
- GB10 上未测试;门控在该处保持当前路径。

## 验证与回归风险

- 新 GPU 测试(`test_qwen4_ple_offload.py`,bf16 与 FP8 表、TP 2 分片):4096-id prefill gather 经 host 路径与源行字节一致(覆盖重复 ids、两个 shard 边缘、shard 外 ids);无 id 落在本 rank shard 时得零行;decode 尺寸 gather 留在 kernel;GPU 控制组:经 host page tables 读的 GPU 上 prefill 尺寸 gather 留在 kernel、字节相同。
- 新 CPU 测试(fake CUDA runtime 应答属性查询):HMM 会 fault、GB10 不会、无 pageable access 不会、runtime 不可查询不会;WILLNEED:相邻页合并、跨两页的行 advise 两页、无行不调用、失败即停、kernel 接受真实表文件上的 advice;GPU host-path 测试检查每次 staged gather 恰好 advise 其行。
- RTX 5070 Ti(SM120, HMM)上全部 PLE 测试通过:`test_qwen4_ple_offload.py` 与 `test_qwen4_exp_ple_table.py` 加 config-read ratchet 共 48 tests、0 failures;CPU-only 跑 26 table tests(3 个 GPU-only skip)。
- 负面控制:无视 device 检查时恰有 GPU 控制测试与 GB10 case 失败;移除 WILLNEED 调用与相邻页合并时恰有 GPU host-path 测试(两种 dtype)与合并测试失败。风险:host 路径多一次 `flat_ids.cpu()` 同步与每步 pinned 分配(由 staging 复用),decode/CUDA graph capture 路径不变。

## 关联

#37068(file offload backend)、#48689(HMM 离散 GPU 支持引入)、#42557 / #48642 / #49195(paged experts)、#49771(FlashInfer 0.7.4 门槛,作者据此外推 main 基线 6a72cc2918)、`SGLANG_QWEN4_PLE_FILE_PREFETCH`(host 路径不依赖它)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
