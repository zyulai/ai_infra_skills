# PR #47685: [Perf] [NVIDIA] dflash: Top up decode KV pages for the whole batch on allocation steps

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @weireweire |
| 合并 | 2026-11-10 |
| 分类 | Speculative Decoding（tier: major） |
| Label | 无（pack 记录 labels: (none)） |
| Diff | +89/-24, 4 files（pack diffstat 缺失，按 pack diff 文本计数） |
| 链接 | https://github.com/sgl-project/sglang/pull/47685 |

## 一句话结论

DFlash decode 在分配步给全 batch 补足 K=2 页 KV headroom，合并零散页跨越分配，DEP32 吞吐 +4.1%、step p50 −0.2 ms。

## 动机

`DFlashDraftInputV2.prepare_for_decode` 只为 committed 长度加 2-block reserve 越过已分配长度的请求分配 KV 页；page size 256、accept ~3.5 时每请求约 73 个 decode 步才跨一次页。DEP32 c5768 的 nsys profile（~36 running/rank）显示一个 rank 在约 35% 的步上发生分配，每个分配步约 0.78 ms host 时间（`get_last_loc` 95 µs、paged `alloc_extend` 578 µs、`req_to_token` assign 56 µs）。attention DP 下这项工作位于 DP agreement 之前：32 个 rank 独立分配，几乎每步都有某个 rank 在付钱，整步等它。

## 优化点

新增 `SGLANG_OPT_SPEC_DECODE_HEADROOM_PAGES`（默认 2，其他值钳到 2..8，0 关闭），逻辑在 `allocation_sizing.add_spec_decode_headroom_pages`，仅 DFlash-family decode 生效：在本来就必须分配的步，把 headroom 不足 K 页的每个请求一次性补到页对齐需求之外 K 页（同一 `alloc_for_spec_decode` 调用），上限受 `req_to_token` 行宽约束。此后每请求约再跑 K 页 decode 才需要新页，各 rank 的分配被批进共享分配步。空闲空间守卫：free 须覆盖分页 extend 的驱逐需求（needed + top-up + bs×page_size）再加 3 倍 top-up，否则本步按原样精确分配——top-up 不会触发本不该发生的 prefix-cache 驱逐；HiSparse 下跳过（其 spec-decode 路径总是驱逐）。代价是 admission 略保守：decode admission 的 per-request 预留与已补页最多重复计入 K 页/请求（PD decode worker 上最坏 98k tokens @192 running），守卫使 top-up 上限为所需之外 free 的 1/4，内存趋紧时自动退化为精确分配。

## 关键改动

```text
python/sglang/srt/mem_cache/allocation_sizing.py: add_spec_decode_headroom_pages（补页 + free 守卫）
python/sglang/srt/speculative/dflash_info_v2.py: prepare_for_decode 接入补页调用
python/sglang/srt/environ.py: SGLANG_OPT_SPEC_DECODE_HEADROOM_PAGES（默认 2）
test/registered/spec/dflash/test_dflash.py: page-256 + K=4 overlap 用例
```

```python
target = min(row_cap, (need + page_size - 1) // page_size * page_size + pages * page_size)
if target > lens[i]:
    extra += target - lens[i]
    lens[i] = target
...
free = batch.token_to_kv_pool_allocator.available_size()
if free < num_needed_tokens + len(lens) * page_size + 4 * extra:
    return nxt_kv_lens, num_needed_tokens  # 内存紧张：本步回退精确分配
```

## 性能声明

GB300、DSV4.1-Flash、K=2、2026-10-30 nightly。DEP32 c5768（10P1D，两臂均含 #47676–#47678，单变量为 top-up）：decode step p50 13.9→13.7 ms；稳态窗口 per-GPU 吞吐 660.6k→687.8k tok/s/GPU（+4.1%，作者注明来自单对运行、各有一次 dip bin、run-to-run 漂移可达 ~3%，吞吐差单独不构成结论，step 时间 −0.2 ms 才是信号）；p90 interactivity 219.8→223.4 tok/s/user。9P1D DEP16 bs192 c7480（四项 host 改动合并对比，非 top-up 单变量）：step p50 每档低 0.2–0.26 ms（192 running：33.64→33.38 ms），吞吐持平 1127.2k tok/s/GPU，KV headroom 无回退（full KV max 0.55→0.54，0 retraction）。GSM8K（1P1D DEP16，真实 DSpark acceptance）96.82% vs 96.82%。

## 验证与回归风险

诊断 overlay 一致性检查 870 次 0 失败（新分配 [cur,nxt) 区间非零、不与其他请求共享、kv_allocated_len=nxt）；现有 DFlash/decode-prep CPU 测试在 flag 关闭与 =2 下通过；`test_dflash.py` page-256（默认 K=2）与新增 K=4 overlap 用例在 GB300 通过（GSM8K 0.770/0.770/0.765；main page-256 为 0.775）。风险：行为默认开启（页大小>1 的 DFlash-family decode），改变分配与 admission 行为；NPU DSV4 C128 sidecar pool 不在守卫内且该路径未在 NPU 运行；仅测过 K=2；top-up 在低 free 时自动跳过但守卫边界依赖 3x top-up 余量假设。

## 关联

与 #47676、#47677、#47678（DSV4.1 decode host 路径改动）同区域、独立开发且测量时堆叠；同分类的 DSpark 验证调度见 #44588。无其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
