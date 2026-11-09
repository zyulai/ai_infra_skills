# PR #59779: [Bugfix][Watermarking] Keep draft prompt lengths valid under CUDA graphs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @simon-veitner-redhat |
| 合并 | 2026-10-02 |
| 分类 | Speculative Decoding（tier: major） |
| Label | bug, ready, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/59779 |

## 一句话结论

修复 CUDA graph 下 draft 水印 dedup mask 读到陈旧 prompt_len，graph 与 eager 恢复一致，吞吐变化 <1%。

## 动机

`dual_key_gumbel` 配默认 `deduplicate_contexts="single_turn"` 时，FULL CUDA graphs（默认 `FULL_AND_PIECEWISE`）下 draft sampling 从陈旧 UVA buffer 读 prompt lengths：新请求到达后 `prompt_len.gpu` 已切到 round-robin 池下一块，而捕获的 dedup kernel 仍读旧地址，导致部分 draft 缺失 key A、部分重复 context 未去重，graph 与 eager 分叉。PR 称该 bug 随 #56807 引入、复核 #59554 时发现（DFlash 亦会暴露）。

## 优化点

把 `DraftWatermarker` 的 `prompt_lens` 从持有 UVA 张量引用改为持久 device buffer：构造时 `torch.zeros(max_num_reqs, dtype=torch.int32, device=device)`，`prepare()` 用 `copy_(prompt_lens)` 每次 propose 做一次异步拷贝。被捕获的 dedup kernel 改读 device memory 而非 pinned host memory（正文称微基准更快），且无 host sync；该 kernel 每个 draft step 跑一次，K=3 时每次 propose 三次，在 CUDA graph 重放下因改读 device memory 反而更快。影响 EAGLE3、MTP、DSpark（已复现），EAGLE 共用 autoregressive 路径。

## 关键改动

```text
vllm/v1/watermarking/spec_decode.py: prompt_lens 改为持久 device buffer 并 copy_ 更新
tests/watermarking/test_watermarking.py: 新增 reads_current_prompt_lens 回归用例
```

```python
+        self.prompt_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
         self.all_token_ids: torch.Tensor | None = None
-        self.prompt_lens: torch.Tensor | None = None
         self.total_lens: torch.Tensor | None = None
...
         self.all_token_ids = all_token_ids
-        self.prompt_lens = prompt_lens
+        self.prompt_lens.copy_(prompt_lens)
```

## 性能声明

正文 Performance（B300、Qwen3-4B + EAGLE3（AngelSlim/Qwen3-4B_eagle3, K=3）、dual_key_gumbel、temperature 1.0、random 1024 in / 256 out 带 `--ignore-eos`、`FULL_AND_PIECEWISE`、双 GPU 并行 10 轮×2 passes）：输出吞吐 c=1 464.7→462.3 tok/s（−0.5%±0.6）、c=8 3032→3012（−0.6%±0.8）、c=64 10543→10495（−0.4%±0.8）；Δ TPOT +0.3%~+0.5%、Δ TTFT +0.8%~+2.3%。吞吐/TPOT 变化 <1% 且 CI 含零，作者视 TTFT 差异为噪声。微基准：`prepare()` host 48.4–48.9→51.4–52.1 µs（+3 µs）；graph 重放 dedup kernel 12.3→8.7 µs（1 req）、30.2→17.0 µs（1024 req）。

## 验证与回归风险

Engine harness（Qwen3-4B targets，MTP 用 Qwen3.5-0.8B，single_turn）：EAGLE3 K=3 crafted 576/1905 mask 行错误→0/1896；online Poisson（λ=0.2）43/23178→0/23178；MTP/DSpark crafted 32.6%/39.5% falsely skipped→0。`VLLM_GPU_SYNC_CHECK=error` 无 host sync。`pytest tests/watermarking/test_watermarking.py -k reads_current_prompt_lens` 在 main 失败、修复后通过，全套 148 passed（B300）；该测试 CUDA-only，CI 的 CPU "V1 Others" 步骤会跳过。风险：MTP/DSpark "after" 数字来自 harness 补丁而非 commit tree；约 1–2% 低流量 drafts 受影响；TTFT 可能略升。

## 关联

PR 正文提及 #56807（引入该 bug）、#59554（复核时发现，DFlash 亦会暴露）。素材未列其他关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
