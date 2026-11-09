# PR #43171: [Perf] gdn: Select FlashInfer prefill for page-major pools

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @caihuali95 |
| 合并 | 2026-10-18 |
| 分类 | Speculative Decoding (tier: major) |
| Label | memory-pool |
| Diff | diffstat 缺失(pack 未提供 +/− 行数与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/43171 |

## 一句话结论

page-major 池不再禁用 GDN FlashInfer prefill,消除 `--enable-unified-memory` 下 -1.1..-1.5% 吞吐回退。

## 动机

`flashinfer_gdn_prefill_default` 的后端选择策略只要设置了 `enable_page_major_kv_layout` 就直接拒绝。`--enable-unified-memory` 跑的是 page-major 布局,于是 Qwen3.5 GDN 的 prefill 在该模式下只能停留在 Triton chunk kernel,而同一 GPU 上的默认池却自动选中 FlashInfer。SM90 上(Qwen3.5-9B,page size 256,无 spec)测得 unified 池每轮 prefill 周期比默认池长 +1.24..+1.41%,全部是 GPU 时间:Triton chunk kernel 每周期 +17.8 ms,而 FlashInfer GDN kernel 仅 10.7 ms;端到端表现为所有 attention backend 上一致 -1.1..-1.5% 吞吐。

## 优化点

改动很小但机制清晰:从 `flashinfer_gdn_prefill_default` 的拒绝条件中删掉 `enable_page_major_kv_layout` 一项。其正确性依据是 prefill 对布局不敏感——`forward_extend` 在任何 prefill kernel 运行前,通过 `needs_state_gather` 把非连续池的 conv 与 SSM state 按 sequence gather 成连续副本、跑完再 scatter 回去,因此 FlashInfer 的 extend 看到的与默认池完全相同的连续、identity-indexed state。decode 与 verify 不受影响,它们的后端另行解析(verify 跟随 decode,默认 Triton)。同时把 page-major 的 server-arg 白名单(`prefill_allowed`)加入 `flashinfer`,使显式 `--linear-attn-prefill-backend flashinfer` 与自动选择行为一致,避免一个本来合法的 unified-memory 配置因显式 pin 而启动失败。

## 关键改动

```text
python/sglang/srt/layers/attention/linear/gdn_backend.py:
  flashinfer_gdn_prefill_default 删除 enable_page_major_kv_layout 拒绝条件
python/sglang/srt/arg_groups/kv_cache_hook.py:
  handle_page_major_kv_layout 的 prefill_allowed 集合加入 "flashinfer"
test/registered/unit/layers/attention/test_gdn_prefill_backend_policy.py:
  新增 test_selects_flashinfer_for_a_page_major_pool(SM100 bf16 / SM90 fp32)
test/registered/unit/server_args/test_page_major_backend_allowlist.py:
  新增显式 flashinfer prefill 白名单用例
```

```python
# gdn_backend.py:page-major 不再使 FlashInfer prefill 落选
        or get_exec().deterministic.enable_deterministic_inference
-        or get_memory().enable_page_major_kv_layout
        or sm_major not in (9, 10)
```

## 性能声明

bench_serving,Qwen3.5-9B,无 spec,page size 256,radix cache,overlap scheduler,CUDA graphs on,`--mem-fraction-static 0.55`,concurrency 64,random prompts,两种 shape(2048 in/512 out 200 prompts;8192 in/64 out 100 prompts),对比 unified 池相对默认池的差距(main → 本 PR):req/s 差距如 fa3 2048/512 为 -0.9% → -0.4%、flashinfer 8192/64 为 -1.3% → -0.2%、triton 8192/64 为 -1.1% → -0.2%;剩余 TTFT 差距归因于调度(radix cache 缓存 token 量不同,约 61k vs 19k),同 new/cached token 数的 prefill batch 上 unified 吞吐已达默认池的 0.998。A/A 噪声底 0.0-0.3%。精度:GSM8K 1314 题 5-shot greedy,三 backend 变化在 0.893-0.898 内。另:SM90 prefill 周期差 +1.24..+1.41%,Triton chunk kernel +17.8 ms/cycle vs FlashInfer 10.7 ms。作者注明这些为 rebase 前的历史结果。

## 验证与回归风险

新增策略单测在改动前失败(policy 返回 None)、改动后通过;`test_rejects_unvalidated_runtime_modes` 保留 page-major 对其他未验证模式的拒绝。组合栈验证:455 tests + 921 subtests 通过(B300),unified memory GPU 单测 4102 passed/0 failed;作者注明 GitHub CI 对该 head 尚未复验。风险面:仅影响 GDN 线性注意力的 prefill 后端选择与 page-major 白名单;decode/verify 路径不变;KDA-hybrid(MLA)家族另有 cutedsl/helion 分支,不在本 PR 修改范围。

## 关联

本 PR 是一个 stack 的一部分(stack predecessor: #43170;同栈:#43168、#43325、#43326、#44969-#44973)。同区域(GDN/线性注意力后端选择)其余编号 pack 未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
