# PR #50911: [Spec Decode] Enable fused non-causal TokenSpeed MLA for DSpark

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @NVShreyas |
| 合并 | 2026-08-04 |
| 分类 | Speculative Decoding（tier: major） |
| Label | ready, verified |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/50911 |

## 一句话结论

为 DSpark 启用 TokenSpeed MLA 的 fused non-causal 多 token 路径，B200 kernel 最快 7.3x，e2e 聚合吞吐 +3.1%。

## 动机

DSpark 在单次 non-causal forward pass 中产出全部 draft positions，但 TokenSpeed 未声明该能力，且 vLLM 未把 metadata 的 causal mode 转发给 kernel，导致 draft model 无法使用 TokenSpeed 的 fused multi-token path，只能走 shipped flattened FlashInfer path。TokenSpeed kernel 本身已接受 `causal_mask`，无需改 kernel。

## 优化点

为 TokenSpeed MLA 声明 non-causal 与 non-causal multi-token 支持（`supports_non_causal_multi_token_decode = True`、`supports_non_causal()` 返回 `True`）；在 `forward_mqa` 中把 `attn_metadata.causal` 作为 `causal_mask` 传给 `tokenspeed_mla_decode`（已有 causal 请求继续传 `True`，kernel 无需改动）；扩展既有 backend contract 测试，覆盖 capability selection、causal DCP decode 与 non-causal multi-token decode，并断言 `decode_call["causal_mask"] is causal`、DCP 下 `causal_seqs` 传 `dcp_tot_seq_lens`、非 DCP 下为 `None`。

## 关键改动

```text
vllm/v1/attention/backends/mla/tokenspeed_mla.py: 新增 supports_non_causal_multi_token_decode / supports_non_causal()，并在 forward_mqa 把 attn_metadata.causal 作为 causal_mask 传给 tokenspeed_mla_decode
tests/v1/attention/test_mla_backends.py: 新增 capability 测试并把 decode_contract 参数化为 causal-dcp 与 noncausal-multi-token，断言 causal_mask 语义
```

## 性能声明

测试条件：Kernel measurements 用 Kimi-K3 MLA shapes，单 B200，FP8 KV cache，64K context，7 个 draft tokens，CUDA graphs，对比 shipped flattened FlashInfer path；e2e 为 Kimi-K3 TP8 64K/400 sweep（3 draft tokens、synthetic acceptance lengths from MTBench）；GSM8K 为 1,319-example 5-shot、real block rejection。per-layer draft attention time（concurrency / flattened / fused / speedup）：1 / 51.6 us / 22.6 us / 2.3x；8 / 337 us / 57.7 us / 5.8x；32 / 1.35 ms / 202 us / 6.7x；64 / 2.81 ms / 386 us / 7.3x；128 / 5.11 ms / 789 us / 6.5x。数值：对 flattened non-causal reference 在 2/64/128 requests 与 seq len 256 为 bit-exact，其余 shape 最大绝对差 0.023（不同 split-K reduction order），causal control 差至多 0.257；causal 回归检查 c=128 为 787 us after vs 784 us before。E2E（c=8）：No SpecDec / FlashInfer target 267.06 tok/s、56.93 tok/s/user、mean TTFT 4.751 s；TokenSpeed target + FlashInfer draft 287.44 / 63.79 / 4.458 s；TokenSpeed target + TokenSpeed draft 296.23 / 69.96 / 4.673 s。TokenSpeed/TokenSpeed 相比同 target 的 FlashInfer draft 聚合吞吐 +3.1%、per-user +9.7%，相比 no SpecDec +10.9% / +22.9%，mean TTFT 保持 <5 s。GSM8K：0.9651±0.0051 vs no SpecDec 0.9666±0.0049，zero request errors，绝对差 0.0015 小于不确定度。

## 验证与回归风险

测试：`test_mla_backends.py -k 'tokenspeed_mla_noncausal_capability or tokenspeed_mla_decode_contract'` 3 passed / 2,886 deselected；`test_mla_noncausal.py` 6 passed；`pre-commit run --from-ref upstream/main --to-ref HEAD` all hooks passed；GPU-only TokenSpeed matrix cases 在本地 CPU 开发机 skip，硬件验证由 B200 kernel 与 e2e 结果提供。回归风险：改动覆盖 TokenSpeed MLA 的 capability 声明与 causal_mask 转发，涉及 causal DCP decode 与 non-causal multi-token decode 两条路径；causal 性能回归检查 787 us after vs 784 us before（c=128）；PR 注明 kernel 无需改动、已有 causal 请求继续传 `True`。

## 关联

无（正文的 Duplicate-work check 明确说明未发现 open vLLM PR / issue，未给出其他 PR 编号线索）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
