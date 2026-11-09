# PR #4218: Support nextn for flashinfer mla attention backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Fridge003 |
| 合并 | 2025-03-09 |
| 分类 | Speculative Decoding(tier: major) |
| Label | high priority |
| Diff | pack diffstat 缺失;diff 为 flashinfer_mla_backend.py、deepseek_v2.py、eagle_worker.py、deepseek 文档与 MLA 测试 |
| 链接 | https://github.com/sgl-project/sglang/pull/4218 |

## 一句话结论

让 FlashInfer MLA attention backend 兼容 NextN/EAGLE 投机解码:8×H200 上 DeepSeek-R1 长输入(4000/200)吞吐 3757 → 5702 tok/s。

## 动机

nextn(DeepSeek MTP)与 flashinfer mla attention backend 不兼容。由于 flashinfer MLA wrapper 缺少 custom mask 支持,投机解码此前无法在 `--enable-flashinfer-mla` 下运行(topk 只能为 1 也源于此限制)。长输入场景下 flashinfer mla 显著快于 Triton 路径,补齐兼容即可把投机加速与该 backend 组合。

## 优化点

两个层面:(1) 新增 `FlashInferMLAMultiStepDraftBackend`,把 `speculative_num_steps` 个 `FlashInferMLAAttnBackend` 实例包装为一个多步 draft backend——每个 step 一个实例,`skip_prefill=True` 跳过 prefill wrapper 分配,各 step 共享 per-step `kv_indptr` buffer 与 `q_indptr_decode`;用 `generate_draft_decode_kv_indices` Triton kernel 一次性生成各 step 的 KV indices;topk > 1 直接抛错。(2) `FlashInferMLABackend` 本体支持 draft extend 与 target verify 两类 batch:verify 用专用 `prefill_wrapper_verify`(`BatchMLAPagedAttentionWrapper`),CUDA graph capture/replay 增加 target_verify 分支;decode 与 prefill 的 `call_begin_forward` 接受 `spec_info`——存在时 `kv_indptr/kv_indices` 由 `spec_info.generate_attn_arg_prefill` 提供(即 custom mask 路径,代码中标注 TODO: 以 custom mask 支持 topk > 1)。另将 `deepseek_v2.py` 的 `no_absorb` 条件排除 target_verify/draft_extend 模式,并在文档注明与投机解码同用时 `speculative_eagle_topk` 须为 1。

## 关键改动

```text
python/sglang/srt/layers/attention/flashinfer_mla_backend.py: 新增 FlashInferMLAMultiStepDraftBackend;backend 支持 draft extend/target verify 及其 CUDA graph
python/sglang/srt/speculative/eagle_worker.py: attention_backend == "flashinfer_mla" 时构建多步 draft backend
python/sglang/srt/models/deepseek_v2.py: no_absorb 排除 target_verify/draft_extend
test/srt/test_mla_flashinfer.py: 新增 TestFlashinferMLAMTP 集成测试
```

```python
if spec_info is None:
    kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
    ...
else:
    # TODO: Support topk > 1 with custom mask
    kv_indices, kv_indptr, qo_indptr, custom_mask = (
        spec_info.generate_attn_arg_prefill(
            req_pool_indices, paged_kernel_lens, paged_kernel_lens_sum, self.req_to_token
        )
    )
```

## 性能声明

8×H200,DeepSeek-R1,EAGLE + `lmsys/DeepSeek-R1-NextN`(num-steps 3、topk 1、num-draft-tokens 4),TP8,bench_serving random 数据集,每组 5 次取平均,total throughput(tok/s):

| workload | Flashinfer + NextN | Flashinfer Only | Triton + NextN |
|---|---:|---:|---:|
| Input-4000-Output-200(32 prompts) | 5702.74 | 4047.45 | 3757.34 |
| Input-128-Output-128 | 905.28 | 630.62 | 929.86 |
| Single prompt(send_one) | 54.83(acc length 2.22) | 37.71(1.0) | 55.25(2.34) |

精度:GSM8K(8-shot、200 questions、parallel 128)accuracy 0.975、invalid 0.000、latency 93.031 s、output throughput 213.779 tok/s;MMLU(nsub 100、ntrain 5、parallel 128)accuracy 0.870。PR 未注明精度测试所用硬件。

## 验证与回归风险

新增 `TestFlashinferMLAMTP` 集成测试(启动 EAGLE + NextN + flashinfer mla 服务,gsm8k accuracy > 0.60 且 `avg_spec_accept_length > 2.5`)。约束与风险:`speculative-eagle-topk` 必须为 1(wrapper 无 custom mask 之外的完整支持,代码留 TODO);`speculative-num-draft-tokens` 须为 2 的幂;短输入(128/128)下 Triton + NextN(929.86)仍快于 FlashInfer + NextN(905.28),收益集中在长输入。

## 关联

无(pack 未给出同区域其他 PR 编号;draft 模型为 DeepSeek-R1 的 NextN/MTP 头)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
