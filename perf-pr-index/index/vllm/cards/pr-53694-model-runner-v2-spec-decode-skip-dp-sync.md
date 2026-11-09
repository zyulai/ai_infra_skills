# PR #53694: [Model Runner V2][Spec Decode] Skip DP sync before EAGLE/MTP draft prefill

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @TheEpicDolphin |
| 合并 | 2026-08-27 |
| 分类 | Speculative Decoding（tier: major） |
| Label | speculative-decoding, ready, mrv2, dflash |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/53694 |

## 一句话结论

V2 runner 复用 DP-synced 值跳过 EAGLE/MTP draft prefill 前的 CPU all-reduce，两个模型吞吐均约 +2.6%。

## 动机

V2 model runner 的 EAGLE/MTP speculator 在每次 draft prefill（drafts first token）与 draft multi-step decode（drafts N-1 tokens）之前，都通过 `sync_cudagraph_and_dp_padding` 做一次 CPU all-reduce，跨 DP ranks 同步 `num_tokens`、`uniform_token_count`、`cg_mode` 等；multi-module MTP speculator 也在 multi-step drafting loop 前做同样的 DP-sync。该 CPU all-reduce 在服务期间带来显著 CPU 开销。

## 优化点

把 model runner 在 target forward 之前调用 `dispatch_cg_and_sync_dp` 得到的 DP-synced 值（number of tokens、uniform token count、eager）传递给 speculator，复用给 EAGLE/MTP 的 draft prefill 以及 multi-module MTP 的 draft loop，从而跳过这次 CPU all-reduce，并用 DP-synced 值直接取正确的 batch descriptor。`dp_utils.py` 新增 frozen dataclass `DPSyncState`（含 `num_tokens_across_dp`、`uniform_token_count`、`eager`，各字段在所有 rank 上一致），`sync_cudagraph_and_dp_padding` 的返回值从 `(batch_desc, num_tokens_across_dp|None)` 改为 `(batch_desc, DPSyncState|None)`，speculator 接口相应以 `dp_sync` 取代 `num_tokens_across_dp`。

## 关键改动

```text
vllm/v1/worker/gpu/dp_utils.py: 新增 DPSyncState dataclass，并让 sync_cudagraph_and_dp_padding 返回 (batch_desc, DPSyncState|None)
vllm/v1/worker/gpu/model_runner.py: 把 dispatch_cg_and_sync_dp 的 DP-synced 值传递给 speculator
vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py: EAGLE/MTP draft prefill 复用 dp_sync，跳过额外 CPU all-reduce
vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py: multi-step drafting loop 前复用 dp_sync
vllm/v1/worker/gpu/spec_decode/speculator.py、dflash/speculator.py、extract_hidden_states.py: 接口由 num_tokens_across_dp 改为 dp_sync
```

## 性能声明

测试条件：GSM8K（1319 questions），`dp=2, tp=2, --enable-expert-parallel` on 4×GB200，`vllm bench serve --max-concurrency 64 --custom-output-len 2048 --temperature 0 --top-p 1.0`，吞吐为单机 3 次运行的中位数，精度为单次全量 pass，双侧 `VLLM_USE_BREAKABLE_CUDAGRAPH=0`。DeepSeek-V4-Flash + MTP=3（`AutoRegressiveSpeculator`）：输出吞吐 2946.74→3024.36 tok/s（+2.6%）、mean TPOT 20.68→20.14 ms（-2.6%）、mean TTFT 365.04→345.02 ms（-5.5%）、GSM8K 0.948→0.950、acceptance rate 60.19%→60.17%、acceptance length 2.81→2.80。Inkling-NVFP4 + MTP=8（`MultiModuleMTPSpeculator`）：吞吐 1763.53→1809.68（+2.6%）、mean TPOT 31.50→30.39（-3.5%）、mean TTFT 1042.77→998.50（-4.2%）、accuracy 0.899→0.894、acceptance 42.98%→42.81%、length 4.44→4.42。PR 总结 accuracy 与 acceptance 不变，两模型吞吐稳定但温和地提升约 2.6%。

## 验证与回归风险

PR 正文未列出专门单测；diff 更新 `tests/v1/worker/test_gpu_model_runner_v2_eplb.py`（构造参数由 `num_tokens_across_dp` 改为 `dp_sync`）。回归面：EAGLE/MTP 与 multi-module MTP 的 draft prefill / multi-step decode 路径，以及 DP>1 下 cg_mode/eager 的一致性——`DPSyncState` 要求各 rank 字段一致，否则各 rank 选取不同 graph 会分歧。正文注明双侧设定 `VLLM_USE_BREAKABLE_CUDAGRAPH=0` 是因 dp>1 下 PyTorch `CachingHostAllocator` 的 capture assert，与本改动无关。

## 关联

无（PR 正文未给出其他 PR 编号线索）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
