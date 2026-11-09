# PR #52816: [Spec Decode] DFlash2: local convolution + candidate selector

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @SubSir |
| 合并 | 2026-08-21 |
| 分类 | Speculative Decoding (tier: major) |
| Label | new-model, speculative-decoding, ready, needs-rebase, qwen, mrv2, dflash |
| Diff | diffstat 缺失(pack 中为 +None/-None,None files;diff 内容可用,约 13 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/52816 |

## 一句话结论

新增 DFlash2 drafter(分组动态深度卷积 + 格候选选择器):GSM8K 接受长度较 DSpark +25.2%,吞吐最高 3.51x 自回归。

## 动机

DFlash v1 drafter 每个 slot 独立 argmax,proposal 位置看不到同块中前序位置的结果,要看得再跑一次 backbone。DFlash2 用单独的 `DFlash2DraftModel` 架构承载两项升级:块内分组动态深度卷积让 proposal 位置无需额外 backbone pass 即可感知前序位置;候选选择器用目标 head 的 top-K 构建格并走最优路径,替代逐槽位独立 argmax。既有 `DFlashDraftModel` checkpoint 的解析行为不变。

## 优化点

(1) 分组动态深度卷积:`out[i,c] = Σ_t (base[t,c] + δ[i,t,g(c)]) · x[i−t,c]`,块边界处 tap 为零,每个 sublayer 的输入经一次投影得到 in/out 包装;尺寸由 draft config 的 `conv_kernel_size`/`conv_group_size` 决定。(2) 候选选择器:保留目标 head 每 slot 的 top-K,给相邻转移打分 `edge(p→c) = ⟨A[p] ⊙ project(h), B[c]⟩ + unary[c]`,从验证锚点出发走最优路径;T>0 时按逆 CDF 采样并返回 K 个候选上的 q 分布用于无损验证;尺寸由 `selector_rank`/`selector_top_k` 决定。(3) path walk 是每请求一个 Triton program:K 个分数留在寄存器,槽位间依赖用 program 内循环而非每槽一个 kernel。(4) 词表 top-k(选择器最大单项开销)在有 FlashInfer 时用 radix kernel,否则退回 `torch.topk`;`LogitsProcessor` 新增 `get_top_k_tokens`,通信量 O(batch*2k*tp_size) 而非 O(batch*vocab)。(5) DFlash2 只跑在 V2 model runner:`use_v2_model_runner` 对 DFlash2 draft 强制 V2(否则 V1 `DFlashProposer` 会静默降级为 DFlash1)。

## 关键改动

```text
vllm/model_executor/models/qwen3_dflash2.py: 卷积与候选选择器模型实现
vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py: DFlash2Speculator、_selector_walk_kernel(每请求单 Triton program 走格)、_cache_draft_logits_kernel
vllm/model_executor/layers/logits_processor.py: get_top_k_tokens(vocab-parallel top-k,FlashInfer radix kernel 优先)
vllm/config/vllm.py: _is_dflash2_draft 强制 V2 model runner
vllm/v1/worker/gpu/sample/gumbel.py: gumbel_noised_argmax 供 walk 采样使用
```

关键代码(转移打分与 walk 的选择器接口,见 qwen3_dflash2/speculator):

```python
# edge(p→c) = <A[p] ⊙ project(h), B[c]> + unary[c]  (PR 正文公式)
candidate_ids, unary_logits = self.model.compute_candidates(...)
scores = self.model.model.candidate_selector(
    candidate_ids, unary_logits, hidden_states, anchor_token_ids)
self._sample_path(candidate_ids, scores, num_reqs)
```

## 性能声明

Qwen/Qwen3.8-27B,单张 H200,对比 RadixArk/Qwen3.8-27B-DSpark 与自回归;两者均每验证步 7 个 draft token;温度 1.0、top-p 0.95、top-k 20、xhigh reasoning、4096 max new tokens。GSM8K 接受长度(per-request mean):DSpark 4.27 vs DFlash2 5.34(+25.2%);pooled ratio 3.64 vs 4.65(+27.8%),且 DFlash2 在并发 1/8/32 下 pooled ratio 4.65/4.82/4.74 基本不降。GSM8K 吞吐(tok/s,单 GPU 数字测四次取平均):并发 1 自回归 64.1 / DSpark 178.5(2.79x)/ DFlash2 224.6(3.51x);并发 8 为 416.4 / 966.2(2.32x)/ 1,211.7(2.91x);并发 32 为 1,256.9 / 2,220.9(1.77x)/ 2,759.4(2.20x)。组件开销(5 层、hidden 5120、vocab 248320、block 8、K=16):conv+selector 在 batch 1/8/32 为 0.174/0.227/0.370ms,占 serving step 的 0.84%/0.71%/0.67%;FlashInfer radix top-k 比 torch.topk 快 1.9x(batch 1)至 4.5x(batch 32);卷积 20 次小调用 compile 后 0.0096ms(单图)/0.113ms(逐调用编译)。

## 验证与回归风险

新增 `tests/v1/spec_decode/test_dflash2.py`(4 passed)与 `tests/test_config.py` 的 dflash2 强制 V2 用例(1 passed);DFlash v1 的 causality/lookahead/prepare_inputs 测试 19 passed(`test_dflash_causality.py` 在本 PR 内被编辑,作者注明旧版测试在新代码上也通过)。风险面:改动共享 `LogitsProcessor`(新增方法,不动原路径)与 `qwen3_dflash.py` 的 `_dflash_layer_causal` 因果判定;gumbel.py 被共享给 walk 采样;V2 runner 强制逻辑影响 DFlash2 draft 的部署配置。

## 关联

同区域:DFlash v1 系列改动 #52559(adaptive K)、#47511(SWA)、#46911(fused cache insert)、#52782(profiling annotations);DSpark(PR #46995,基线对比对象);Kimi K3 DSpark(#50000)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
