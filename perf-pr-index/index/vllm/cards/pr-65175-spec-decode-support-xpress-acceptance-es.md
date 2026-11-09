# PR #65175: [Spec Decode] Support XPress + Acceptance Estimator

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tomasruizt |
| 合并 | 2026-11-03 |
| 分类 | Speculative Decoding |
| Label | documentation, speculative-decoding, ready, qwen, mrv2 |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 内容本身可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/65175 |

## 一句话结论

给 XPress 接通 acceptance estimator,AV 开启从损失 26-71% 吞吐变为 c32 以上反而 +21-40%。

## 动机

XPress 的 draft_token_confidence_probs 用 `torch.empty` 分配,只由 `_maybe_predict_acceptance` 写入;DFlash 与 DSpark 都会调用它,但 XPress 从不调用。于是打开 Adaptive Verification(AV)时,预算读取的是未初始化的 confidence buffer,每步只验证第一个 draft token,其余全被剪掉——XPress 开 AV 反而比不开慢约 3x,e2e acceptance rate 掉到 0.005。需要把 XPress 真正接入在线 acceptance estimator。

## 优化点

机制:让 `jacobi_refine_greedy` 在返回 draft ids 的同时返回最后一轮 refine 所用的 logits(`return_logits` 参数),XPress 的两条 `xpress_topc` 路径(top-C 候选分数或全词表 summed bias)都返回 argmax 所依据的那份 logits;`XPressSpeculator._jacobi_refine` 在 acceptance_estimator 存在时把这些 logits 按 (req, draft slot) 行序交给 `_maybe_predict_acceptance`,由在线 logistic 模型按 target 验证结果重校准,为 AV 预算提供每个 draft 位的接受概率。AV 关闭时完全零开销(不返回 logits、不做预测);draft 生成本身不变(位级一致),AV 只改变每步验证多少个 draft token。

## 关键改动

```text
vllm/model_executor/models/qwen3_xpress.py: jacobi_refine_greedy 增加 return_logits,两条 topc 路径返回 argmax 所用 logits
vllm/v1/worker/gpu/spec_decode/xpress/speculator.py: _jacobi_refine 把 logits 喂给 _maybe_predict_acceptance
tests/v1/spec_decode/test_xpress_head.py: 确定性测试扩展到 topc=0/16 并校验返回的 logits
tests/v1/e2e/spec_decode/acceptance_rates/xpress/test_xpress.py: 增加 enable_adaptive_verification 变体
docs/features/speculative_decoding/adaptive_verification.md: 支持列表措辞更新
```

关键片段(speculator 侧接线):

```python
draft, logits = head.jacobi_refine_greedy(
    base_full, h_full, anchor_ids, tok_am1,
    self.num_jacobi_passes,
    return_logits=self.acceptance_estimator is not None,
)
self.draft_tokens[:num_reqs, : self.num_speculative_steps] = draft
if logits is not None:
    num_sample = num_reqs * self.num_speculative_steps
    self._maybe_predict_acceptance(
        logits, self.sample_idx_mapping[:num_sample],
        self.sample_col[:num_sample])
```

## 性能声明

单张 H100,Qwen3-8B + XPress b16 draft,15 draft tokens,greedy,prefix caching 关闭,SPEED-Bench throughput_1K(shuffle,seed 42),AIPerf 并发 1/8/32/64/128、30s 窗口、3 trials:

- AV off 在 main 与本 PR 上曲线相同(每格差 <1%,与 trial 间波动相当)。
- main 上 AV on 每步只验证 1 token,吞吐损失 26%-71%(如 c1:408 → 118 tok/s,AL 3.78 → 1.00)。
- 本 PR AV on:c1/c8 代价 8%(408 → 375;2494 → 2291 tok/s),c32 起反超 AV off +21%-40%(c32:4468 → 5389;c64:4471 → 6242;c128:4967 → 6810 tok/s);AL 从 3.5-3.7 降到 3.0-3.3,更少的验证 draft 使每步 target 更便宜。
- 精度不变:GSM8K 0.885(AV on/off 均是,1319 题)。

## 验证与回归风险

测试:`tests/v1/e2e/spec_decode/acceptance_rates/xpress/test_xpress.py` 新增 AV 变体(进入 nightly XPress job,main 上该用例失败:acceptance_rate=0.005 < 0.375;本 PR 2 passed,AV off 0.43/7.39,AV on 0.42/7.26);`pytest -v tests/v1/spec_decode/test_xpress_head.py` 4 passed(覆盖两条 xpress_topc 路径,断言无 logits 请求时 draft 不变)。风险:XPress + AV 组合路径;logits 复用 scratch buffer(bias 被 add_ 原地覆盖),依赖"当轮即消费"的时序;其他 drafter(DFlash/DSpark/EAGLE/MTP)不受影响。

## 关联

同区域:DFlash、DSpark(已接 `_maybe_predict_acceptance` 的前例)、Adaptive Verification 框架本身;pack 未列出更具体的前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
