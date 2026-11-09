# PR #44200: [Feat] watermark: Add DSPARK speculative decoding support

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JustinTong0323 |
| 合并 | 2026-10-29 |
| 分类 | Speculative Decoding |
| Label | documentation, run-ci, run-ci-extra |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 可见改 watermarking core、dspark_worker_v2、eagle_worker_common、验证钩子、文档与测试 |
| 链接 | https://github.com/sgl-project/sglang/pull/44200 |

## 一句话结论

让文本水印功能支持 DSPARK 投机解码:verify 侧强制水印 token 保证可检测性,投机路径水印请求的 TPOT 开销 +3.73%,并修复 EAGLE/NEXTN 水印输出不可检测的回归(#44199)。

## 动机

SGLang 的文本水印(text watermarking)此前只支持 NGRAM、EAGLE/EAGLE3(及经 EAGLE 路径的 NEXTN)投机解码,其他投机算法启动验证直接 fail-closed。DSPARK 验证线性 draft 链,每个 verify 行的上下文是 committed watermark tail 加更早的 draft tokens,需要新的接入方式使水印输出在投机下仍可检测。同时 #44199 指出的回归需要修复:融合 verify-commit 路径(#41175)跳过了 watermark ring append,导致 main 上 EAGLE/EAGLE3 的水印输出不可检测。

## 优化点

设计是 verify-side forcing:被强制(forced)行的 target 分布是 one-hot,因此 DSPARK 的 rejection-sampling accept(无论是否开 block verification)恰好输出水印 token,检测端契约完全不变(单 key、双 key 均可,无 DSPARK 特殊检测逻辑)。实现上:`WatermarkState` 新增 `force_speculative_chain`(把线性链 reshape 后复用 `force_speculative_from_tree`,配缓存的 `tril` 下三角 chain mask)与 `commit_speculative_chain`(用顺序 accept_index 调 `record_speculative` + `append_speculative`);`dspark_worker_v2._forward_decode` 在 verify epilogue 前经 `_force_watermark` 强制水印行,accept 后把 committed tokens 追加进 ring;`eagle_worker_common.run_eagle_verify` 把 watermark append 移出 fused-commit 条件之外以修复回归。draft 侧 keyed sampling 可提高接受率,但需要把 selector 放进 DSPARK 折叠 draft graph 并定义 cropped-vocabulary hash 契约,成本更高,留作后续。

## 关键改动

```text
python/sglang/srt/sampling/watermarking/core.py: force_speculative_chain/commit_speculative_chain/_chain_mask
python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py: _force_watermark 接入 verify 与 commit
python/sglang/srt/speculative/eagle_worker_common.py: watermark append 移出 fused_commit 分支(修复 #44199)
python/sglang/srt/arg_groups/validation_hook.py: 允许 speculative_algorithm=DSPARK
test/registered/kernels/ops/sampling/test_watermark_state.py: DSPARK chain 用例
docs/docs/advanced_features/text_watermarking.mdx: 支持/拒绝矩阵与开销数据
```

线性链 mask 的构造与缓存:

```python
mask = torch.ones((chain_length, chain_length), dtype=torch.bool, device=...).tril()
self.chain_masks[chain_length] = mask  # 按 chain_length 缓存
```

## 性能声明

条件:DeepSeek-V4.1-Flash,8x H200,TP8/EP8,DSPARK block size 5,bench_serving random 1024/512、c8、100 prompts、temperature 0.8、top-p 0.95、top-k 20,两次启动共六次重复的均值:水印开 vs 关,TPOT 5.879 → 6.098 ms(**+3.73%**),聚合输出吞吐 1,226.0 → 1,198.4 token/s(**−2.25%**),mean accept length 4.420 → 4.276(−3.27%)。检测有效性:40 essays x 512 tokens、temperature 1.0、top-p 0.95,DSPARK 中位请求 z 值 2.73(16/40 达 p<1e-3),非投机同 prompt 为 3.06(18/40),错误 key z 0.91;水印关时中位 z 0.21、0/40。GSM8K-100(temperature 1.0)两种模式均 99%、0 截断 0 错误。回归修复效果:Qwen3-8B + EAGLE3 3/1/4,修复前 0/40 请求达 p<1e-3,修复后 40/40(tree 5/8/32 亦 40/40)。

## 验证与回归风险

测试:CUDA watermark selector/state 套件加 DSPARK chain 用例,验证 rejection-sampling accept(含/不含 block verification)输出被强制 token、committed ring 与 context history 等于非投机 decode(合计 22 passed);端到端 `test_watermark.py` 5 passed。CI Base/Extra/AMD ROCm 10 均通过。风险点:DSPARK + DP attention、PD disaggregation 未覆盖(PD 仍被验证拒绝);DFLASH 仍被拒绝,后续 PR 可复用 `force_speculative_chain`;`return_logprob` 时 target_logits 需 clone 以免污染 spec logprobs;chain mask 按 chain_length 缓存于 dict,显存随长度档位增长;fix #44199 改动 EAGLE verify 通用路径,影响所有带水印的投机模式。

## 关联

同区域优化:依赖 #44199(EAGLE verify-append hotfix,本 PR 分支首 commit)与 #41175(fused verify-commit 路径,回归来源);NGRAM/EAGLE/EAGLE3/NEXTN 水印支持为既有前序;同批 PR #33599 提到的 DSPARK aux capture(aggregate_stream 的 ow=None 路径)与该功能相关。draft 侧 keyed sampling 为声明的 follow-up。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
