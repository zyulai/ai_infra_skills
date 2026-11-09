# PR #34337: [Spec][LoRA] Support multi-adapter LoRA with EAGLE/NEXTN/DFLASH/DSPARK speculative decoding

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jybsuper |
| 合并 | 2026-08-21 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, lora, run-ci |
| Diff | pack 未提供 diffstat（diff 内容可用：约 16 个文件，另 3 个文件因超长未入 pack） |
| 链接 | https://github.com/sgl-project/sglang/pull/34337 |

## 一句话结论

把 #12903 的 LoRA+NGRAM 组合扩展到 EAGLE/NEXTN/EAGLE3/DFLASH/DSPARK 的多 adapter 共批投机解码，c=1 时 LoRA+spec 吞吐为 LoRA-only 的 2.12x。

## 动机

#11762 的 LoRA 清单要求支持多 adapter 与投机解码同开。此前 `server_args` 硬性拒绝（只放行 NGRAM），且草稿 worker 会因共享的 `enable_lora=True` 尝试把 target 的 adapter 载入层数不同的 draft 模型而启动失败。`TARGET_VERIFY` 前向虽然 `is_extend()` 为真但 `extend_seq_lens(_cpu)` 为 None，会让 triton eager 分段、MoE `_add_moe_lora_info`、csgmv `_determine_chunk_size` 三处直接崩（max(None)/sum(None)/None 比较）。

## 优化点

设计为"adapter 只作用于 target、共享 draft 不带 adapter"（verify 仍从适配后的 target 采样，投机保持无损）：1) `maybe_init_lora_manager` 按 `is_draft_worker` 排除 draft；`ModelRunner.lora_manager` 常驻（未设为 None），全部 LoRA 路径改判 `lora_manager is not None`，draft runner 天然跳过；2) 新增 `get_batch_token_counts` 统一 forward-mode 分派，decode 返回 `(bs, 1)`、TARGET_VERIFY 返回 `(bs * draft_token_num, draft_token_num)`、extend 走原逻辑，三处崩溃点全部收编；3) MoE LoRA cuda-graph buffer 从按请求数改为按 verify 宽度（`max_bs * draft_token_num`）配额——这也修了 NGRAM+MoE LoRA 的欠配；4) triton graph 路径断言 verify 宽度等于 capture 时的 `seg_lens` 宽度，防止静默错位映射 adapter；5) draft 共享的 lm_head/embeddings 一律 `unwrap_lora_layer` 取裸层，lm_head adapter 只影响 accept rate 不影响正确性，载入含 embedding 权重的 adapter 时告警；6) `_check_lora_speculative_compatibility` 放行 EAGLE/NEXTN/EAGLE3/DFLASH/DSPARK，拒绝 `--speculative-adaptive`、`experimental_sgl_trtllm` MoE runner、overlap plan stream。

## 关键改动

```text
python/sglang/srt/lora/utils.py: 新增 get_batch_token_counts / warn_if_adapter_targets_embeddings
python/sglang/srt/model_executor/model_runner.py: lora_manager 常驻 + draft 排除
python/sglang/srt/lora/lora_manager.py: MoE cuda-graph buffer 按 token 配额；embedding 告警
python/sglang/srt/lora/layers.py: 新增 unwrap_lora_layer
python/sglang/srt/server_args.py: _check_lora_speculative_compatibility 替换 NGRAM-only 拒绝
test/registered/unit/model_executor/test_draft_runner_skips_lora.py: 新增 draft 不建 lora_manager 的单测
```

## 性能声明

256 请求 x 512 in / 128 out，adapter round-robin，Qwen3.5-35B-A3B + NEXTN，TP4（4xGB300），output tok/s：c=1 base 无 spec 290.8、base+spec 552.2、LoRA 无 spec 154.4、LoRA+spec 327.3（对 LoRA-only 2.12x）；c=8 时 942.5 vs 758.0（spec 反而降）；c=32 时 2975.4 vs 1419.7（spec 在有无 LoRA 下都低于 1x）。accept length：公开 adapter 下 NEXTN 3.097 -> 3.208（+3.6%），DFLASH/DSPARK 与 base 无统计差异；三个强偏移私有 adapter 下 NEXTN 掉 19-31%、DFLASH 约 -34%、EAGLE3 约 -41%（首 token 接受率同步下降）。正确性：`run_spec_lora_matrix.py` 按 adapter 对比 spec-on/off（NEXTN/DFLASH 于 triton 与 csgmv 全 PASS，DSPARK 9/36 对可比）；单批混三个 adapter 时 16/16 请求各用自己 adapter 的输出前缀。

## 验证与回归风险

单测覆盖参数矩阵、draft-runner 门控、TARGET_VERIFY 分段数学、embedding 告警（base-a-test-cpu 与 1-gpu-small 各一文件）；e2e 测试断言 serving 属性而非文本（batch 形状间贪心解码不可位级复现）。风险：共享 draft 设计使 adapter 偏移越大 accept rate 损失越大（作者明确说明并量化）；不支持 DSPARK 非 static ragged verify、speculative-adaptive、experimental_sgl_trtllm、overlap plan stream；两个已知无关本 PR 的 LoRA 通用问题（TBO children 不重建 LoRA batch info；overlap loading 下 adapter 偶发未生效）。

## 关联

前序：#12903（LoRA+NGRAM）、#28395（被本 PR 取代）、#11762（LoRA 路线图）。同区域：#24007（csgmv+virtual experts，本 PR 依赖其 per-request segment 语义）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
