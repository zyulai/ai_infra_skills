# PR #30107: [diffusion] perf: add unified SP shard helpers and zero-copy tail-pad attention

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-07-04 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数;diff 另有 7 个文件因超长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/30107 |

## 一句话结论

统一 diffusion 序列并行的 shard/pad 语义并把 pad 移到全局尾部,USP 注意力零拷贝跳过 pad,较旧 gap-cat 路径快 4–8%。

## 动机

每个 SP(ulysses)序列在 all-to-all 前必须 pad 到 SP 度的倍数。仓库里有四份独立的 pad 实现、三种语义,其中两份是错的:flux/flux_2/qwen_image 的 pad+gap mask 正确但三份逐字复制的 helper 且 attention 侧 gap 路径每次调用用全序列 `torch.cat` 重排 q/k/v;mova 对 video+audio tower 零 pad 后无 mask 调 attention,pad token 进入 softmax 污染所有真实 token(pad 占比 1/16 时 MAE 达 1.2e-2);`base.shard_latents_for_sp` 在非整除时间维上有同样的无 mask pad 问题;ernie_image 的 [image, text] 联合流是全复制的,却仍跑了 ulysses all-to-all,重复序列。

## 优化点

建立单一不变式(`sp_shard_utils.py`):pad 永远位于最后一个 rank 本地 chunk 的末尾,使 gather 后的序列只在全局尾部有一段连续 pad。据此 USPAttention 走零拷贝尾部路径:按请求构建一次 `cu_seqlens_tail`,把每个 batch 行拆成 [valid | pad] 两个 varlen 段,直接把 padded 布局交给 varlen FA,仅做连续 reshape,无 repack cat、无 index gather;旧 gap 语义键保留为别名,非尾部 span 仍走旧 repack fallback。joint [text, image] 模型用 `join_seqs` 把最后一个 rank 的 text 尾 pad 挪到 image 之后(cat 拷贝量不变),flux single blocks 配套重排 RoPE cache(构建一次)。`plan_text_strategy` 实测 sharding 在所有 text 长度上都赢,默认 always-shard(ring>1 且 text 非整除时回退 replicated,`SGLANG_SP_TEXT_SHARD_MIN` 留作逃生口)。附带修复:mova 向两个 tower 传递尾部 meta;非整除时间维停止 shard(正确性优先);ernie 在流真正被 shard 前跳过 SP;`shard_rotary_emb_for_sp` 并入 `shard_seq(pad_mode="repeat_last")`。作者考虑过自写 kernel 后放弃:尾部布局下 FA 周边数据移动已接近零。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/distributed/sp_shard_utils.py: 新增 SpShard/shard_seq/join_seqs/tail_attn_meta/plan_text_strategy
python/sglang/multimodal_gen/runtime/layers/attention/layer.py: USPAttention 增加零拷贝 cu_seqlens_tail varlen FA 路径
python/sglang/multimodal_gen/runtime/models/dits/flux.py, flux_2.py: 删除各自的 _shard_text_for_sp,改用统一 helper
python/sglang/multimodal_gen/configs/pipeline_configs/base.py: shard_rotary_emb_for_sp 折叠;非整除时间维跳过 shard
python/sglang/multimodal_gen/runtime/models/dits/ernie_image.py: 复制流下 skip_sequence_parallel=True
```

## 性能声明

2×H100,sp=2 ulysses,bf16,FA backend。attention 级 pad 路径对比(B=1,H=24,D=128):txt=77/img=2048 每 rank 0.483 vs 旧 gap-cat 0.526 ms;txt=511/2048 0.562 vs 0.614;txt=77/8192 3.290 vs 3.443;txt=513/8192 3.718 vs 3.745——尾部路径接近整除 baseline,pad 成本消除,比旧路径快 4–8%。MAE vs sp=1:tail/gap 约 1e-5–3e-6(exact),unmasked 随 pad 占比恶化。replicate-vs-shard:各长度 shard 均更快(如 txt=64 0.461 vs 0.537 ms)。E2E QwenImage DiT(4 层,同随机权重):rel MAE 5.075e-3 与 main bit-identical;forward 时延在噪声内持平(6.45/6.10/5.91 vs 6.10/6.03/6.27 ms,ABAB×3)。

## 验证与回归风险

新增 `test_sp_shard.py`(17 例:shard 数学、尾部 meta vs 旧 gap 公式、策略门控、gather/trim)。风险面:零拷贝路径要求 pad 恰在全局尾部(`meta_pad_end == seq` 才走),否则回退 repack;masked SP 路径不支持 ring(非整除 text + ring>1 强制 replicate);mova/base/ernie 的行为变化是 bugfix,依赖模型消费 sp_shard meta;非 FA backend 走 meta-only 生成的 gathered mask。

## 关联

无具体 PR 编号(素材中的后续方向:hunyuan3d/hunyuanvideo 的 LocalAttention 需 attention-backend 迁移;ernie 待 latents sharding 后经 `num_replicated_suffix` 完整 sharding)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
