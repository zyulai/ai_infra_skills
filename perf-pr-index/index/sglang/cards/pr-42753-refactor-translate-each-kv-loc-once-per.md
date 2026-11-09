# PR #42753: [Refactor] Translate each KV loc once per iteration

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ch-wan |
| 合并 | 2026-10-08 |
| 分类 | Model-specific Fast Path |
| Label | deepseek, speculative-decoding, blackwell, run-ci, jit-kernel, run-ci-extra, memory-pool, bypass-fail-fast, parallel-stages |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用,另有 89 个文件因超长被剔除) |
| 链接 | https://github.com/sgl-project/sglang/pull/42753 |

## 一句话结论

统一内存池下 KV 虚拟/物理 id 转换收敛为"每次迭代一次写翻译 + 每次读一次翻译",减少重复 kernel 启动并消除双重翻译隐患。

## 动机

统一内存池中 KV slot 有虚拟 id(scheduler 分配、`req_to_token` 存储)与物理 id(kernel 索引)两种,转换散落各处:每个 ForwardBatch 翻译自己的写窗口,各 attention backend 各建翻译后 page table(per forward / per layer group / per captured mode 不等),multi-step draft kernel 逐 id 翻译,投机 KV writer 再翻译拿到的 id。一次投机迭代内同一窗口被翻译 3-4 次(draft、verify、draft extend、target-hidden KV 写),捕获 graph 还会把写 id 二次翻译进私有 buffer。除浪费的 launch 外,各点必须一致约定哪些 id 已是物理的,翻两次或不翻都会静默读写错误 slot。

## 优化点

新增 `KVLocPlan`(`mem_cache/kv_loc_plan.py`):一次迭代的 id 计划,按子池(`IdSpace`:FULL/SLIDING_WINDOW)组织——写窗口只翻译一次(`write_physical`,原虚拟窗口不变供 radix tree/accept 路径读);`write_ids(reader, kind, cols)` 按需派生各子池的写 id;`read_table(kind, rows)` 惰性构建,只有表形读者(page-table kernel 或捕获 graph 的 `copy_page_table`)首次请求时才建,之后共享,且直接建进捕获读者的自有 buffer(无副本);`bind(batch, reader, cols)` 是 forward 获取写 id 的唯一入口(ForwardBatch 只携带 plan 与 `kv_loc_cols`)。`KVIndexTranslator` 提供表构建器、`read_source`(gather kernel 直译)、`pack_read_stream`(CSR 流)、`write_ids`、DCP 读产出等原语。分 7 层落地:每 forward 一个 plan;写路径让捕获 forward 直接写 graph runner 的 `out_cache_loc` 槽(triton 与 trtllm_mla 家族删掉二次翻译的私有捕获 buffer);triton、multi-step draft kernel、fa3、trtllm_mla/flashmla/trtllm_mha/flashinfer/flashinfer_mla/aiter 等后端与投机输入的 `generate_attn_arg_prefill` 改读 plan 表;EAGLE(单/多层)、DSPARK、DFLASH 的 draft/verify/draft-extend/target-hidden 写共享一个 iteration plan;删除旧 per-forward 翻译 API(`index_table_for_batch`、`build_index_table` 等约 8 个);子池统一为 IdSpace;删约 60 个重复/钉死开发期细节的测试。静态池不做任何翻译工作,plan 是几个 host 对象,读走 `req_to_token` passthrough。

## 关键改动

```text
python/sglang/srt/mem_cache/kv_loc_plan.py: 新增 KVLocPlan/IdSpace(写一次翻译、读惰性共享表)
python/sglang/kernels/ops/kvcache/kv_read_table.py: build_kv_read_table 增加 zero_tail;新增 _gather_translate_items
python/sglang/kernels/ops/speculative/cache_locs.py: draft decode kernel 改读 read_source(ENTRY_PAGE_SIZE/TRANSLATE)
python/sglang/srt/layers/attention/flashattention_backend.py: fa3 全量图 prefill 改 copy_page_table,SWA 写 loc 取 plan
多后端(index updater 显式传 plan)+ 投机路径(EAGLE/DSPARK/DFLASH 共享 iteration plan)
```

## 性能声明

- 每迭代翻译次数(H200,25-prompt greedy,server 进程内统计):各配置写翻译 1.00 次/迭代;读从"1 次表构建 + 读约 2.95-2.98 次"变为"2.00 次流 gather + 约 0.97 次在 draft kernel 的 gather 内,不建表"(Qwen3.5-4B MTP/flashinfer、gpt-oss-20b EAGLE3/triton 等);fa3 配置仍 2 次表构建(fa3 读表)。
- 对比 `main`(B200,bench_serving random 1024/256,unified pool,cuda graph on,greedy 输出逐行一致):gpt-oss-20b 无投机 triton:c1 TPOT 1.865→1.840 ms(-1.3%)、c1 吞吐 +1.5%、c32 +10.8%;DFLASH:TPOT -1.1%、c1 +1.4%、c32 +2.7%;Qwen3.5-4B MTP:c1 0.0%、c32 -2.7%。注意:正文说明 c32 两次同树运行差异可达 18%(纯测量噪声,非加速声明),c1 稳定在约 0.3% 内。
- 对比 main(DSPARK unified pool):Kimi-Linear-48B TP2 dummy draft flashinfer 私有→fused:TPOT +1.1%/+1.6%(付一次 plan 构建 launch),但 GPU 内存 127.8 → 90.7 GiB/rank;Qwen3.8-27B 真实 DSPARK draft:fused 后 TPOT +0.4%,池分配后省 14.1 GiB。
- 内存:`max_total_num_tokens` 与栈顶一致;transient 峰值差在 8 MiB 内;不建无人请求的读表(避免 page size 1、256 请求、单行 128k 时约 128 MiB 的表)。

## 验证与回归风险

Greedy parity(24×128 或 25×128 prompts,token id 与 per-token logprob,B200/H200)覆盖无投机/MTP/EAGLE3/DFLASH/DSPARK/NGRAM × flashinfer/triton/fa3/trtllm_mha/trtllm_mla × unified/static 池与 DP attention:**每组与栈顶或 main 完全一致,max |dlogprob| 0.0000**;accept length 逐行一致(DFLASH 3.7465、MTP 2.8355)。页迁移下 GSM8K 400 题(小池,每轮惰性压缩约 31k 页):MTP 0.882/0.875、DFLASH 0.890、无投机 0.873,均不低于栈顶。风险面(正文 Notes):NPU(仅单测覆盖 DCP 写 id)、fa3+DFLASH/DSPARK、多层 EAGLE、cutedsl_mla/tokenspeed_mla/flashmla、`--dcp-size>1`、two-batch overlap 未做端到端验证;trtllm_mla eager block table 行内死页从 -1 改为读 sink(0);CI 状态三项均 no_entry_sign。改动面极大(89 文件被剔除),属全后端重构。

## 关联

同栈(自底向上):PR #37627、#37628、#40491、#40492,本 PR 为栈顶。同区域:统一内存池 + 投机解码翻译审计(kv_cache_hook 中 audited backends 机制)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
