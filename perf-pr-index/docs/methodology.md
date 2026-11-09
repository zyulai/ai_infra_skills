# 方法论:收录、分类与分级标准

## 0. 数据快照与更新基线

每次抓取后运行 `scripts/snapshot.py`,把两个上游仓库的 HEAD commit id、API 抓取
窗口与各层数量写入 `data/snapshot.json`;`build_index.py` 会把它渲染到
`index/*/PERF-PRS.md` 和 `index/STATS.md` 的「数据快照」一节。

**本次(v1)基线**:

| 仓库 | 上游 HEAD | API 抓取窗口(UTC) |
| --- | --- | --- |
| vllm-project/vllm | `c9e4cb2b793b` (2026-11-07) | 2026-11-07 02:54 ~ 03:15 |
| sgl-project/sglang | `8920463c90c8` (2026-11-07) | 2026-11-07 02:50 ~ 03:28 |

即:本库收录的是**截至上述 commit 为止**的已合并 PR。下次更新时,新旧两次
snapshot 的 HEAD diff 就是增量范围;证据卡只需为新增重点档 PR 补写。

## 1. 收录范围(全量候选)

对 `vllm-project/vllm` 与 `sgl-project/sglang`,收集满足以下**任一**条件的已合并 PR
(`is:pr is:merged`):

1. `label:performance`(两仓库均有该 label);
2. 标题含以下关键词之一(每词一条独立查询,规避 issues search API 不支持
   括号/OR 优先级的问题):
   `perf / speedup / faster / optimize / optimization / optimized / fused / fusion /
   overlap / cuda graph / flashinfer / deepep / speculative / eagle / mtp /
   quantization / quantize / kernel`;
3. 以下高价值 label 之一(vLLM):`quantization, speculative-decoding, torch.compile,
   kv-cache-manager, kv-connector, scheduler`;
   (SGLang):`kernel, sgl-kernel, quant, quantization, release-highlight, flashinfer,
   jit-kernel`。

Search API 单查询上限 1000 条 → 总数 >950 的查询自动按季度时间窗
(`merged:YYYY-MM-DD..YYYY-MM-DD`)切片,窗口仍超限再按月切片。
原始记录含搜索结果全部有用字段(title/labels/body 前 30k 字符/作者/评论数/reactions
/时间),按 PR 号去重、多查询命中的 label 取并集。

**已知盲区**(_keyword 法的固有遗漏,后续可人工补录)_:
- 标题与 label 均无 perf 信号、正文性能描述含糊的优化(如纯重构带来的收益);
- 只在 release note 中提及的优化;
- 关键词误伤:`kernel`/`quantization` 等词会命中模型支持类 PR —— 由分类规则
  `CI / Test-Perf`、`Model-specific Fast Path` 与分级规则兜底,不删除、只降权。

## 2. 分类(13 类,规则优先级从上到下,首中即停)

| 优先级 | 分类 | 标题正则示例(完整规则见 `classify.py`) |
| --- | --- | --- |
| 1 | CI / Test-Perf | 标题以 `test/bench/ci` 开头且无优化措辞 → 永远不进重点档 |
| 2 | Speculative Decoding | `speculative, eagle, mtp, medusa, ngram, draft model, lookahead` |
| 3 | MoE | `moe, deepep, fused-moe, expert parallel, all-to-all, grouped gemm` |
| 4 | Quantization | `quant, fp8/fp4/int8/int4, awq, gptq, marlin, modelopt, nvfp4, w4a8` |
| 5 | CUDA Graph / Compiler | `cuda graph, piecewise, torch.compile, jit, inductor, graph capture` |
| 6 | Overlap / Scheduler | `overlap, scheduler, two-batch, tbo, disaggregat, chunked prefill, zero-busy` |
| 7 | Attention / Kernel | `attention, flashinfer, flash-attn, mla, nsa, triton, kernel, gemm, rmsnorm, rope` |
| 8 | KV Cache / Memory | `kv cache, paged, radix cache, prefix cach, lmcache, hicache, memory pool, offload` |
| 9 | Parallelism / Communication | `tensor/pipeline/data parallel, allreduce, nccl, nvshmem, sequence parallel` |
| 10 | Sampler / Detokenization | `sampler, sampling, detokeniz, logits, top-k/p, penalty` |
| 11 | Frontend / Serving | `router, load balanc, api server, openai server, tokenizer, max-concurrency` |
| 12 | Model-specific Fast Path | 未命中技术类,但标题含知名模型族名(deepseek/qwen/glm/llama/…) |
| 13 | Other | 以上皆未命中 |

正文规则仅在标题全空时使用(扫描正文前 12k 字符),避免正文偶发词误导分类。

## 3. 分级(重点 🔥 / 一般)

满足**任一**条件即为重点档(`CI / Test-Perf` 类除外):

1. **正文含带数字的加速声明**(正则提取,如 `2.5x faster`、`up to 30%`、
   `throughput improved by 25%`、`speedup of 1.8`),数字存入 `speedup_numbers`;
2. **高热度**:reactions ≥ 10 或 comments ≥ 25;
3. **高价值 label + 强标题关键词**:label 命中高价值集合 **且** 标题含
   `fused/kernel/cuda graph/overlap/deepep/flashinfer/eagle/mtp/speculative/triton/
   mla/nsa/flash-attn/quant/rope/rmsnorm/allreduce/zero-busy/two-batch` 之一;
4. **无数字的优化声明 + (高价值 label 或强标题关键词)**。

未命中 → 一般档。判断只用标题/label/正文/reactions,不依赖网络请求。

## 4. 证据卡(重点档,diff 级)

每张卡(`index/<repo>/cards/pr-<N>-<slug>.md`)包含:

- PR 元信息:编号、标题、作者、合并日期、label、分类、链接、diffstat;
- **动机**:PR 正文提炼(解决什么问题);
- **优化点**:算法/内核/调度路径层面改了什么;
- **关键改动**:核心文件与代码摘录(来自 `.diff`);
- **性能声明**:数字 + 测试条件(模型/硬件/精度/workload,以 PR 原文为准);
- **验证与风险**:benchmark/accuracy 通道、可能的回归面;
- **关联 PR**:同区域的历史优化(链回本索引)。

生成流程:`fetch_prs.py --enrich data/major/<repo>-major.jsonl` 拉取
merged_at/diffstat → `generate_cards.py` 为每个 PR 输出素材(正文要点 + diff 摘录)
→ Agent 按模板撰写。分批执行,SGLang 全部重点档先行,vLLM 按分类分批。

## 5. 限制与诚实声明

- `speedup_numbers` 是正则产物,**未经人工核实**,STATS 的加速 Top 榜只代表
  "PR 作者声称的数字";请点击 PR 核对测试条件。
- closed_at ≈ merged_at(已合并 PR);若需精确 merged_at,重点档已通过
  `/pulls/{n}` 富化,一般档保留 closed_at。
- 分类/分级规则为启发式,`--list` 输出各分类计数便于人工抽查;
  误分类可通过修改 `classify.py` 规则后重跑 `classify.py && build_index.py` 修正。
- GitHub Search API 不保证 100% 覆盖(偶发索引延迟/限流丢弃页),
  脚本会打印告警;对齐锚点 PR(见 README 验证节)可评估覆盖率。
