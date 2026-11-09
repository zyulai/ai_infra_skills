# 方法论:RFC 收录、分类与分级标准

## 0. 数据快照与更新基线

每次抓取后运行 `scripts/snapshot.py`,把上游 HEAD commit id、API 抓取窗口与各层
数量写入 `data/snapshot.json`;`build_rfc_index.py` 会把它渲染到所有生成 md 的
「数据快照」一节。

**本次(v1)基线**:

| 仓库 | 上游 HEAD | API 抓取窗口(UTC) |
| --- | --- | --- |
| vllm-project/vllm | `1ecc9cc4ce66` | 2026-11-07 12:33 |
| sgl-project/sglang | `ba5906471c62` | 2026-11-07 12:33 |

本库收录的是**截至上述时间**的 RFC issue 全量;下次更新以新旧 snapshot 的
HEAD diff 为增量范围,新增重点档用 `make_batches.py` 补卡(自动跳过已有卡)。

## 1. 收录范围(全量)

GitHub issues,以下两条查询的**并集**(按 issue 编号去重,labels 取并集,
`sources` 字段记录来源):

1. `repo:<org>/<repo> is:issue label:rfc`
2. `repo:<org>/<repo> is:issue RFC in:title`(GitHub 搜索大小写不敏感,
   覆盖 `[RFC]`/`[rfc]`/`RFC:` 等变体)

两仓库均在仓库内无 RFC 文档目录(GitHub API contents 核实过),issue 即唯一载体。
Search API 单查询上限 1000 条,当前两查询均未超限(908/719 与 197/18),
无需时间窗切片;正文保留前 30k 字符,评论数/reactions 取自搜索结果原字段。

**已知盲区**:

- 标题不含 "RFC" 且未打 `rfc` label 的设计讨论(如纯 roadmap issue)不会收录;
- issue 评论区不抓取 —— RFC 的最终结论常在评论里,卡片「现状与落地」只反映
  正文可见信息,深入结论请点原帖;
- Discussions 里的提案不在 Search API 范围内。

## 2. 分类(15 类,标题正则首中即停)

| 分类 | 标题正则要点 |
| --- | --- |
| CI / Infra / Process | `^(test\|bench\|ci)`、benchmark、contribut、roadmap、governance |
| Speculative Decoding | speculative、eagle、mtp、medusa、ngram、draft、lookahead |
| MoE | moe、deepep、expert、all-to-all、grouped gemm |
| Quantization | quant、fp8/fp4/int8/int4、awq、gptq、marlin、modelopt、w4a8 |
| Disaggregated Serving | disaggregat、pd disagg、prefill-decode separation、kv transfer、mooncake、nixl |
| CUDA Graph / Compiler | cuda graph、piecewise、torch.compile、inductor、jit、compil |
| Overlap / Scheduler | overlap、schedul、two-batch、tbo、chunked prefill、zero-busy、batching |
| Attention / Kernel | attention、flashinfer、flash-attn、mla、nsa、triton、kernel、gemm、rmsnorm、rope |
| KV Cache / Memory | kv cache、kv connector、paged、radix、prefix cach、lmcache、offload、memory |
| Parallelism / Communication | tensor/pipeline/data parallel、allreduce、nccl、nvshmem、sequence parallel、shard |
| Sampler / Detokenization | sampler、sampling、detokeniz、logits、penalt |
| Frontend / Serving / API | api server、openai、router、load balanc、tokenizer、http、grpc、metrics、structured output |
| Architecture / Engine | engine、worker、multiproc、asyncio、refactor、v1、loopy、lifecycle、core loop |
| Hardware / Backend | npu、ascend、rocm、tpu、cpu、xpu、hardware、pluggable |
| Feature / Ecosystem | support、integrat、adopt、enable、add、dependency(兜底,防 Other 过大) |
| Other | 以上皆未命中 |

正文规则不参与分类(RFC 标题信息量足够,正文偶发词易误导)。

## 3. 分级(重点 🔥 / 一般)

满足**任一**即重点档:

1. **高热度**:reactions ≥ 10 或 comments ≥ 30;
2. **性能信号 + 参与度**:perf 信号(标题命中
   `perf|speedup|faster|throughput|latency|optimiz|fused|overlap|cuda graph|
   overhead|efficient|fast|gpu util`,或正文提取到带数字的加速声明,或
   label 含 performance)且 reactions ≥ 5 或 comments ≥ 10;
3. **强主题词**:标题命中 `fused|kernel|cuda graph|deepep|flashinfer|mla|nsa|
   disaggregat|speculative|eagle|mtp|allreduce|zero-busy|two-batch|pd disagg`。

正文加速声明提取:perf 词与 `%`/`x` 相邻数字同句才算(% 限 1–200,
x 限 1.05–50),剔除表格/代码块片段 —— 与 perf-pr-index 的 `classify.py` 同一套思路。

## 4. 摘要卡(重点档)

每张卡(`index/<repo>/rfc-cards/rfc-<N>-<slug>.md`)按 `docs/rfc-card-template.md`:

元信息表 → 一句话结论 → 提案动机 → 提出的方案 → 优化点与收益(数字照抄)→
现状与落地(open/closed 按正文如实描述)→ 关联 → 免责尾注。

硬性规则:只写 pack 素材里有的信息;300-800 字;不编造;落地情况不猜测。
生成流程:`make_batches.py` 打包正文(≤20k)+ 打分排序(热度×2 + 评论 +
8×声明数 + perf)+ 生成 agent 清单 → 子代理(并发 ≤3)按模板撰写。
`make_batches.py` 幂等,重跑自动跳过已有卡片的 RFC。

## 5. 限制与诚实声明

- 摘要卡为 LLM 生成,基于 RFC 正文,未经人工核实;个别正文在 20k/30k 字符处
  截断的,卡内已注明。
- `closed` ≠ 已实现:很多 RFC 被否决/搁置/转移走。卡片不做落地猜测,
  需要确切结论请看关联 PR 或原帖评论。
- 分类/分级为启发式,规则可在 `classify_rfcs.py` 中调整后重跑
  `classify_rfcs.py && build_rfc_index.py` 全量修正。
