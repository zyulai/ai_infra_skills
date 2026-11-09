# 方法论:Bug Fix 收录、分类与分级标准

## 0. 数据快照与更新基线

每次抓取后运行 `scripts/snapshot.py`,把上游 HEAD commit id、API 抓取窗口与各层
数量写入 `data/snapshot.json`;`build_bugfix_index.py` 会把它渲染到所有生成 md 的
「数据快照」一节。

**本次(v1)基线**(具体 commit 与抓取窗口见 `data/snapshot.json` 及各索引 md 尾部):

| 仓库 | 上游 HEAD | API 抓取窗口(UTC) |
| --- | --- | --- |
| vllm-project/vllm | 见快照 | 见快照 |
| sgl-project/sglang | 见快照 | 见快照 |

本库收录的是**截至上述时间**的已合并修复 PR;下次更新以新旧 snapshot 的
HEAD diff 为增量范围,新增重点档用 `make_packs.py` 补卡(自动跳过已有卡)。

## 1. 收录范围(全量)

GitHub 已合并 PR,以下三条查询的**并集**(按 PR 号去重,labels 取并集,
`sources` 字段记录来源):

1. `repo:<org>/<repo> is:pr is:merged label:bug`
2. `repo:<org>/<repo> is:pr is:merged fix in:title`
3. `repo:<org>/<repo> is:pr is:merged bug in:title`

Search API 单查询上限 1000 条 → 每条查询按季度窗口(`merged:A..B`)切片,
**超过 1000 的季度自动按月拆分**(2026 年以来单季 bug 修复量 vLLM 已达 1200+)。
已合并 PR 的 `closed_at` 即合并时间。正文保留前 30k 字符。

**已知盲区**:

- 标题不含 fix/bug 且无 `bug` label 的修复(如 `[Kernel] correct xxx` 这种措辞)
  可能漏收 —— 后续可人工补录;
- 修复描述含糊、正文没有根因分析的 PR,修复卡只能如实写「正文未披露根因」;
- 标题里 `fix` 一词也可能命中 "prefix/suffix" 类措辞 —— 组件分类兜底,
  不删除、只归档。

## 2. 组件分类(13 类,标题正则首中即停)

与 perf-pr-index 同一套分类体系,便于跨库对照:

| 优先级 | 分类 | 标题正则要点 |
| --- | --- | --- |
| 1 | CI / Test-Perf | `^(test\|bench\|ci)`、workflow、lint(永不进重点档) |
| 2 | Speculative Decoding | speculative、eagle、mtp、medusa、ngram、draft |
| 3 | MoE | moe、deepep、expert、all-to-all、grouped gemm |
| 4 | Quantization | quant、fp8/fp4/int8/int4、awq、gptq、marlin、modelopt |
| 5 | CUDA Graph / Compiler | cuda graph、piecewise、torch.compile、inductor、jit |
| 6 | Overlap / Scheduler | overlap、schedul、two-batch、tbo、chunked prefill |
| 7 | Attention / Kernel | attention、flashinfer、mla、nsa、triton、kernel、gemm、rope |
| 8 | KV Cache / Memory | kv cache、kv connector、paged、radix、prefix cach、offload |
| 9 | Parallelism / Communication | tensor/pipeline/data parallel、allreduce、nccl、shard |
| 10 | Sampler / Detokenization | sampler、sampling、detokeniz、logits、penalt |
| 11 | Frontend / Serving | api server、openai、tokenizer、http、grpc、metrics、serving |
| 12 | Model-specific Fast Path | deepseek/qwen/llama/glm/gpt-oss/mamba… 等模型族名 |
| 13 | Other | 以上皆未命中 |

## 3. bug 类型标注(标题首中即停)

| 类型 | 标题正则要点 |
| --- | --- |
| correctness 正确性 | incorrect、wrong (result\|output)、mismatch、accuracy、silent、corrupt、nan、precision、diverge、off-by |
| crash 崩溃 | crash、segfault、abort、core dump、assert fail、device-side assert |
| hang 挂死 | hang、deadlock、stuck、freeze、infinite loop、unresponsive |
| resource 资源泄漏/OOM | oom、out of memory、memory leak、leak、unbounded、vram |
| regression 回归 | regression、revert、broken (since\|in\|after) |
| startup 启动/加载 | startup、launch fail、failed to (start\|load\|init)、weight loading |
| general 一般 | 以上皆未命中 |

类型只是标题措辞的粗标(同一 PR 可能既是 crash 又是正确性问题,只记第一个命中的);
准确性质以 PR 正文为准。

## 4. 分级(重点 🔥 / 一般)

满足**任一**即重点档(`CI / Test-Perf` 永远不进):

1. **correctness 或 regression** 类型 —— 正确性与回归是最高价值的修复知识;
2. **crash/hang/resource** 类型 且(标题命中核心组件词
   `kernel|attention|flashinfer|moe|deepep|quant|cuda graph|schedul|kv cache|
   speculative|eagle|mtp|parallel|allreduce|nccl|tokenizer|detokeniz|prefix cach|
   radix|logits|sampling|paged` 或 reactions ≥ 3 或 comments ≥ 10);
3. **高热度**:reactions ≥ 5 或 comments ≥ 25;
4. **高价值 label**:`release-highlight` / `critical` / `security`。

## 5. 修复卡(重点档,diff 级)

每张卡(`index/<repo>/fix-cards/pr-<N>-<slug>.md`)按 `docs/fix-card-template.md`:

元信息表 → 一句话结论 → 问题现象 → 根因(只能来自正文,diff 推断须标注)→
修复方案(≤15 行 diff 摘录)→ 影响与回归风险 → 关联 → 免责尾注。

生成流程:`make_packs.py` 为每个重点档拉 diff(patch-diff.githubusercontent.com,
不占 API 配额;超 40k 按文件粒度裁剪,源码文件优先于测试/文档文件)+ 正文 ≤20k
打包 → 按严重度打分排序(correctness 30 > crash/hang 20 > regression 20 >
resource 15)> 生成 agent 清单 → 子代理(并发 ≤3)按模板撰写。
`make_packs.py` 幂等,重跑自动跳过已有卡片的 PR。

## 6. 限制与诚实声明

- 分类/类型/分级均为启发式,规则可在 `classify_bugfix.py` 中调整后重跑
  `classify_bugfix.py && build_bugfix_index.py` 全量修正。
- 修复卡由 LLM 生成,未经人工核实;影响版本窗口正文不写就不写,不做推测。
- GitHub Search API 不保证 100% 覆盖(偶发索引延迟/限流);抓取脚本对
  got != total 的窗口会打印告警,并对 422 次级限流做长退避重试。
