# RFC Index — vLLM / SGLang RFC(设计提案)全量索引

对 [vllm-project/vllm](https://github.com/vllm-project/vllm) 与
[sgl-project/sglang](https://github.com/sgl-project/sglang) 的**全部 RFC 设计提案**
做全量收录、主题分类、分级(重点 🔥 / 一般),并为重点档 RFC 生成摘要卡
(提案动机 / 方案 / 优化点与收益 / 现状落地)。与姊妹库
[perf-pr-index](../perf-pr-index/)(已合并性能 PR 的 diff 级证据卡)互补:
那边看「已落地的优化」,这边看「设计提案与方向」。

## 收录范围

GitHub issues,两条查询的并集(去重、label 取并):

1. `is:issue label:rfc`(两仓库均有 RFC label)
2. `is:issue "RFC" in:title`

**本次抓取规模**:

| 仓库 | RFC 总数 | 🔥 重点 | 开放 | 已关闭 | 摘要卡 |
| --- | --- | --- | --- | --- | --- |
| vLLM | 917 | 222 | 371 | 546 | 222 |
| SGLang | 202 | 42 | 68 | 134 | 42 |

> 注:`closed` 只代表 issue 关闭(多数已被实现/否决/搁置),卡片「现状与落地」
> 一节按正文可见信息如实描述,不猜测落地结果。

## 快速浏览

```bash
index/vllm/RFC-INDEX.md     # vLLM 全量索引(按 15 个主题分类分表,🔥 标注 + 卡片链接)
index/sglang/RFC-INDEX.md   # SGLang 全量索引
index/STATS.md              # 统计:分类/年度分布、热度 Top、Top 作者
index/<repo>/rfc-cards/     # 重点档摘要卡(每 RFC 一个 md)
```

## 分级规则(重点 🔥)

满足任一即重点档:

1. **高热度**:reactions ≥ 10 或 comments ≥ 30;
2. **性能信号 + 参与度**:标题/正文有性能信号(perf/speedup/fused/overlap…或正文
   带数字的加速声明)且 reactions ≥ 5 或 comments ≥ 10;
3. **强主题词**:标题命中 `fused/kernel/cuda graph/deepep/flashinfer/mla/nsa/
   disaggregat/speculative/eagle/mtp/allreduce/zero-busy/two-batch/pd disagg` 之一。

## 目录结构

```
rfc-index/
├── README.md
├── data/
│   ├── raw/                    # fetch_rfcs.py 原始数据(jsonl,body ≤30k)
│   ├── classified/             # classify_rfcs.py 分类结果(全量)
│   ├── major/                  # 🔥 重点档
│   └── cards-workdir/          # 摘要卡素材包与 agent 批次清单
├── index/
│   ├── vllm/RFC-INDEX.md       # vLLM 全量索引
│   ├── vllm/rfc-cards/         # vLLM 重点档摘要卡
│   ├── sglang/RFC-INDEX.md
│   ├── sglang/rfc-cards/
│   └── STATS.md
├── scripts/
│   ├── fetch_rfcs.py           # 抓取(label ∪ 标题,断点续传)
│   ├── classify_rfcs.py        # 15 类主题分类 + 重点档分级
│   ├── make_batches.py         # 重点档 pack + agent 批次清单
│   ├── build_rfc_index.py      # 生成 RFC-INDEX.md / STATS.md
│   └── snapshot.py             # 记录上游 HEAD 到 data/snapshot.json
└── docs/
    ├── methodology.md          # 分类/分级规则明细
    └── rfc-card-template.md    # 摘要卡模板与硬性规则
```

## 更新流程

```bash
export GITHUB_TOKEN=<your_token>            # 只走环境变量,不落盘
python scripts/fetch_rfcs.py                # 1) 增量抓取(按编号合并,可反复跑)
python scripts/classify_rfcs.py             # 2) 重新分类分级(全量重算)
python scripts/snapshot.py                  # 3) 记录新的上游 HEAD 到 data/snapshot.json
python scripts/build_rfc_index.py           # 4) 重建索引(快照自动写入所有生成 md)
python scripts/make_batches.py --repo both  # 5) 新增重点档生成批次清单 → 子代理写卡
```

**数据快照基线**:`data/snapshot.json` 记录本次数据的上游 commit id
(vLLM `1ecc9cc4ce66`、SGLang `ba5906471c62`,2026-11-07)、API 抓取窗口与各层数量,
并渲染在每个生成 md 的「数据快照」一节。下次更新前后各跑一次 `snapshot.py`,
对比即知增量范围;新增重点档 RFC 用 `make_batches.py` 补写摘要卡(已存在卡片自动跳过)。

## 数据说明与免责

- 分类与分级是关键词规则,可能有误判(`docs/methodology.md` 说明规则与盲区)。
- 摘要卡由 LLM 基于 RFC 正文生成,**只写正文可见信息,数字照抄原文**,
  落地情况不做猜测;准确信息以原帖为准。
- RFC 正文超过 30k 字符的会截断(卡内已注明);issue 评论未抓取,
  讨论结论请点原帖查看。
