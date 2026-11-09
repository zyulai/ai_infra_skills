# Bugfix Index — vLLM / SGLang Bug Fix 全量索引

对 [vllm-project/vllm](https://github.com/vllm-project/vllm) 与
[sgl-project/sglang](https://github.com/sgl-project/sglang) 的**已合并 bug 修复 PR**
做全量收录、组件分类、bug 类型标注(正确性/崩溃/挂死/资源泄漏/回归…)与分级
(重点 🔥 / 一般),并为重点档生成 diff 级修复卡(问题现象 / 根因 / 修复方案 / 回归风险)。

与姊妹库配套使用:

- [perf-pr-index](../perf-pr-index/) — 已合并性能优化 PR(优化了什么)
- [rfc-index](../rfc-index/) — RFC 设计提案(打算怎么改)
- **bugfix-index(本库)** — 已合并 bug 修复(哪里坏过、怎么修的)

## 快速开始

```bash
# 查询(需要先跑完 fetch + classify)
python scripts/query.py --list                                    # 分类/类型统计
python scripts/query.py "cuda graph"                              # 关键词检索
python scripts/query.py --repo vllm --tier major --type crash     # 重点档崩溃类
python scripts/query.py --type correctness --since 2026-01-01     # 近期正确性修复
python scripts/query.py --card 12345 --repo sglang                # 查看修复卡

# 直接浏览
index/vllm/BUGFIX-INDEX.md      # vLLM 全量索引(按分类分表,双语)
index/sglang/BUGFIX-INDEX.md    # SGLang 全量索引
index/STATS.md                  # 统计:分类/类型/年度/热度/Top 作者
index/<repo>/fix-cards/         # 重点档修复卡
```

## 收录范围

GitHub 已合并 PR,三条查询的并集(去重、label 取并):

1. `is:pr is:merged label:bug`
2. `is:pr is:merged fix in:title`
3. `is:pr is:merged bug in:title`

Search API 单查询上限 1000 → 全部按季度时间窗(`merged:A..B`)切片,
超 1000 的季度自动按月拆分。正文保留前 30k 字符。

## 分级规则(重点 🔥,`CI / Test-Perf` 类除外)

满足任一即重点档:

1. **正确性 bug**:标题命中 incorrect / wrong output / NaN / mismatch / accuracy /
   silent corruption 等 —— 永远重点;
2. **回归修复**:标题含 regression / revert / broken since;
3. **崩溃/挂死/资源类** 且(标题命中核心组件词 或 reactions ≥ 3 或 comments ≥ 10);
4. **高热度**:reactions ≥ 5 或 comments ≥ 25;
5. **高价值 label**:`release-highlight` / `critical` / `security`。

## 目录结构

```
bugfix-index/
├── README.md
├── data/
│   ├── raw/                    # fetch_bugfixs.py 原始数据(jsonl,body ≤30k)
│   ├── classified/             # classify_bugfix.py 分类结果(全量)
│   ├── major/                  # 🔥 重点档
│   └── cards-workdir/          # 修复卡素材包(diff+正文)与 agent 批次清单
├── index/
│   ├── vllm/BUGFIX-INDEX.md    # vLLM 全量索引
│   ├── vllm/fix-cards/         # vLLM 重点档修复卡
│   ├── sglang/BUGFIX-INDEX.md
│   ├── sglang/fix-cards/
│   └── STATS.md
├── scripts/
│   ├── fetch_bugfixs.py        # 抓取(3 查询 × 季度/月度切片,断点续传)
│   ├── classify_bugfix.py      # 组件分类 + bug 类型 + 重点档分级
│   ├── make_packs.py           # 重点档 pack(diff ≤40k)+ agent 批次清单
│   ├── build_bugfix_index.py   # 生成 BUGFIX-INDEX.md / STATS.md
│   ├── query.py                # 查询工具
│   └── snapshot.py             # 记录上游 HEAD 到 data/snapshot.json
└── docs/
    ├── methodology.md          # 收录/分类/分级标准明细
    └── fix-card-template.md    # 修复卡模板与硬性规则
```

## 更新流程

```bash
export GITHUB_TOKEN=<your_token>            # 只走环境变量,不落盘
python scripts/fetch_bugfixs.py             # 1) 增量抓取(按 PR 号合并,可反复跑)
python scripts/classify_bugfix.py           # 2) 重新分类分级(全量重算)
python scripts/snapshot.py                  # 3) 记录新的上游 HEAD 到 data/snapshot.json
python scripts/build_bugfix_index.py        # 4) 重建索引(快照自动写入所有生成 md)
python scripts/make_packs.py --repo both    # 5) 新增重点档生成批次清单 → 子代理写卡
```

**数据快照基线**:`data/snapshot.json` 记录本次数据的上游 commit id、API 抓取窗口
与各层数量,并渲染在每个生成 md 的「数据快照」一节。下次更新前后各跑一次
`snapshot.py`,对比即知增量范围;新增重点档用 `make_packs.py` 补写修复卡(自动跳过已有卡)。

## 数据说明与免责

- 分类与 bug 类型是标题关键词规则,可能有误判(`docs/methodology.md` 说明盲区);
  bug 类型只反映标题措辞,准确性质请以 PR 正文为准。
- 修复卡由 LLM 基于 PR 正文与 diff 生成:根因只能来自正文(从 diff 推断必须标注),
  版本号/复现条件照抄原文,落地与回归影响不做猜测。
- 已合并 PR 的 `closed_at` 即 `merged_at`;issue 编号与评论详情请点原帖。
