# Perf PR Index — vLLM / SGLang 性能优化 PR 全量索引

对 [vllm-project/vllm](https://github.com/vllm-project/vllm) 和
[sgl-project/sglang](https://github.com/sgl-project/sglang) 两个仓库的**已合并性能优化 PR**
做全量收录、分类、分级(重点 🔥 / 一般),并为重点档 PR 生成 diff 级证据卡。
灵感与目录设计参考 [BBuf/AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS)
的 `model-pr-optimization-history`(他按模型族组织,本库按 PR 全量索引组织)。

## 快速开始

```bash
# 查询(需要先跑完 fetch + classify)
python scripts/query.py --list                                   # 分类统计
python scripts/query.py "cuda graph"                             # 关键词检索
python scripts/query.py --framework vllm "fused moe" --tier major
python scripts/query.py --category "Attention / Kernel" --min-speedup 20
python scripts/query.py --card 11061 --repo sglang               # 查看证据卡

# 直接浏览
index/vllm/PERF-PRS.md        # vLLM 全量索引(按分类分表)
index/sglang/PERF-PRS.md      # SGLang 全量索引
index/STATS.md                # 跨仓库统计(分类/年度/作者/加速声明 Top)
index/<repo>/cards/           # 重点档证据卡(分批生成中)
```

## 目录结构

```
perf-pr-index/
├── README.md
├── data/
│   ├── raw/                    # fetch_prs.py 抓取的原始数据(jsonl)
│   ├── classified/             # classify.py 分类结果(全量)
│   └── major/                  # classify.py 重点档筛选结果
├── index/
│   ├── vllm/PERF-PRS.md        # vLLM 全量索引
│   ├── vllm/cards/             # vLLM 重点档证据卡
│   ├── sglang/PERF-PRS.md      # SGLang 全量索引
│   ├── sglang/cards/           # SGLang 重点档证据卡
│   └── STATS.md                # 统计总览
├── scripts/
│   ├── fetch_prs.py            # GitHub API 抓取(搜索 + 富化)
│   ├── classify.py             # 规则分类 + 重点档筛选
│   ├── build_index.py          # 生成 PERF-PRS.md / STATS.md
│   ├── generate_cards.py       # 证据卡数据组装(辅助)
│   └── query.py                # 查询工具
└── docs/methodology.md         # 收录/分类/分级标准明细
```

## 更新流程

```bash
export GITHUB_TOKEN=<your_token>            # 只走环境变量,不落盘
python scripts/fetch_prs.py --repo both     # 1) 增量抓取(支持断点续传)
python scripts/classify.py                  # 2) 重新分类分级(全量重算)
python scripts/snapshot.py                  # 3) 记录新的上游 HEAD commit id 到 data/snapshot.json
python scripts/build_index.py               # 4) 重建索引(快照自动写入所有生成的 md)
```

**数据快照基线**:`data/snapshot.json` 记录了本次数据的上游 commit id
(vllm `c9e4cb2b793b`、sglang `8920463c90c8`,2026-11-07)、API 抓取窗口与各层数量,
并自动渲染在 `index/*/PERF-PRS.md` 与 `index/STATS.md` 的「数据快照」一节。
下次更新前后各跑一次 `snapshot.py`,对比两次快照即可明确知道增量范围;
重点档新增的 PR 用 `next_batch.py`(或其 `--make-split` 多工人模式)补写证据卡。

证据卡生成(分批流水线):

```bash
python scripts/generate_cards.py --repo sglang   # 1) 为重点档拉 diff 打包(断点续传)
python scripts/next_batch.py --repo sglang       # 2) 选取下一批 51 张未写卡 PR,生成 agent 工作清单
# 3) 按 agent-batches/batch-*-agent-*.md 逐包撰写,输出 index/<repo>/cards/
python scripts/build_index.py                    # 4) 重建索引,卡片自动挂链
```

`next_batch.py` 自动跳过已有卡片/缺失素材的 PR,按重要度打分排序
(加速声明 > release-highlight/performance label > 热度),可反复调用直到清零。

## 数据说明与免责

- 数据快照时间见 `index/STATS.md` 头部;`merged_at` 缺失时用 `closed_at` 代替
  (对已合并 PR 两者时间一致)。
- `speedup_claims` / `speedup_numbers` 由正则从 PR 正文提取,**未经核实**,
  实际收益请以 PR 中的 benchmark 条件(模型/硬件/精度/workload)为准。
- 分类与分级是关键词规则,可能有误判(`docs/methodology.md` 说明了规则与已知盲区)。
