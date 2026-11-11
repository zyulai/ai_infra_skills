# ai_infra_skills — vLLM / SGLang 知识库合集

围绕 **vLLM** 与 **SGLang** 构建的三套知识库 + 三个上游仓库快照(以 git submodule 形式钉在固定 commit)。
后续所有优化 / 排障 / 技术调研都以这套库为底座。

## 目录结构

```
ai_infra_skills/
├── perf-pr-index/               # ① 已合并性能优化 PR 全量索引 + diff 级证据卡
├── rfc-index/                   # ② RFC 设计提案全量索引 + 摘要卡
├── bugfix-index/                # ③ 已合并 bug 修复 PR 全量索引 + 修复卡
├── vllm/                        # [submodule] vllm-project/vllm
├── sglang/                      # [submodule] sgl-project/sglang
└── AI-Infra-Auto-Driven-SKILLS/ # [submodule] BBuf 的 agent skills(参考)
```

## 克隆(含子仓库)

```bash
# Windows 用户先执行(sglang 上游存在超过 MAX_PATH 的文件名):
git config --global core.longpaths true

git clone --recurse-submodules git@github.com:zyulai/ai_infra_skills.git
# 已克隆过:
git submodule update --init --recursive
```

## 三个知识库(数据规模与基线)

| 知识库 | 收录范围 | vLLM | SGLang | 重点档卡片 | 上游基线 |
| --- | --- | --- | --- | --- | --- |
| perf-pr-index | 已合并性能优化 PR(label:performance ∪ 标题关键词) | 6972 | 6219 | 1633 证据卡 | vllm `b32a87094afc` / sglang `56e38dfe0fdb` |
| rfc-index | RFC 提案(label:rfc ∪ 标题含 RFC) | 937 | 204 | 267 摘要卡 | vllm `b32a87094afc` / sglang `95999790a4a2` |
| bugfix-index | 已合并 bug 修复 PR(label:bug ∪ 标题含 fix/bug) | 7975 | 6411 | 1257 修复卡 | vllm `b32a87094afc` / sglang `56e38dfe0fdb` |

- **perf-pr-index**:按 13 类组件分类、重点/一般分级;每张证据卡含动机、优化点、关键改动(diff 摘录)、性能声明(数字照抄 + 测试条件)、验证与回归风险。
- **rfc-index**:15 类主题、开放/已关闭状态;每张摘要卡含提案动机、方案、优化点与收益、现状落地。
- **bugfix-index**:13 类组件 + bug 类型(正确性/崩溃/挂死/资源泄漏/回归);每张修复卡含问题现象、根因(只来自正文,diff 推断必标注)、修复方案、影响与回归风险。

## 快速上手

```bash
# ① 性能优化怎么做的
cd perf-pr-index
python scripts/query.py "cuda graph" --framework vllm --tier major
python scripts/query.py --card 11061 --repo sglang          # 看某张证据卡

# ② 某个方向的设计讨论有哪些
cd ../rfc-index
python scripts/query.py --list                              # 分类统计

# ③ 某类 bug 历史上怎么修的
cd ../bugfix-index
python scripts/query.py --type correctness --since 2026-01-01
python scripts/query.py "oom" --tier major
python scripts/query.py --card 14237 --repo vllm
```

或者直接读索引(双语分类表,重点档 🔥 标注,卡片挂链):

- [perf-pr-index/index/vllm/PERF-PRS.md](perf-pr-index/index/vllm/PERF-PRS.md) / [sglang](perf-pr-index/index/sglang/PERF-PRS.md) / [STATS](perf-pr-index/index/STATS.md)
- [rfc-index/index/vllm/RFC-INDEX.md](rfc-index/index/vllm/RFC-INDEX.md) / [sglang](rfc-index/index/sglang/RFC-INDEX.md) / [STATS](rfc-index/index/STATS.md)
- [bugfix-index/index/vllm/BUGFIX-INDEX.md](bugfix-index/index/vllm/BUGFIX-INDEX.md) / [sglang](bugfix-index/index/sglang/BUGFIX-INDEX.md) / [STATS](bugfix-index/index/STATS.md)

各库的收录标准、分类/分级规则与已知盲区见各自 `docs/methodology.md`;更新流程见各自 `README.md`。

## 子仓库基线(submodule 钉住的 commit)

| submodule | 上游 | 基线 commit |
| --- | --- | --- |
| vllm | https://github.com/vllm-project/vllm | `1c7637504238` (2026-11-11) |
| sglang | https://github.com/sgl-project/sglang | `b174f4bb4fe0` (2026-11-11) |
| AI-Infra-Auto-Driven-SKILLS | https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS | `5df356636bf9` (2026-11-09) |

> 知识库的数据基线(抓取时刻的上游 HEAD)记录在各自 `data/snapshot.json`,
> 并渲染在每个生成索引 md 尾部的「数据快照」一节 —— 上表 submodule 基线是克隆
> 时刻的 HEAD,与知识库数据基线略有时间差,以 snapshot.json 为准。

## 更新

```bash
# 更新 submodule(拉上游新代码)
cd vllm && git pull && cd ..
git add vllm && git commit -m "bump vllm"

# 更新某个知识库(以 bugfix-index 为例,见其 README)
export GITHUB_TOKEN=<token>
python scripts/fetch_bugfixs.py && python scripts/classify_bugfix.py
python scripts/snapshot.py && python scripts/build_bugfix_index.py
python scripts/make_packs.py --repo both   # 为新增重点档补卡
```

## 数据说明与免责

- 索引/卡片由脚本 + LLM 流水线生成,数字照抄 PR 原文、根因只来自正文,未经人工核实;
  准确信息以 PR 原帖为准。
- 分类/分级是关键词规则,规则与盲区详见各库 `docs/methodology.md`。
- GitHub API 抓取只走环境变量 `GITHUB_TOKEN`,仓库内容不含任何凭据。
