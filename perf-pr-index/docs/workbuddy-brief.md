# Workbuddy 任务书:vLLM 证据卡 split-1(366 张)

> 把本文件全文(或下方"粘贴用提示词")发给 workbuddy 即可开工。
> 与主会话(split-0)通过**固定切分清单**分工,不会撞车。

## 背景

`C:\Users\Norma\Desktop\ai_infra\perf-pr-index\` 是 vLLM/SGLang 性能优化 PR 索引库。
每个重点档(fire)PR 需要一张"证据卡"。当前进度:

- ✅ SGLang 645/645(**已完成,不要动**)
- 🔄 vLLM 936 张:主会话负责 `split-0.json`(366 张,进行中);**workbuddy 负责 `split-1.json`(366 张)**

每个 PR 的素材包已就绪:`data/cards-workdir/vllm/packs/pr-<N>.pack.md`
(含 PR 元信息、正文、diff)。卡片模板与硬性规则:`docs/card-template.md`。

## 你的工作循环(重复直到 split-1 清零)

工作目录:`C:\Users\Norma\Desktop\ai_infra\perf-pr-index\`

1. 生成下一批工作清单(每次 51 张,自动跳过已有卡片的 PR):

   ```bash
   python scripts/next_batch.py --repo vllm --from-split 1 --batch 51 --agents 3 --worker workbuddy
   ```

   输出 `data/cards-workdir/vllm/agent-batches-workbuddy/batch-XXX-agent-{1,2,3}.md`,
   每行一张卡:`PR# | 标题 | pack路径 | out路径`。若报告 taking 0 则 split-1 完成。

2. 并行派发 **最多 3 个**子代理写卡(并发上限 3,超了会报
   "user concurrency limit exceeded",遇到就减并发重试)。子代理提示词模板
   (N=清单里的卡数,K=agent 编号,XXX=batch 号):

   ```text
   你要为 vLLM 性能优化 PR 索引库撰写 N 张"证据卡"。根目录:C:\Users\Norma\Desktop\ai_infra\perf-pr-index\
   步骤:
   1. Read docs/card-template.md(模板与硬性规则,严格遵守)
   2. Read data/cards-workdir/vllm/agent-batches-workbuddy/batch-XXX-agent-K.md,
      每行一张卡:PR# | 标题 | pack路径 | out路径(只处理这个文件,不要读其他 batch 文件)
   3. 逐卡 Read pack 素材,按模板 Write 卡片到 out 路径(文件名不要改;标题栏用 pack
      里的真实标题;diff 不可用或 diffstat 缺失时如实注明;若 out 文件已存在则跳过)
   完成后报告:写出文件数(列 PR 号)、跳过的已存在文件、缺 diff 的卡号。不要写其他文件。
   ```

3. 子代理返回后核对:清单里每个 PR 号在 `index/vllm/cards/` 都有 `pr-<N>-*.md`;
   缺的补派一个子代理处理对应清单(模板里已有"已存在则跳过"规则)。

4. 每完成 1-2 批重建一次索引(原子写,可与主会话并发):

   ```bash
   python scripts/build_index.py --repo vllm
   ```

5. 回到第 1 步。全部完成后(或必须停止时):运行
   `python scripts/build_index.py --repo vllm`,统计 `index/vllm/cards/` 文件数,
   向用户报告 split-1 完成数。

## 硬性规则

- **只处理 split-1 的 PR**;SGLang 和 split-0 归主会话,不要写。
- 不要修改 `scripts/`、`data/`(卡片和索引除外)、`docs/` 下任何文件。
- 卡片质量红线(来自 card-template.md):只写素材里有的信息、数字照抄 PR 原文、
  摘录 ≤15 行、300-800 字、文件名以 `pr-<N>-` 开头。
- 任何时候中断都安全:重跑第 1 步会自动从未完成的 PR 继续。
