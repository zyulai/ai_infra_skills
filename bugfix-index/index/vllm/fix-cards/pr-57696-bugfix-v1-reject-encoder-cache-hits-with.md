# PR #57696: [Bugfix][V1] Reject encoder-cache hits with mismatched embedding counts

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57696](https://github.com/vllm-project/vllm/pull/57696) |
| 作者 | jackLei0901 |
| 合并日期 | 2026-09-24 |
| labels | bug, ready, scheduler |
| 热度 | 👍 0 · 💬 9 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
客户端复用多模态 UUID 会让 encoder 缓存条目与新请求的 embedding 数不匹配:缓存更短时 EngineCore 崩死、更长时静默串内容;在缓存准入处校验数量,失配请求按单请求错误收尾。

## 问题现象
客户端提供的多模态 UUID 被意外复用于不同内容时,缓存的 encoder 输出会配给一个期望不同 embedding 数的请求:
- 缓存张量更短:错位到达 embedding merge,EngineCore 退出,该引擎上所有后续请求失败(GPU 复现中失配请求与后续健康请求均报 `EngineDeadError`)。
- 缓存张量更长:切片可能成功,静默地用不同内容的 embedding 服务——既是可用性又是静默正确性问题。
- 旧的释放路径按发起释放的请求重推缓存大小,同一标识符配不同请求尺寸时会破坏 freeable 容量记账,引发后续分配断言。

GPU 复现:SmolVLM、`max_num_seqs=1`、禁用 P0 处理器缓存,短图(320 image tokens)先建缓存,长图(1088)复用同一标识符。

## 根因
正文披露:客户端 UUID 应唯一标识其数据,但该契约由客户端负责,意外复用会把缓存 encoder 输出与期望不同 embedding 数的请求配对;引擎在缓存命中准入处不校验 embedding 数量,错位直达 embedding merge。前置的 UUID 校验文档([Multimodal Media UUID Security](https://github.com/vllm-project/vllm/blob/main/docs/usage/security.md#multimodal-media-uuid-security))已把唯一性责任归于客户端,本 PR 不改变该契约,只封堵可用性缺口。

## 修复方案
- `vllm/v1/core/encoder_cache_manager.py`:新增 `num_embeds_by_hash`(identifier → 已存 encoder embedding 数),分配时 `setdefault` 记录、驱逐/`reset` 时移除;`free_encoder_input` 改用存储值而非从请求重推;新增 `get_cached_num_encoder_embeds`。
- `vllm/v1/core/sched/scheduler.py`:新增 `encoder_cache_mismatch_reqs` 与 `_reject_on_encoder_cache_embed_mismatch`(打 warning,含标识符、缓存/请求计数与 UUID 复用提示);`_try_schedule_encoder_inputs` 在缓存命中准入改动引用状态前比较计数,并对同一步内重复标识符比较且不重复计预算;`update_from_output` 把失配请求并入既有 error drain,以 `FINISHED_ERROR` 收尾。设计上拒绝请求而非驱逐条目,防止客户端刷掉共享条目。
- 已知边界(正文声明):EC connector 首次远端命中不在校验内;同数不同内容的复用仍超出范围。

## 影响与回归风险
- 影响窗口:正文未写明版本范围。
- 回归面:改动全部在调度器与 `EncoderCacheManager`,对两个 model runner 都生效;刻意行为变化——「缓存更长」场景从表面成功(实际服务错内容)变为显式请求错误(`FINISHED_ERROR` 映射 HTTP 500,复用既有 grammar 错误 drain,未新增开关)。校验只在准入 identifier 命中时运行,不改变满足契约请求的输出。验证:encoder-cache-manager 21 passed、scheduler 相关 4 passed、5 个相邻 scheduler 测试通过;GPU 对照(基线 vs 修复)确认失配请求 `finish_reason="error"`、健康请求正常、进程状态 0。

## 关联
- #55546(engine-containment 部分,不因此关闭)
- #55547(related)
- #57833(related:`MultiModalReceiverCache` 偏好 P0 重发完整 payload)
- #55549、#55952(已关闭未合并的先例)
- #55583、#55616(open,改处理器/渲染与 UUID 派生路径,非本 PR 范围)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
