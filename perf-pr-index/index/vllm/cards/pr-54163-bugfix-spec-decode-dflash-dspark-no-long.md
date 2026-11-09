# PR #54163: [Bugfix][Spec Decode] DFlash/DSpark no longer drop the last prefix-cache block (mamba align context recompute)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Ledgero |
| 合并 | 2026-10-23 |
| 分类 | Speculative Decoding(tier: major) |
| Label | bug, ready, dflash, scheduler |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数) |
| 链接 | https://github.com/vllm-project/vllm/pull/54163 |

## 一句话结论

修正 DFlash/DSpark 被误判为需要 EAGLE 式末块丢弃的问题,恢复 mamba align 模式下可复用的前缀 checkpoint,避免多轮对话全量重算。

## 动机

Fixes #53477。`SpeculativeConfig.use_eagle()` 是宽泛谓词,同时匹配 eagle/eagle3/mtp/dflash/dspark;#53388 从它派生了 `use_eagle_block_drop()`。但 EAGLE 系 drafter 构造某位置的 draft KV 需要 target hidden state 加下一个 token,而 DFlash/DSpark 只用 target hidden state——只有前者需要丢弃末尾 prefix-cache block。误分类导致 DFlash/DSpark + `--mamba-cache-mode align` 时,prompt 短于两个 Mamba block 的请求走 spurious back-off:单一 chunk 结尾不在 block 对齐网格上,Mamba align 无法物化可复用 recurrent state,`find_longest_cache_hit()` 收敛到 0,下一轮全量重算。

## 优化点

新增精确谓词 `needs_eagle_block_drop()`:`use_eagle() and not (use_dflash() or use_dspark())`,即仅 EAGLE/EAGLE3/MTP 为真;`use_eagle_block_drop()` 改为由它加 `disable_eagle_block_drop` 组成。调度器复用该谓词:split back-off、`KVCacheManager(use_eagle=...)`、offloading、Mooncake 等路径对 DFlash/DSpark 收到 `False`,而 `use_eagle()` 本身(encoder shift、lookahead budgeting 等)保持不变;EAGLE 系的既有默认行为与 opt-out 警告逻辑不动。改动范围限于 prefix-cache 末块策略。

## 关键改动

```text
vllm/config/speculative.py: 新增 needs_eagle_block_drop();use_eagle_block_drop() 改用精确谓词
vllm/v1/core/sched/scheduler.py: 警告条件改用 needs_eagle_block_drop() + disable opt-out
tests/test_config.py: 对 eagle3/mtp(drop)与 dflash/dspark(no drop)parametrize
```

## 性能声明

正确性/缓存命中数据(2x L40S,vLLM 0.26.0 布局,Qwen3.5-4B hybrid target + Qwen3.5-4B-DFlash draft,三轮共享 30x system prefix 的对话):DFlash + mamba align 的 cached token 从 `[0, 0, 0]`(每轮全量重算)变为 `[0, 560, 560]`;DFlash + mamba all 同样从 `[0,0,0]` 变 `[0,560,560]`;no spec 与 MTP + align 无回归。

独立验证:vedcsolution(4x DGX Spark GB10, TP4,GLM-5.3-Flash + DFlash2)——相同请求 prefix hit 从 `0/12,552` 到 `4,608`,TTFT 从 3.9s 降到 1.55s,另修复一处 KDA over-truncation(69 场景 tool-call suite 89 → 91)。dtandersen(Intel Arc B70,Qwen3.8-27B-INT4 + DFlash2)确认 prefix caching 下的 token corruption 消失。

接受率对照(同模型/seed/temperature=0/max_tokens=128,热身 prefix cache,V2 runner):DFlash 32 prompts 基线 0.14804/3.36859 → 0.15199/3.43189;DSpark 128 prompts 0.34399/3.40794 → 0.34512/3.41582(基本持平)。

## 验证与回归风险

`tests/test_config.py` 四方法 × opt-out 开关的谓词测试(8 passed);`tests/v1/core/test_mamba_align_chunk_split.py` 46 passed;`test_scheduler.py -k mamba_align` 2 passed;ruff/git diff --check 干净;#54041 被作为本 PR 的重复项关闭。风险很小:谓词语义变化影响所有 DFlash/DSpark 用户的缓存策略(行为即修复目标);EAGLE/EAGLE3/MTP 路径无行为变化;`use_eagle_block_drop()` 的返回值变化可能影响依赖它做其他判断的代码(检索显示仅调度器 warning 使用)。

## 关联

修复 #53477;前序 #53388(EAGLE 末块丢弃 opt-out)、#30877(Mamba align prefix caching)、#42971、#44082;#54041 为重复实现。同分类可参考 DFlash 主卡(pr-36847)与同 batch 的 pr-51725、pr-56323。
