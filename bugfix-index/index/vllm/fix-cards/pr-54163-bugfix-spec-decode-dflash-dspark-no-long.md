# PR #54163: [Bugfix][Spec Decode] DFlash/DSpark no longer drop the last prefix-cache block (mamba align context recompute)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54163](https://github.com/vllm-project/vllm/pull/54163) |
| 作者 | Ledgero |
| 合并日期 | 2026-10-23 |
| labels | bug, ready, dflash, scheduler |
| 热度 | 👍 4 · 💬 26 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
use_eagle() 宽泛匹配令 DFlash/DSpark 也丢最后一个 prefix-cache block,mamba align 全量重算;新增 needs_eagle_block_drop() 精确区分。

## 问题现象
DFlash/DSpark + prefix caching:最后一个 prefix-cache block 因误判被丢弃,`--mamba-cache-mode align` 下可复用的 Mamba checkpoint 失效,后续轮次全量重算(Fixes #53477)。本地复现(2x L40S,Qwen3.5-4B hybrid target + DFlash draft,三轮共享 30x system prefix):DFlash+align 的 cached=[0,0,0] → [0,560,560]。独立验证:Intel Arc B70 上前缀缓存 token corruption 解决;4x DGX Spark GB10 上同请求 prefix hits 0/12,552 → 4,608、TTFT 3.9s → 1.55s。EAGLE/EAGLE3/MTP 与无 spec 路径无回归。

## 根因
正文披露:`SpeculativeConfig.use_eagle()` 是宽泛谓词,同时匹配 eagle/eagle3/mtp/dflash/dspark;#53388 从它派生了 `use_eagle_block_drop()`。EAGLE 式 drafter 构造某位置的 draft KV 需要 target hidden state 与 next token,而 DFlash/DSpark 仅用 hidden state、保留 target KV cache,因此只有 EAGLE/EAGLE3/MTP 需要 trailing block drop。对短于两个 Mamba block 的 prompt,错误的 back-off 造成:last_cache_position==0 的单 chunk 结束在 Mamba block 网格外 → align 模式只在 block 对齐的 chunk 末尾物化可复用状态 → `find_longest_cache_hit()` 把下一轮命中收敛到 0 → 全量重算。

## 修复方案
`vllm/config/speculative.py` 新增 `needs_eagle_block_drop()`(use_eagle 且非 dflash/dspark),`use_eagle_block_drop()` 改由它组合 `disable_eagle_block_drop`;`scheduler.py` 的警告条件改用新谓词。`use_eagle()` 本身保持不变(encoder shift、lookahead budgeting 等路径不受影响):

```diff
+    def needs_eagle_block_drop(self) -> bool:
+        return self.use_eagle() and not (self.use_dflash() or self.use_dspark())
...
-        return self.use_eagle() and not self.disable_eagle_block_drop
+        return self.needs_eagle_block_drop() and not self.disable_eagle_block_drop
```

## 影响与回归风险
- 影响窗口:正文说明自 #53388 从宽泛谓词派生 block drop 策略起,DFlash/DSpark 被误分类。
- 回归面:前缀缓存 trailing-block 策略:DFlash/DSpark 不再 drop 最后一个块;EAGLE/EAGLE3/MTP 默认行为不变,`disable_eagle_block_drop` opt-out 与警告仍可用;scheduler 的 Mamba-align 相关警告条件与测试 fixture 同步调整。

## 关联
- Fixes #53477
- #53388(引入 use_eagle_block_drop opt-out)
- #30877(Mamba align prefix caching)
- #42971
- #44082

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
