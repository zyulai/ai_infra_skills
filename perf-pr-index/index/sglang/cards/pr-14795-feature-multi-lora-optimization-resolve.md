# PR #14795: [Feature] Multi lora optimization - resolve scheduler blocking issue and save Non-Lora inference performance

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ConnorLi96 |
| 合并 | 2025-12-13 |
| 分类 | Overlap / Scheduler(major) |
| Label | run-ci |
| Diff | pack 未提供 diffstat(additions/deletions/files 均缺失),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/14795 |

## 一句话结论

修复 LoRA 与非 LoRA 混部时 base model 槽位被逐出、调度器整批阻塞的问题,非 LoRA TTFT 从 12,717ms 回到 73ms。

## 动机

LoRA 与非 LoRA 请求混在同一 batch 时,LoRA adapter cache miss 会严重阻塞非 LoRA 流量,原因有二:一是 LoRA memory pool 的逐出策略把 base model(uid=None)排在最低优先级、最先被逐出,导致非 LoRA 请求必须等待 adapter 重新加载;二是 scheduler 在遇到超出槽位上限的 LoRA 请求时用 `break` 中断整个 waiting queue 遍历,即便后续非 LoRA 请求仍可调度。

## 优化点

两处调度层改动。第一,在 LoRA memory pool 中"钉住"base model 槽位:`get_available_buffer_slot` 构造逐出候选时先剔除 None,只有当 batch 全部是 LoRA 请求、没有任何可逐出 adapter 时才允许逐出 None,保证所有非 LoRA 请求共享的单个 base model 槽位常驻。对应地,`eviction_policy.py` 的 LRU/FIFO `select_victim` 删掉"base model 永远最先逐出"的前置分支,改为仅在没有 tracked UID 候选时才返回 None。第二,scheduler 的 `get_new_batch_prefill` 中,当 LoRA 请求会超出槽位上限时,由 `break` 改为 `continue` 跳过该请求,让后续非 LoRA 请求继续被调度,而不是阻塞整个队列;非 LoRA 请求(lora_id=None)共享单一保留槽位,不会触发该检查失败。

## 关键改动

```text
python/sglang/srt/lora/mem_pool.py: 逐出候选剔除 base model(None),LoRA 优先逐出
python/sglang/srt/lora/eviction_policy.py: LRU/FIFO 移除 base-model-first 分支,
  仅在无 tracked UID 时逐出 None
python/sglang/srt/managers/scheduler.py: 超限 LoRA 请求 continue 跳过,
  不再 break 阻塞整个 waiting queue
```

## 性能声明

PR 使用 `sglang.bench_serving`(random 数据集,100 prompts,request-rate 2,input 2048/output 1024),对比组为并发运行的 B3(纯非 LoRA 流量)与 B4(16 个 LoRA adapter 流量),硬件未注明:

- 修复前:非 LoRA TTFT 29ms -> 12,717ms(438x slower),非 LoRA E2E 4,829ms -> 19,469ms(4x slower)。
- 修复后:非 LoRA TTFT 29ms -> 73ms(2.5x slower),非 LoRA E2E 4,829ms -> 7,472ms(1.5x slower)。
- LoRA 流量(16 adapters):Mean TTFT 8,666ms -> 3,802ms(2.3x faster,median 306ms),Mean E2E 15,157ms -> 10,897ms(1.4x faster),Max ITL 953ms -> 327ms(2.9x faster)。

## 验证与回归风险

PR 正文未附 accuracy 测试结果(Checklist 中该项未勾选),也未新增单测。风险集中在 LoRA 槽位管理路径:逐出优先级反转后,纯 LoRA batch 仍可能逐出 base model 槽位,若 `can_run_lora_batch` 判定与 `get_available_buffer_slot` 的候选逻辑不一致,可能反复触发 adapter 加载;`continue` 改动改变了 waiting queue 的推进语义,batch_is_full 标记不再被置位,极端情况下可能增加每轮扫描开销。仅影响 `--enable-lora` 路径,非 LoRA 推理不受影响。

## 关联

前序引用:PR 正文指出问题代码位于 scheduler.py `get_new_batch_prefill` 的 `can_run_lora_batch` 检查(旧实现 `break`)。同分类其他 LoRA 相关 PR 编号在素材中未给出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
