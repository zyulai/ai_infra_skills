# PR #54625: [Core] Add cache-aware admission ordering

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fululi12 |
| 合并 | 2026-11-04 |
| 分类 | Speculative Decoding |
| Label | ready, scheduler, kv-cache-manager |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计,diff 内容有保留) |
| 链接 | https://github.com/vllm-project/vllm/pull/54625 |

## 一句话结论

调度器在 waiting 队列有界窗口内把前缀已驻留 KV cache 的请求提前准入,避免冷请求驱逐可复用 block,高并发下 p90 交互性 +2~5%。

## 动机

严格 FCFS 准入与 prefix caching 存在负交互:排在前面的冷请求先分配 block,压力下会驱逐队列后方请求仍需要的前缀 block,把本可命中的 cache hit 变成重算;共享前缀越深损失越大。需要一个有界 look-ahead,让"驻留请求"先被准入,冷请求的分配就不再破坏即将兑现的复用。

## 优化点

新增两个默认关闭的调度参数:`--cache-aware-admission-window N`(look-ahead 深度,0=严格到达序)与 `--cache-aware-admission-threshold F`(KV cache 使用率低于该值不启用;diff 中默认 0.5,即 cache 过半使用才参与重排)。`schedule()` 在非 defer_prefills 步骤调用 `_reorder_waiting_by_cached_prefix`:取窗口内前 N 个请求,只对"纯 WAITING 且 num_computed_tokens==0"的槽位按缓存前缀深度排序(并列保持到达序),blocked 状态(如 `WAITING_FOR_REMOTE_KVS`)与已部分计算的请求位置固定、任何请求不会越过它们;实现为一次窗口摘除 + 逆序 `prepend_request` 重插(避免逐个删除的二次复杂度)。评分用新增的 `KVCacheManager.get_num_cached_tokens`:只探测 cache,不取引用、不分配 block、不发 `BlockStored` 事件(`get_computed_blocks` 在 `kv_cache_report_mode="full"` 下会对仅打分的请求发布幽灵事件)。构造时要求 fcfs policy 与 prefix caching,否则窗口强制为 0;Mamba 等非全注意力 spec 下探测自然返回 0、退化为 no-op。

## 关键改动

```text
vllm/v1/core/sched/scheduler.py: _reorder_waiting_by_cached_prefix(窗口内重排)+ 构造门槛
vllm/v1/core/kv_cache_manager.py: 新增无副作用探测 get_num_cached_tokens
vllm/config/scheduler.py, vllm/engine/arg_utils.py: 两个新 CLI 参数
tests/v1/core/test_scheduler.py: 新增 7 个准入重排测试
```

## 性能声明

Kimi-K3 FP4、8x MI355X、DCP=8、replay agentic-coding trace。window=16 单独(offload 配置,并发 52):p90 interactivity 12.56→12.80(+1.96%),tok/s/GPU 7,983→8,339(+4.47%),prefix hit 90.7%→91.5%;offload OFF 的 nightly 复测(并发 48):p90 +4.95% 而 throughput -1.01%。窗口过大有害:同配置 window=64 使并发 44 损失 12.9% throughput、8.6% p90。与 #54627 的 prefill_schedule_interval 组合(非本 PR 单独效果):并发 44/48 p90 交互性 +44.7%/+44.3%,并发 1/14 仅 +0.3~0.4%。对照两次运行噪声:throughput 0.11%、p90 0.19%。无 accuracy eval(仅改变准入顺序,batch 组成变化,输出非 bit-identical)。

## 验证与回归风险

`tests/v1/core/test_scheduler.py` 新增 7 个测试(warm 优先准入、窗口外不越位、阈值下不启用、blocked/started 位置固定、Mamba spec 安全、priority/无 caching 下禁用等);scheduler 测试套件 253/180 passed。风险面:默认关闭;FCFS 公平性被窗口内打乱(设计上限定为"同辈间重排");正文提到阈值默认值前后表述不一致(Purpose 段写 0.0,后文与 diff 为 0.5);测试环境为 nightly 镜像而非标准 uv 工作流,作者声明合并前重跑。

## 关联

同区域:#53571(SLO-aware 调度,TTFT 目标驱动,与本 PR 信号不同可共存)、#54627(prefill_schedule_interval,组合实验伙伴)、#53772/#54348(skipped_waiting 提升)、#45558(priority preemption)、#54327(KV offload 文件系统驱逐)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
