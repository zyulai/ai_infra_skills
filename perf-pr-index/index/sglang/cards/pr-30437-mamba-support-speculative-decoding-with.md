# PR #30437: [Mamba] Support speculative decoding with extra_buffer_lazy

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @YAMY1234 |
| 合并 | 2026-07-21 |
| 分类 | Speculative Decoding(major) |
| Label | run-ci, bypass-fastfail, run-ci-extra |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 覆盖 5 个文件(schedule_batch、batch_result_processor、server_args、spec_utils、测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/30437 |

## 一句话结论

让 mamba 的 `extra_buffer_lazy` 策略支持 speculative decoding,受限 mamba pool 下 checkpoint 缓存复用提升约 7-9x。

## 动机

#27118 引入的 `extra_buffer_lazy` 把每个 running request 的 mamba checkpoint 占用从 2 个 slot 降到 1 个,但当时明确不支持 speculative decoding("Follow-up: Support MTP")。省出的 slot 很关键:mamba pool 是 running requests 与 RadixCache 缓存 checkpoint 共享的,running 少占的 slot 直接变成可复用的缓存容量。

## 优化点

非 spec 场景每步恰好推进 1 token,边界在 prepare 时已知,可直接预分配第二个 slot 并交换写指针;而 MTP 的 verify 一步推进 1..num_draft_tokens 个 token,是否跨边界只有 verify 之后才知道。本 PR 把交换推迟:prepare 阶段先确保 spare slot 存在但不改写指针(`mamba_lazy_spec_prepare`,用 2×max_draft_tokens 窗口吸收 overlap 下 `kv_committed_len` 的滞后),把本步写目标记录在 batch 上随 result queue 传递;verify 后 `_mamba_lazy_spec_update` 在 accept 数已知时才 promote(释放旧 checkpoint、重指指针)。另一关键改动:slot 分配失败不再触发 evict 重试(非 spec 边界预分配也一并改掉)——临时 slot 不值得拿缓存 checkpoint 换,失败时退化为原地写入。请求仍保留 slot 所有权,batch 只携带只读的每轮写计划。仍不支持 DFLASH/DSPARK(其 verify 绕过 `prepare_mamba_track_for_verify`)与 PD disaggregation。

## 关键改动

```text
python/sglang/srt/managers/schedule_batch.py: 新增 mamba_lazy_spec_prepare/mamba_lazy_spec_in_window,预分配不再 evict
python/sglang/srt/managers/scheduler_components/batch_result_processor.py: 新增 _mamba_lazy_spec_update 做 verify 后 promote/finish 捐赠判定
python/sglang/srt/speculative/spec_utils.py: spec_prepare_for_decode 调用 prepare,verify 使用 batch 携带的 track plan
python/sglang/srt/server_args.py: 校验规则从"spec 全禁 lazy"放宽为仅禁 DFLASH/DSPARK 与 PD disagg
test/registered/models_e2e/test_qwen3_next_models_mtp.py: 新增 TestQwen3NextMTPLazyV2 与强制分配失败的 manual 测试
```

## 性能声明

复用基准:48 个独立会话 × 3 轮,turn 2-3 报告 prompt 由缓存服务的比例(`cached_tokens/prompt_tokens`),`max_running_requests=16`,模型为 Qwen3-Next-80B-A3B(NEXTN,steps=3,topk=1,draft=4,TP4)。

- mamba pool 不受限(1024 slots):lazy 与 extra_buffer 均为 0.9677,无差异。
- pool 受限(96 slots):lazy 为 0.106-0.147(3 次运行),extra_buffer 仅 0.0161,约 7-9x;PR 归因于 16 个 running 请求从 pin 48 slots 降到 32,留给缓存的 slot 从 48 变 64。
- 中间列(lazy + 恢复 evict-on-failure)回落到 0.0161,证明 evict 会吃掉 lazy 省出的容量。
- accept length 所有配置不变(2.36-2.43),checkpoint 跟踪不触碰采样路径。

## 验证与回归风险

精度:GSM8K(200q)lazy 0.960 vs extra_buffer 0.965;decode/prefill cache KL 与基线同量级(0.000214/0.000334)。专门的 checkpoint-boundary finish 测试验证请求恰在边界结束且 overlap verify 可能仍在写 slot 的竞态:各种 finish 位置与 pool 状态组合下 greedy token 32/32 一致、max |Δlogprob| ≤3.7e-4;强制 spare-slot 分配失败也通过。新增 CI 测试 `TestQwen3NextMTPLazyV2`(H100×4,est_time 290→430)。风险:DFLASH/DSPARK、PD disaggregation 仍被排除;result 处理对无 plan 的 filtered/merged snapshot 采取保守跳过捐赠策略。

## 关联

前序 PR #27118(extra_buffer_lazy 本体);同区域为 mamba RadixCache/extra_buffer 系列,本卡未收录其他同分类 PR,其余写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
