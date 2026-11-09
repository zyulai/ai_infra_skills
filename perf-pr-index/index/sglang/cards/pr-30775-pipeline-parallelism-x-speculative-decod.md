# PR #30775: Pipeline parallelism x speculative decoding (EAGLE/MTP) compatibility

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @AliceChenyy |
| 合并 | 2026-09-17 |
| 分类 | Speculative Decoding (tier: major) |
| Label | hicache, run-ci, bypass-fastfail, run-ci-extra, unified-radix-cache |
| Diff | pack 未提供 diffstat(diff 因超长截断,13 个文件未收录,新增 pp_spec_relay.py 等未可见) |
| 链接 | https://github.com/sgl-project/sglang/pull/30775 |

## 一句话结论

默认关闭的 `SGLANG_ENABLE_PP_SPEC=1` 打通 PP x EAGLE/MTP,SM120 PCIe 8 卡 GLM-5.2 上 bs32 吞吐 +81%、TTFT 2.7x。

## 动机

PP 与投机解码此前互斥(`server_args.py` 断言禁止)。在只有 PCIe、无 NVLink 的多卡机器上,TP-only 扩展被逐层 all-reduce 瓶颈,PP 才是正确的扩展轴;而 vLLM 已支持 PP x spec(vllm-project/vllm#16568),SGLang 没有。本 PR 以 env gate 实现,默认行为不变。关键观察:非 spec PP decode 已依赖 last->first 的输出 dict relay(下一个采样 token 成为下一轮输入),投机解码在结构上相同——relay 负载扩为 `{accept_lens, new_seq_lens, bonus_token, 下一轮 draft 树}`。

## 优化点

设计五要素:1) draft worker 只存在于最后一个 PP stage(需要最终 hidden states 与 lm_head),非最后 stage 经普通 tp_worker 跑 verify 形状的 target forward;2) tail-drafting:最后 stage 在第 r 轮 verify 结束时为 r+1 轮起草,stage 0 开跑前下一轮 verify token 必须就绪;3) relay 的树是一等 `SpecInput`——`PPSpecRelayInput` 携带 token 与拓扑(`parent_list`/`top_scores_index`),挂在 `ScheduleBatch.spec_info` 并实现 `filter_batch`/`merge_batch`,每个 stage 用 `build_tree_kernel_efficient` 重建相同的 `EagleVerifyInput`,非最后 stage 无需 draft 模型;4) 按 rid 而非位置折叠 relayed 张量(forward 时的组合与结果绕环回来时的 live 组合可能不同);5) KV 记账沿用 overlap 路径的 2x reserve,`kv_committed_len` 由 relayed CPU `accept_lens` 推进。两个 hazard 修复:PP 环序(spec relay 激活时按 pp_rank 奇偶配对 send/recv,否则多个 send 先入队可形成环等)、chunked prefill 中间 chunk 的占位 next_token_ids 不得种入下一轮 verify。另有细节:非最后 stage 的 `eagle_prepare_for_verify` 必须在 `_forward_isolation` 内运行(其把 forward_mode 改为 TARGET_VERIFY,不恢复会被下一轮当 extend 重新合并造成重复请求)。

## 关键改动

```text
python/sglang/srt/managers/scheduler_pp_mixin.py: spec relay 张量拼装/按 rid 折叠/verify input 重建/KV 紧凑/环序配对
python/sglang/srt/managers/scheduler.py: 非最后 stage 不建 draft worker;run_batch 的 verify-round 分支
python/sglang/srt/arg_groups/validation_hook.py: gate 开启时的约束断言(非 overlap/单层 EAGLE/非 PD/非 DP attention)
python/sglang/srt/environ.py: 新增 SGLANG_ENABLE_PP_SPEC(默认 False)
python/sglang/srt/speculative/pp_spec_relay.py: PPSpecRelayInput(diff 未收录,按正文描述)
test/registered/pp/test_pp_spec.py: 一致性与 gate 测试
```

```python
# PP+spec: the draft model (MTP layer) needs final hidden states and
# the lm_head, both of which live on the last PP stage only.
if (envs.SGLANG_ENABLE_PP_SPEC.get()
        and self.ps.pp_size > 1
        and self.ps.pp_rank != self.ps.pp_size - 1):
    self.draft_worker = None
    self.external_corpus_manager = None
    return
```

## 性能声明

测试条件:8x RTX PRO 6000 Blackwell Server(SM120,PCIe 无 NVLink),GLM-5.2-NVFP4、FP8 KV、EAGLE(2 steps/topk 1/3 draft tokens),普通 NCCL all-reduce,8K 输入;GSM8K 200 题 8-shot greedy 每配置 2 轮。

| | TP4xPP2+MTP(gate on) | TP8+MTP(gate off) |
|---|---|---|
| GSM8K x2 | 0.940 / 0.935 | 0.955 / 0.940 |
| accept_length | 2.79-3.00 | 2.68-3.00 |
| 8K bs1 TTFT/TPOT (ms) | **1721** / 9.87 | 1888 / **8.73** |
| 8K bs8 TTFT/TPOT/tok/s | **5521** / 30.28 / **154.5** | 11032 / 30.55 / 108.7 |
| 8K bs32 TTFT/TPOT/tok/s | **13189** / **95.3** / **218.0** | 35897 / 105.2 / 120.1 |
| KV pool | **389,184** | 241,728 |

PP 优势:TTFT(bs8 2.0x、bs32 2.7x)、吞吐(+42%/+81%)、KV 容量 1.61x。PR 附重要更正:早期版本声称 PP 全面 3-6x,复测显示 TP8+MTP 的 bs1 TPOT 反而快 13%(8.73 vs 9.87 ms)——主因是 PP+spec 强制 `--disable-overlap-schedule`;支持 overlap scheduler 是明确 follow-up。另在 DeepSeek-V4-Flash FP8 TP2xPP2(4x RTX 6000D)验证:GSM8K 0.95-1.00,accept 2.44-2.72。测量注意事项:该栈上 GSM8K 有 ±2-3% 的轮次方差(同一 build 两次得 0.940/0.965),结论是"无可检测回归"而非 bit 级一致。

## 验证与回归风险

`test_pp_spec.py` 两个测试:`TestPPSpecConsistency` 要求 PP2+spec 与普通 spec 在 GSM8K 精度与 accept length 双指标匹配(仅精度无法区分坏 relay——树丢了 bonus token 仍被强制接受,输出仍对但接受率塌),覆盖 topk=1 链与 topk=2 树;`TestPPSpecGate` 锁定默认关闭与不兼容组合(adaptive spec、token-map、EAGLE3、DP attention、PD disaggregation、overlap 开启)在启动时被拒。已知缺口:仅非 overlap 调度;topk>1 树已实现但仅单测覆盖(DSA 后端 page_size=64 与 topk>1 上游互斥);tail-drafting 每轮一次 D2H sync(可优化)。回归面:默认关闭,不影响既有 PP 与 spec 路径;开启后要求 `--disable-overlap-schedule`。

## 关联

强相关:#31139(PP + EAGLE)独立收敛到几乎相同设计(draft 在最后 stage、tail-draft、relay 原始树、每 stage 重建);本 PR 不改任何模型文件、gate 默认关、是该组中唯一带测试的,#31139 则支持 PP+DP attention。前序依赖:#26928(SM120 GLM DSA);#30774 修复已并入本 diff,#30994 不再需要。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
