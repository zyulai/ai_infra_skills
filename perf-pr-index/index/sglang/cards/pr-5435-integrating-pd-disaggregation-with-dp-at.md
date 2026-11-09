# PR #5435: Integrating PD disaggregation with DP attention and DeepEP

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ch-wan |
| 合并 | 2025-04-23 |
| 分类 | MoE (major) |
| Label | high priority |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 含 3 个文件:disaggregation decode/prefill 调度循环与 DP controller |
| 链接 | https://github.com/sgl-project/sglang/pull/5435 |

## 一句话结论

让 PD 分离部署可与 DP attention、DeepEP 组合使用,补齐 prefill/decode 两侧调度循环与请求路由。

## 动机

SGLang 此前已分别支持 PD(prefill/decode)分离、DP attention 与 DeepEP MoE,但三者无法组合:PD 分离的 prefill/decode scheduler 事件循环没有对接 `prepare_dp_attn_batch`,DP controller 的 round-robin 路由也不知道 disagg 模式的会话亲和需求。对于 DeepSeek-V3 这类大 MoE 模型,P 分离配 DP attention(attention 用 DP、MoE 用 DeepEP all-to-all)是主流部署形态,组合缺失直接限制了大模型分离部署的可用性。

## 优化点

在 decode 侧,`SchedulerDisaggregationDecodeMixin` 新增 `_prepare_idle_batch_and_run`,为 DP attention 语义补齐 idle batch 的准备与执行:extend batch(从 prefill 迁移来的首 token)处理后需要额外跑一个 idle batch 以维持 DP 组的步调同步;decode batch 与空批次也统一走 `prepare_dp_attn_batch`。overlap 事件循环中把 `last_batch_is_extend` 语义改为 `last_batch_in_queue`,idle batch 的结果与其他 batch 一样进入 result_queue 延迟处理。prefill 侧两个事件循环在 `get_new_batch_prefill` 之后插入 `prepare_dp_attn_batch` 调用。DP controller 的 round-robin scheduler 在 disagg 模式下改为按 `req.bootstrap_room % len(workers)` 路由,保证同一请求的 prefill 与 decode 落到有 bootstrap 元数据关联的 worker;非 disagg 模式保持原行为。配置由 `enable_dp_attention` 或 `enable_sp_layernorm` 触发。

## 关键改动

```text
python/sglang/srt/disaggregation/decode.py: 新增 _prepare_idle_batch_and_run,normal/overlap 两个 disagg decode 事件循环接入 DP attention idle batch 语义
python/sglang/srt/disaggregation/prefill.py: normal/overlap 两个 disagg prefill 事件循环调用 prepare_dp_attn_batch
python/sglang/srt/managers/data_parallel_controller.py: round_robin_scheduler 在 disagg 模式按 bootstrap_room 路由请求
```

路由改动核心:

```python
def round_robin_scheduler(self, req: Req):
    if self.server_args.disaggregation_mode == "null":
        self.workers[self.round_robin_counter].send_pyobj(req)
        self.round_robin_counter = (self.round_robin_counter + 1) % len(self.workers)
    else:
        self.workers[req.bootstrap_room % len(self.workers)].send_pyobj(req)
```

## 性能声明

PR 正文未提供吞吐/延迟 benchmark 数字(正文的 Evaluation 部分只给出部署流程)。准确率:GSM8K(benchmark/gsm8k 的 bench_sglang.py)为 93.5%~95.0%。测试配置:DeepSeek-V3-0324,4 节点(2 prefill + 2 decode),每节点 TP16/DP8,`--enable-dp-attention --enable-deepep-moe`,prefill 用 DeepEP normal 模式、decode 用 low_latency 模式,KV 传输走 Mooncake RDMA(mooncake 传输引擎 + mini_lb 代理),`--cuda-graph-max-bs 128`。除准确率外 PR 未注明其他性能测试条件。

## 验证与回归风险

PR checklist 中单测、文档等项均未勾选,pack 中未见新增测试;PR 的 TODO 明确写着 "Resolving memory leakage",即合并时已知存在内存泄漏问题待解决。风险面:decode 事件循环为 DP attention/SP layernorm 增加了 idle batch 的准备与执行路径,改变了 overlap 循环的 result_queue 语义,可能影响 disagg decode 在各种 batch 组合下的行为;DP controller 路由逻辑分支依赖 `disaggregation_mode` 与 `bootstrap_room`,非 disagg 路径不变。验证主要靠上述手动 4 节点部署跑 GSM8K。

## 关联

同区域:PD 分离基础设施(preconnect/Mooncake 传输)、DP attention 支持、DeepEP MoE 支持(#5526 等同分类 PR,pack 未列具体编号)。本 PR 是这三者的组合使能,讨论对象含 @ByronHsu @ShangmingCai,测试者为 @liz-badada。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
