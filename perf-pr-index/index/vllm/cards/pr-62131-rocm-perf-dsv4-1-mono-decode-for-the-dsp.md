# PR #62131: [ROCm][Perf][DSv4.1] Mono decode for the DSpark draft layers

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Fangzhou-Ai |
| 合并 | 2026-10-19 |
| 分类 | Speculative Decoding \| tier: major |
| Label | rocm, ready, deepseek, dflash, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/62131 |

## 一句话结论

DSpark draft 三层改走 mono K1/K2，MI355X 每步 kernel 约 640us→227us，ITL p50 低约 1.8%。

## 动机

DSv4.1-Flash 在 MI355X 上用 `--monokernel` 时，target 层已走 mono K1/K2 kernel（ATOM 的 `DraftMonoRunner` 也是），但 DSpark draft 的三层仍走 vLLM 的 per-op path。本 PR 是 #60397 与 #61082 的跟进，让 draft 三层同样跟随 `--monokernel`、不引入新开关；与 #60731（gfx942 port）改动文件重叠，后合并者需 rebase。

## 优化点

draft 层是带 window-only attention（compress ratio 0、无 indexer）的 `DeepseekV4DecoderLayer`，无需新增 stage，差异由 build 与 step 的整数承载：`MonoBuild` 新增 `experts`/`topk`/`ug_groups`/`swa_width` 并传入 MoE stage 与 scratch layout（128 routed experts、top-3、每 ug task 一个 FP4 group）；K1 的 key-table stage 原把 SWA 行宽读作常量 128，现改为从 build 读取 draft 的非因果行宽（window 128 + 5 drafts = 192）；首层接收已 repeat 到 hc copies 的 embedding `[rows, hc, dim]`，K1 按 `fold=False` 当作 settled residual。scratch layout 依赖 expert 数，且 mailbox word 不能在一次 launch 作 tag、另一次作 data，故 draft 层共享第二个 `DSV41MonoLayer`（epoch/scratch/peer buffers），每个 rank 在同一 eager step 的同一层创建。target builds 不变：新字段默认 target 值、kernel symbol 不加后缀、target 层 K1/K2 ISA 与 `main` 一致（`FLYDSL_DUMP_IR` diff 0 行）。

## 关键改动

```text
vllm/models/deepseek_v41/amd/mono/layer.py: MonoBuild 增加 experts/topk/ug_groups/swa_width，scratch_layout 与 K1/K2 symbol 按 build 参数化。
vllm/models/deepseek_v41/amd/mono_decode.py: _mono_runner 增加 draft 专用 runner；MonoDecodeLayer.create 按 layer_id 识别 draft 层并取 DSpark routing。
vllm/models/deepseek_v41/amd/mono/runner.py: DSV41MonoLayer 接受 experts/topk/ug_groups/swa_width。
vllm/models/deepseek_v41/amd/mono/attention/front.py: FrontBuild 增加 swa_width。
tests/models/test_dsv41_mono_decode.py: 新增 draft 层走自有 mono runner、首层 broadcast embedding、SWA 行宽隔离 3 个测试。
```

```python
def _mono_runner(device, draft=None):
    global _runner, _draft_runner
    if draft is not None:
        if _draft_runner is None:
            _draft_runner = DSV41MonoLayer(
                get_tensor_model_parallel_world_size(),
                get_tensor_model_parallel_rank(),
                get_tp_group().cpu_group, device,
                experts=draft.experts, topk=draft.topk,
                ug_groups=DRAFT_UG_GROUPS, swa_width=draft.swa_width,
            )
        return _draft_runner
```

## 性能声明

全在 MI355X、DeepSeek-V4.1-Flash、InferenceX #3690 recipe、`main` 于 `ce49c47cbb` + 本 PR；"Off" 为同构建把 draft 层放回 vLLM path（仅 A/B 用的临时 env 开关，不在本 PR 内）。Claims：graph profile（TP4 c1）draft 三层约 227 us kernel 时间/步，原路径约 640 us，约少 414 us GPU 时间/步；K1 约 20 us、K2 约 56 us 每层，一个 step 的非 target kernel 合计 1.76 ms（draft off 为 2.17 ms，排除 all-reduce spin-waits）。AgentX c1、3600 s、InferenceX `infx.bench agentic`、DSpark synthetic acceptance 3.51、Rust frontend（参考 InferenceX run 38377181342，nightly 5555a2d8，非同节点基线）：per-request ITL p50 TP4 1.646 vs 1.678 ms（−0.032，95% CI [−0.034, −0.027]）、TP2 1.970 vs 2.000 ms（−0.030 [−0.034, −0.026]），正文称比带 `--monokernel` 的 InferenceX run 低 0.030 ms（1.8%）；p90 interactivity +4.0%/+6.1% 区间跨零不算显著；TTFT 差异来自节点而非本 PR（prefill 未改）。Draft 质量不变：DSpark mean acceptance length 3.889/3.935（TP4）、3.917（TP2）对 3.899/3.896（off），GSM8K 0.975/0.972。Accuracy（RFC #59665 协议）：TP2 GPQA-D 84.8±2.2、AIME25 98.3±0.9；TP4 GPQA-D 84.5±2.3、AIME25 98.8±0.7；RULER NIAH 32K/128K/256K 全 40/40。

## 验证与回归风险

`tests/models/test_dsv41_mono_decode.py`：41 pass、3 new（draft 层在自有 runner 上走 mono、首层取 broadcast embedding、层只取自身 SWA 行宽）；target build 在 TP4、M=6 的 K1/K2 ISA 与 `main` 一致（`FLYDSL_DUMP_IR` diff 0 行）。Layer level：真实 eager decode steps（TP4 c1/TP2 c1，5-/10-row steps、相同输入）x 输出 cosine ≥0.99975（相对 L2 最高 2.3%，来自 MoE top-3 near-ties）、residual/post/comb/pre mixes ≥0.9999 cosine。限制（正文自述）：仅 MI355X（CDNA4）、TP2/TP4、每步 ≤48 行（draft 每请求 5 行，最多 8 请求），层对比只在 eager steps；draft 剩余每步开销在层之外（`_prepare_dflash_inputs` 3 次 launch 共 113 us、六个 skinny head GEMM 279 us、约 75 个小 elementwise/copy kernel）；submitter 逐行 review 仍 pending。

## 关联

#60397、#61082（前序，target 层跟随 `--monokernel`）；#60731（gfx942 port，文件重叠需 rebase）；#61934（同节点 main 对照中的另一 kernel 改动）；#59665（accuracy 协议 RFC）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
