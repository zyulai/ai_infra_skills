# PR #50514: [Core][MRV2] Support eagle3 spec decode with pipeline parallel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yongqinwang-cmd |
| 合并 | 2026-09-05 |
| 分类 | Speculative Decoding (tier: major) |
| Label | speculative-decoding, ready, ci/build, llama, qwen, deepseek, cpu, nvidia, mrv2, DSv4, dflash, kimi, k3 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明 "6 file(s) dropped from diff" |
| 链接 | https://github.com/vllm-project/vllm/pull/50514 |

## 一句话结论

让 eagle3/dflash/dspark 外置 draft 支持 PP，8x B200 上 PP=1~4 的 EAGLE3 acceptance 稳定在 2.03 左右。

## 动机

为显式 opt-in 的 target 模型启用 EAGLE3 风格外置 draft 模型（`eagle3`、`dflash`、`dspark`）与 pipeline parallelism。draft 模型只在最后一个 PP stage 运行，但其 auxiliary hidden states 可能来自任意 stage 的层，需要在 PP handoff 中传递。正文另说明本 PR 不是 #46994 的重复：后者启用 MTP，其 draft 用最后 stage 已有的 target 最终 hidden state，两者只在共享 PP bookkeeping fixes 上重叠，后落地者可 rebase 这些部分。

## 优化点

每个 stage 把自己产生的 states 用全局有序 slot 加入现有 `IntermediateTensors` handoff；中间 stage 转发从前序 stage 收到的 states；最后 stage 按层序重建列表。模型 forward 内部不增加通信——packing 是普通 tensor/dict 输出工作，relay 处理留在 runner 的 CUDA graph capture 之外。PR 还让 PP 路径其余部分可用于 spec decode：只在最后 stage 实例化 draft 模型而非继承 target 的 PP size；从最后 stage 广播 draft proposals 使前序 stage 下一步 embed 正确 token IDs；每个 rank 用固定 sampled-token 广播宽度并修正投机 token 的 finishing-request 计算；draft 模型用全局 target-layer 编号；draft checkpoint 共享 target input embedding 时在最后 stage 实例化并加载。不支持的 target 实现仍在模型加载期报错，说明 auxiliary hidden states 未跨 PP stage 转发。Full CUDA graphs 仍支持：被捕获的 forward 只把本地 auxiliary hidden states 加入 `IntermediateTensors` 输出，PP forwarding、sampled-token/draft-token 广播在 `PPHandler` 内、capture 之外；draft receive 并入 `PPHandler.receive()` 只是结构简化，用同一 side stream 与 communicator，保持 sampled/metadata/draft 集合顺序并在三次 receive 后只 record 一个 event。

## 关键改动

```text
vllm/v1/worker/gpu/model_runner.py: runner 侧向 IntermediateTensors 加 auxiliary states、draft 仅最后 stage 实例化、广播在 capture 外
vllm/v1/worker/gpu/pp_utils.py: PPHandler 承载 PP forwarding、sampled-token/draft-token 广播与 draft receive
vllm/model_executor/models/interfaces.py: EagleModelMixin / aux hidden states over PP 的校验接口
vllm/config/speculative.py 与 vllm/config/vllm.py: draft 并行配置与 draft_parallel_config
tests/v1/worker/test_eagle3_aux_hidden_states_pp.py: auxiliary-state 排序、PP sizes、边界层与 embedding sharing 的 CPU 单测
```

## 性能声明

Hardware validation（开发期）：Llama-3.2-1B + EAGLE3 on 8x B200，PP=1/2/3/4 的 acceptance lengths 为 `2.033 / 2.033 / 1.996 / 2.046`；DeepSeek-V4-Flash + DSpark at TP=2，PP=1/2/3/4 为 `3.436 / 3.620 / 3.520 / 3.488`。PR head `597ed4934`：Llama-3.3-70B + EAGLE3 at TP=2 x PP=2 跨两节点 NCCL over TCP sockets，达 `2.59 mean acceptance length`，相对 spec-off 在 concurrency 1/8 的吞吐 `2.34x / 1.67x`；正文注明 final review refactor 只有 CPU 与 lint 覆盖，尚未在该硬件重跑。Refactor 后：pre-commit/mypy 与聚焦 CPU 测试通过；2×B200 上 real-weight Kimi-K3 + DSpark at TP8×PP2 完成 FULL 与 PIECEWISE CUDA graph capture，零 restart、零 CUDA/NCCL fault；Full GSM8K lm_eval（1,319 examples、5-shot、temperature 0）strict 与 flexible exact match 均 `0.96361 ± 0.00516`，与先前 `0.96437 ± 0.00511` 统计上无变化，mean acceptance length `4.626`（7 draft tokens）。

## 验证与回归风险

CPU 单测覆盖 auxiliary-state 顺序、PP sizes 1/2/3/4/6/8、边界层、embedding sharing、compile-safe packing：`34 passed`；Ruff check 与 format check 通过；GPU e2e 比较 PP=1 与 PP=2 在 eager 及 `FULL_AND_PIECEWISE` 模式、以及 PP=4 在 `FULL_AND_PIECEWISE` 模式的 acceptance。风险面：未 opt-in 的 target 实现在加载期报错；Full/Piecewise CUDA graph 捕获、跨节点 NCCL、sampled/draft 广播顺序；正文明确 final refactor 未在 Llama-3.3-70B 双节点硬件重跑。

## 关联

#46994（MTP 的 PP 支持，正文明确说明非重复、仅共享 PP bookkeeping fixes 重叠）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
