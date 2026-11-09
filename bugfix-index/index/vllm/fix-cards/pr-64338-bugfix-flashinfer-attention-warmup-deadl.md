# PR #64338: [Bugfix] Flashinfer attention warmup deadlock under PP>1

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#64338](https://github.com/vllm-project/vllm/pull/64338) |
| 作者 | yzong-rh |
| 合并日期 | 2026-10-31 |
| labels | bug, ready, nvidia |
| 热度 | 👍 0 · 💬 11 |
| 分类 | Attention / Kernel |
| bug 类型 | hang |

## 一句话结论
PP>1 时各 rank 对是否参与 Flashinfer attention warmup 判断不一致,部分 rank 卡在 PP send/recv 造成启动死锁;新增跨 PP rank 同步,全体一致才(或任一 rank 同意即)执行 warmup。

## 问题现象
PP(pipeline parallel)>1 的部署在启动 warmup 阶段死锁:当一个 PP rank 只含 Flashinfer attention 组,而另一个 PP rank 含 Flashinfer 与非 Flashinfer 混合的 attention 组时发生。最常见于 Blackwell 上带 dflash / dspark drafter 的模型——drafter 可选用与 target 不同的 attention backend,而 drafter 只存在于最后一个 PP rank。main 分支复现时反复打印 `No available shared memory broadcast block found in 60 seconds...`;本 PR 下 `vllm serve Qwen/Qwen3-4B --pipeline-parallel-size 2 --attention-backend FLASHINFER --enforce-eager --no-enable-flashinfer-autotune --speculative-config '{"method":"dflash",...,"attention_backend":"FLASH_ATTN"}'` 正常完成启动。`--enforce-eager` 与 `--no-enable-flashinfer-autotune` 均无法规避(前者不跳过 Flashinfer warmup,后者只控制 autotune)。

## 根因
PR 正文披露:`kernel_warmup` 期间会调用 `mixed_batch_attention_warmup` 跑混合 prefill+decode batch 以预热两条 attention 路径;该调用当前以「所有 attention 组都用 Flashinfer 后端」为门控(因混合后端不支持 dummy run)。使用 spec decode 时 drafter 可能选用与 target 不同的后端,而 drafter 只在最后一个 PP rank 上,于是不同 rank 对是否参与 Flashinfer warmup 得出不同结论;由于 mixed warmup 使用的 `worker.execute_model` 会参与 PP send/recv,各 rank 决策不一致即造成死锁。

## 修复方案
在 `vllm/model_executor/warmup/kernel_warmup.py` 中把原内联的 Flashinfer warmup 门控抽成 `_warmup_flashinfer_attention(worker)`:先按 `use_v2_model_runner` 用 `any`/`all` 计算本地是否 `local_flashinfer_warmup`;当 `pp_group.world_size > 1` 时,把本地判定值经 `pp_group.cpu_all_reduce_(eligibility)` 跨 PP rank 求和;v2 runner 下 `> 0` 即 warmup,否则要求所有 rank 均同意(`== world_size`)才执行,保证各 rank 决策一致。关键 diff:

```diff
+    pp_group = get_pp_group()
+    eligible_pp_ranks = int(local_flashinfer_warmup)
+    if pp_group.world_size > 1:
+        eligibility = torch.tensor([eligible_pp_ranks], dtype=torch.int32, device="cpu")
+        pp_group.cpu_all_reduce_(eligibility)
+        eligible_pp_ranks = int(eligibility.item())
```

并在 `tests/model_executor/test_attention_warmup.py` 新增 `test_flashinfer_attention_warmup_gate`,以线程级 PP group 模拟多种 rank 内后端组合验证门控。

## 影响与回归风险
- 影响窗口:正文未说明受影响的版本/提交范围。
- 回归面:仅触及 `kernel_warmup.py` 的 Flashinfer attention warmup 路径与对应测试。PP=1 时 world_size==1,跳过 all-reduce,行为与原来一致(除 v2 runner 由 `all` 语义变为 `any` 判定外);PP>1 时所有 rank 现在统一决策,PP=1 与 PP>1 的 warmup 行为保持一致。原代码注释称 v2 场景本应为 `any`,本次在 v2 路径落实了这一意图。若分布式通信在 warmup 阶段不可用,新增的 `cpu_all_reduce_` 是新的依赖点。

## 关联
- #60791: https://github.com/vllm-project/vllm/issues/60791 (MiniMax-M3 跟踪项,Item 8 涉及 M3 attention warmup 在 PP 各 stage 不一致的问题;M3 warmup 同样使用 `mixed_batch_attention_warmup` 但与 Flashinfer warmup 相互独立)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
