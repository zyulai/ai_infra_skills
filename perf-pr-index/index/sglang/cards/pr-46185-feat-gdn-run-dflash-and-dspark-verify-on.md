# PR #46185: [Feat] gdn: run DFLASH and DSPARK verify on the ReplaySSM spec ring

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jianc99 |
| 合并 | 2026-11-11 |
| 分类 | Speculative Decoding(tier: major) |
| Label | run-ci, jit-kernel, memory-pool, bypass-fail-fast |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);pack diff 截断(5 个文件未展示),以下仅覆盖已展示部分 |
| 链接 | https://github.com/sgl-project/sglang/pull/46185 |

## 一句话结论

让 GDN target(Qwen3.5/3.8)上的 DFLASH/DSPARK verify 用 `--enable-linear-replayssm-spec` 的 per-request 紧凑 ring 取代 per-draft 全量 SSM state snapshot:TP1 128 requests 内存 72.6 GB→3.04 GB、KV cache 1.13M→2.03M tokens,c128 decode tok/s +10.6%,并顺带修好 GDN spec ring 的 ring 长度检查、cursor 活跃行一致性与 NaN 泄漏。

## 动机

Qwen3.5/3.8 GDN target 上,DFLASH 与 DSPARK verify 每 draft token 写一份完整 SSM state snapshot(FlashInfer MTP 或 Triton recurrent verify),之后 scatter 被接受的那个。`--enable-linear-replayssm-spec` 用紧凑 per-request ring 避开这些 snapshot,但对非 KDA 模型上的 DFLASH/DSPARK 被拒绝:它们的 commit(`HybridLinearAttnBackend.update_mamba_state_after_mtp_verify`)只认识 KDA ring,会去 scatter ReplaySSM 不分配的 snapshot。把 GDN ring 接进该 commit 还暴露了影响 GDN spec ring 上所有算法的问题:ring 长度从未检查(长于 ring 的 draft window 被静默截断,短于 16 的 ring 在首个 verify 失败);cursor advance 与其余三个 ring kernel 对活跃行的判断不一致;被拒 draft 处的非有限值可能进入 accepted 输出与 committed state;verify 对每行都重放空 history,而 DSPARK、DFLASH 与 bf16 state 的 commit 本来每步都 fold。另外没有任何检查拒绝 DSPARK 在 GDN target 上的 compact verify——GDN target verify 看的是 dense `[batch, draft tokens]` 窗口,而 Qwen3.6 的 eager compact verify 曾被报告触发 illegal memory access。本 PR 是 DFlash adaptive-verification 系列的 component 2。

## 优化点

(pack diff 截断,以下按正文七项概述;代码摘录取自已展示部分)

1. [Fix] 检查 ReplaySSM ring 长度并按 state slot 推进 cursor:`MambaPool` 要求 power-of-two 且 ≥16 的 `--linear-replayssm-cache-len` 能容纳 draft window;cursor advance 跳过没有 state slot 的行。
2. [Fix] 拒绝 GDN target 上的 compact verify:`GDNAttnBackend` 在 target 的 spec 算法构建 ragged layouts 且 `SGLANG_RAGGED_VERIFY_MODE=compact` 时启动即报错(static 与 cap-accept verify、draft workers、其他算法不变)。
3. [Fix] 被拒 draft 不进 ReplaySSM spec 输出与 state:verify 用 `tl.where` 选择因果项而非乘零;commit 把 ring load mask 到 committed history。
4. [Feat] 为 DFLASH/DSPARK commit ReplaySSM spec state:cursor advance、accepted window 的 fold、conv rollback 收进一个 helper `commit_gdn_replayssm_after_verify`,generic spec commit 与 backend commit 都调用;backend commit 无论 state dtype 都 fold 每个 accepted window,`temporal` 即每步后的 committed state;删除 configurator 拒绝,scatter snapshot 前加 assert。
5. [Perf] 无 history 的行跳过 ReplaySSM verify history replay:history 项仅在 `write_pos > 0` 时运行,且挪到 K loop 之后构建。
6. [Fix] 保持 verify selects 中的 fused multiply-add(0d3649d36):commit 3 的 `tl.where(m, a*b, 0)` 拆散了 FMA;改为 `(I + A)^-1` reduction mask 操作数、另两处用 `tl.where(m, tl.fma(a, b, acc), acc)`。
7. [Fix] (test) compact-verify 拒绝测试改按 `supports_ragged_verify()` 判定。

无新 flag/env var。行为变化:容纳不下 draft window 或短于 16 的 ring 现在启动即失败(GDN ring 上任何算法;例如 EAGLE/NGRAM 超过 16 draft tokens 需 `--linear-replayssm-cache-len 32`);GDN + DSPARK + compact mode 启动即失败。内存上 ring 取代 per-draft snapshots:TP1 128 requests Qwen3.8-27B 用 3.04 GB 而非 72.6 GB snapshots,KV cache 从 1.13M 增至 2.03M tokens;DFLASH/DSPARK 每 commit fold,ring 不会超过一个 window,16 是 compact commit 支持的最小 ring(以 `tl.dot` 规约整个 ring)。

## 关键改动

```text
python/sglang/kernels/ops/attention/linear/gdn_replayssm_spec_decode.py: verify kernel 的 history 项按 has_history 门控并移到 K loop 后;因果项改 select/FMA;compact commit 的 ring load mask 到 committed history;cursor advance 按 state slot 判活跃行;新增 commit_gdn_replayssm_after_verify 共享 helper
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py: update_mamba_state_after_mtp_verify 增加 ReplaySSM-GDN 分支调用共享 helper;PLE state commit 抽成 commit_ple_states_after_verify
python/sglang/srt/layers/attention/linear/gdn_backend.py: 新增 _validate_gdn_target_verify_layout(拒绝 GDN + compact verify)
python/sglang/srt/mem_cache/memory_pool.py 与 mem_cache/kv/decl.py: GDN spec ring 长度校验(power-of-two、≥16、容纳 draft window)
python/sglang/srt/mem_cache/kv_cache_configurator.py: 删除 _validate_replay_spec_algorithm(DSPARK/DFLASH 的 KDA-only 拒绝)
python/sglang/srt/speculative/spec_utils.py: 两处 commit 路径统一走 commit_gdn_replayssm_after_verify 与 commit_ple_states_after_verify
test/registered/unit/spec/test_gdn_replayssm_dflash_commit.py(新,CPU)、test/registered/attention/unittests/gdn/test_gdn_replayssm_direct_commit.py(新,GPU)
```

```python
    b_wp_i = b_write_pos.to(tl.int32)
    cache_valid = o_c < b_write_pos
    # A row folded at its last commit has no committed history to replay; its
    # history terms would add exact zeros, so they are skipped.
    has_history = b_write_pos > 0
...
        if not IS_FLUSH:
            if has_history:
                # cached-key history load -> phys_c (output reconstruction only)
                ...
                scores_q += tl.dot(khist_tile, qT, input_precision=DOT_PRECISION)
                scores_k += tl.dot(khist_tile, kT, input_precision=DOT_PRECISION)
```

## 性能声明

(全部先于 item 6 测得,item 6 的影响见 kernel 表;设置:Qwen3.8-27B(48 GDN + 16 attention layers)、DFlash2 block-8 draft、GB300、TP1、`--max-running-requests 128`、bf16 mamba state、radix cache off、overlap on;SPEED-Bench first turns,T=1,top_p 0.95,top_k 20)

- 端到端 decode tok/s: c128 snapshot 6967/6921/6994 vs ReplaySSM 7730/7714/7672/7686(+10.6%,均值);c64 +10.4%;c32 +7.1%;c8 +8.8%;c1(16 requests)−5.9%;c1(48 requests)−0.5%;c1 greedy(48 requests)+2.0%。c128 时 running cap 等于并发、约一个请求排队,#46182 显示该档位读数比更大 cap 高 7–15%(两侧同样),只在表内比较。acceptance length c8–c128 不变(3.55–3.69 vs 3.55–3.70);c1 的 acceptance 由轨迹主导,verify step rate 在 48 requests 上差距 1% 以内(93.7 vs 94.3 steps/s)。
- Verify kernel 每层时间(batch 128,GB300,commit 3/5 前后):bf16 无 history 0.111→0.116→0.109 ms;fp32 无 history 0.186→0.188→0.154 ms;fp32 带 history(lazy fold)0.244→0.250→0.209 ms(before 3 / after 3 / after 5)。bf16 带 history 在 item 5 后慢 9%,但没有 commit 路径会产生它。
- item 6 后的 ReplaySSM spec verify kernel(main 为本 PR 前 kernel,batch 128,中位数):bf16 window 4: 83.97→85.34→79.23 µs;window 8: 120.13→114.69→114.69;window 16: 264.19→235.52→233.47;fp32 带 history window 4: 149.54→141.31→139.26;window 8: 200.70→196.61→192.54;fp32 window 16: 430.08→382.98→371.68。item 6 比拆散乘加快 0–7%;应用于无 commit 5 的 commits 1–4 时每行都在 main 的 2.3% 以内——masking 不再费时。

## 验证与回归风险

- 仅在 GB300(SM100)验证,upstream 镜像 `lmsysorg/sglang:v0.5.21` 挂载本树;GPU 测试的数值界以 Triton snapshot verify(可在任何 GPU 上跑)为参照,SM90 上未跑过。
- 新 CPU 测试覆盖 ring-length 报错、DFLASH/DSPARK backend commit 路由(padding/PP/track slots)、generic commit 传 state slots、snapshot assert、compact 拒绝与 static/cap-accept 放行。新 GPU 测试在 Qwen3.8 head shapes(HV/H=3, K=V=128, block 8, ring 16)下对 bf16/fp32 state:256 步 verify/commit 链对照 float64 sequential GDN reference(界为 Triton snapshot verify 误差的 1.25x;实测 outputs 1.05/1.03、states 0.97/1.00);128 步 lazy-fold 链(重放至多 16 条 history records,ratio 1.05);track slots;在被拒 draft 注入 NaN/Inf 时输出/state/conv/cursor bit-identical;无 state slot 行 cursor 不动;CUDA graph replay 与 eager 一致。两文件合计 16 passed + 13 subtests;既有回归集 53 passed(本分支与 main)。compute-sanitizer:memcheck 无 device error,racecheck 0 hazards。
- item 6 的 bit-exactness(12 链式 verify+commit 步,batch 16):commits 1–4 + item 6 在有限输入上对 main 全 bit-identical(window 4/8/16,bf16/fp32,fold/lazy,8 种配置);全分支 bf16 bit-identical,fp32 仅因 commit 5 的 TF32 舍入不同(全分支 4.7M 输出中 2827 个不同 at window 4,lazy fold 9.4M 中 2121 个 at window 8);item 6 之前 bf16 在 window 4/8/16 分别有 52/235/619 个输出偏离 main。NaN/Inf 注入下 main 会把 accepted 输出全变 NaN,本 PR 不变。
- Greedy serving 检查(Qwen3.8-27B + DFlash2 block-8,TP1):acceptance 匹配(c1 3.83 vs 3.77,c8 3.88–3.90 vs 3.88–3.89);ReplaySSM 在 c1 与 snapshot 精确匹配 4/16 请求,分歧平均在 47% 文本后,且全部(12/12)起于 top-2 logprob gap <0.05 的近平局;snapshot 路径对舍入同样敏感(fp32 state 时自匹配仅 4/16)。TP2 greedy(c1/c8)无错误。这些 serving 检查先于 item 6。风险:fp32 state 的 TF32 舍入差异(commit 5)与 c1 单请求吞吐 −5.9%(16 requests)。

## 关联

#46182(系列集成与含 item 6 的端到端结果)、#46189(作者对 ReplaySSM 路径的 ragged-verify 放行版本)、#38897(冲突:声明 GDN ragged-graph 支持,与 commit 2 互斥)、#42688(与 commit 3 相同的 history-load masks)、#36683 / #35725 / #42209(commit 4 的替代方案,Reviewer 需择一)、#35413(commit 2 所拒绝的 eager compact illegal memory access 报告)、#28511(ReplaySSM spec-verify RFC,注释中提及)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
