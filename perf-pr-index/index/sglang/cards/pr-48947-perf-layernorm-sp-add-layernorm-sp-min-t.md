# PR #48947: [Perf] layernorm_sp: Add --layernorm-sp-min-tokens to run small extend passes replicated

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @rohithrr9 |
| 合并 | 2026-11-12 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/48947 |

## 一句话结论

新增 `--layernorm-sp-min-tokens`:token 数低于阈值的 extend pass 改走 replicated 路径、不再跑 sequence-parallel;GB300 x4 实测 1024 token 以下 prefill pass TTFT 降 7–17%,1024/1280 token 两路径成本相当(该部署 crossover 约 1024 token),gsm8k 精度持平。

## 动机

`--enable-layernorm-sp` 下每个 EXTEND forward 都跑 sequence-parallel。SP collectives(entry scatter、attention 投影上的 all-gather + GEMM 与 GEMM + reduce-scatter、routed-expert all-to-all)每层成本与 pass 大小基本无关,所以小 extend 走 SP 可能比 decode 已在用的 replicated 路径更慢,而大 chunk 更快;短 follow-up turn、小 chunk 和 prefix-cache hit 都落在小的一侧。

## 优化点

- 新 server argument `--layernorm-sp-min-tokens N`:pass 的 token 数大于 N 才跑 sequence-parallel。做成 server argument 是因为 crossover 是部署属性(replicated 路径的 custom all-reduce 能覆盖哪些 all-reduce 规模),不是模型属性。默认 0,即每个 extend 仍 sequence-parallel。
- `runs_sp(forward_mode, num_tokens)` 在 entry scatter 之前按 pass 的 token 数决策;每个 in-tree caller(model entry、layer-boundary entry、DFLASH aux-tap gather、input-embeds exit、`maybe_exit_gather`、ERNIE 5.5 的 `logits_rows`)都传入 token 数,使 entry 与每个 exit 对同一 pass 的决策一致。
- captured prefill graph 里 entry 按 bucket size 决策,而 graph 外的 exit 看到真实 token 数;两者一致的前提是阈值必须是某个 captured bucket、或不低于最大 bucket——在 argument resolution 时校验,同时校验负值与"开了该 flag 但没开 `--enable-layernorm-sp`"。询问 decode / verify pass(从不跑 SP)的 caller 可省略 token 数;extend pass 省略则 raise。
- 文档:新参数写入 `server_arguments.mdx`。测试:`test_layernorm_sp.py`(每 pass size 的决策、bucket 校验、entry / exit 一致性)。

## 关键改动

```text
python/sglang/srt/arg_groups/fields/parallel.py: 新增 layernorm_sp_min_tokens 字段(默认 0)
python/sglang/srt/arg_groups/layernorm_sp_hook.py: 新增 validate_layernorm_sp_min_tokens(负值 / 未开 SP / 必须 captured prefill bucket 或 >= 最大 bucket)
python/sglang/srt/layers/layernorm_sp.py: runs_sp 增加 num_tokens 参数与阈值判断;maybe_exit_gather 改用 input_ids.shape[0] 决策
python/sglang/srt/layers/layer_boundary/stage.py: layer-boundary entry 传 hidden_states.shape[0]
python/sglang/srt/models/muse_spark.py: model entry / DFLASH aux-tap gather / input-embeds exit 传 pass token 数
python/sglang/srt/models/ernie55.py: logits_rows 传 forward_batch.input_ids.shape[0]
docs/docs/advanced_features/server_arguments.mdx: 新参数文档
test/registered/unit/layer_boundary/test_layernorm_sp.py: 新增阈值决策 / bucket 校验 / exit 一致性测试
```

```python
def runs_sp(forward_mode, num_tokens: Optional[int] = None) -> bool:
    """Whether this forward runs SP: an enabled model, prefill only, and more
    than --layernorm-sp-min-tokens tokens (0, the default, admits every pass)."""
    if not (layernorm_sp_enabled() and forward_mode == ForwardMode.EXTEND):
        return False
    if num_tokens is None:
        raise ValueError(
            "runs_sp needs num_tokens (the pass's token count) for an extend pass: "
            "without it the caller skips --layernorm-sp-min-tokens and, on a pass "
            "at or under the threshold, disagrees with the entry and exit gathers."
        )
    return num_tokens > get_parallel().layernorm_sp_min_tokens
```

## 性能声明

- Prefill pass 延迟(GB300 x4,`lmsysorg/sglang:nightly-dev-cu13-20261106-858073e4` + 本 PR 文件,`Muse-Spark-Open-816B-A42B-NVFP4` TP4/EP4、`--enable-layernorm-sp`、DFLASH(RadixArk/Muse-Spark-Open-DFlash2,8 draft tokens,real acceptance)、FP8 KV cache、HiCache;`--max-running-requests 16`;TTFT 为一次 prefill pass;stock/PR 服务器在同一节点交替两轮(stock, PR, stock, PR),每格为四 run 均值的 mean,括号内为四 run 最快值):128 token 35.8→30.7 ms(−14.4%);256 38.7→32.0 ms(−17.3%);512 43.3→40.2 ms(−7.0%);1024 57.9→58.7 ms(+1.4%);1280 61.5→63.2 ms(+2.8%);1536 66.2→67.8 ms(+2.4%);2048 66.3→67.3 ms(+1.4%);4096 96.1→96.9 ms(+0.9%);8192 158.1→162.8 ms(+3.0%)。
- 正文解读:1024 token 以下 replicated 路径省 7–17%;1536 token 起两臂走同一路径,run 间漂移可达任一臂 10%(8192 token 时每个 run 要么 153–156 ms 要么 168–174 ms,与臂无关),change 列的 +1~3% 即来自漂移,四 run 最快值相互差在 1.2% 内;1024/1280 token 两路径成本约同(最快 run −0.7% / +3.5%),该部署 crossover 约 1024 token。阈值需是 captured prefill bucket;main 默认 buckets 含 1280–4096 每 256 的倍数,故 1280 在默认 `--cuda-graph-backend-prefill breakable` 下有效。
- 精度(同节点 stock/PR 交替,`--layernorm-sp-min-tokens 1280`;gsm8k 经 sgl-eval,1319 题,temperature 1.0,top-p 1.0,max 16384 tokens,seed 42;gsm8k prompt 平均约 225 token,故每个 gsm8k prefill 都走新的 replicated 路径):max running requests 4 时 stock 97.57% vs PR 97.35%;16 时 97.73% vs 97.65%。

## 验证与回归风险

- 单测 `test_layernorm_sp.py` 20 passed(含新增:阈值只放行其上的 pass、decode/verify 可省略 num_tokens 而 extend 缺省会 raise、exit gather 遵循 entry 决策的回归测试、min_tokens 必须 captured prefill bucket、无 prefill graphs 时不设限)。
- 风险:captured prefill graph 的 entry 按 bucket size 决策、graph 外 exit 按真实 token 数决策,阈值若不是 bucket 或低于最大 bucket,replayed graph 会取一条与 exit gather 不一致的路径——`validate_layernorm_sp_min_tokens` 在 resolution 阶段拒绝;prefill graphs disabled 时不受约束。
- 大 pass 下 run 间漂移最高 10%(正文如实标注),因此 ≥1536 token 的 +1~3% 变化不构成回归证据,以最快 run(差 <1.2%)为准。

## 关联

无(正文未列出关联 PR;`--enable-megamoe-for-layernorm-sp` 与 `--enable-layernorm-sp` 的既有校验逻辑在本 PR 中被引用/沿用)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
