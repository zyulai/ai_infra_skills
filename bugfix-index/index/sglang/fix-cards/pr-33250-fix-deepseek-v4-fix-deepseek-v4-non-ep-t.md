# PR #33250: [Fix] deepseek-v4: Fix DeepSeek-V4 non-EP TBO for attention TP > 1

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33250](https://github.com/sgl-project/sglang/pull/33250) |
| 作者 | mikekg |
| 合并日期 | 2026-10-26 |
| labels | amd, bug, deepseek, run-ci |
| 热度 | 👍 0 · 💬 26 |
| 分类 | Overlap / Scheduler |
| bug 类型 | general |

## 一句话结论
DeepSeek-V4 非 EP TBO 在 attention TP>1 时对组内复制张量做全 TP 变长集合通信,输出损坏或崩溃;改为分片 gather、reduce-scatter 后组内 all-gather 重建。

## 问题现象
DP attention + `moe_a2a_backend=none` + attention TP > 1 且开启 TBO 时:在较旧的 `main`(如 pristine `574ead753`、父提交 `68d70aa8f`)上,gather/combine 布局被破坏,服务返回 HTTP 200 但生成文本完全乱码(H100 TP16/DP2/attention-TP8 上问 "7+5" 输出 `我这里特 connKadaghanJF by,]_ ...`,`grep -q 12` 失败);在当时的 `main` 上,TBO gather 直接报错 "a per-replica sequence of 4 entries does not belong to a DP gather of width 2"。修复过程中还暴露同一配置的连带问题:单个短请求时 TBO 会切出全 rank 均为空的 micro-batch,triton runner 的 fused SiLU-and-mul 拒绝零行 launch 导致服务崩溃;eager decode 下 TBO decode 子 batch 不随 token padding 断言失败(`req_pool_indices 1, out_cache_loc 2`);MI355X 上 ROCm 的 aiter 自定义 all-gather 报 "invalid configuration argument"。

## 根因
正文披露:TBO 与 DP attention、`moe_a2a_backend=none`、attention TP > 1 组合时,全 TP 变长集合通信(gatherv/reduce_scatterv)被直接应用在 attention-TP 组内已复制的张量上——每个子 batch 的 DP counts 每个 TP rank 各有一条,而非每 DP rank 一条,破坏了 gather 与 combine 的布局。此外(正文 "Added after review" 披露):空 micro-batch 在零行上跑 experts 触发 kernel 拒绝;decode 子 batch 保持请求数而 token 已被 pad 到 attention TP 的倍数,二者不一致;ROCm 上 `all_gather_into_tensor` 走 aiter 自定义 all-gather,与另一 micro-batch 的 compute-stream 集合通信冲突。

## 修复方案
- `deepseek_v4.py` 新增 `_tbo_collective_sizes()`:把 per-TP-rank sizes 折叠为 DP counts 并均分成 TP shards,校验组内一致且可整除;`_forward_layers_tbo` 改用 `dp_sizes`/`tp_sizes`,`global_num_tokens_gpu` 只取 `gathered[::attn_tp_size]`;input_ids 先切 shard 再 all_gatherv。
- `moe_transport.py`:`_tbo_gather_moe_input` 在 attention TP>1 时对本 rank shard 走 `tp_group.all_gatherv`(attn_tp==1 保留 `dp_gather_replicate`);`dp_reduce_scatterv_async` reduce-scatter 成每 rank 一片后,用 `attn_tp_group.device_group` 做 all-gather 重建 DP-local 输出。
- `_can_run_tbo` 收紧为 CP1、multi-rank DP、SUM_LEN padding(MAX_LEN 走非重叠路径);空 micro-batch 直接跳过 experts;decode 子 batch 以 padded token 数 pad(`_pad_tbo_child_inputs`);`shared_experts` 显式初始化为 `None` 替代 `hasattr` 判断。

关键 diff(moe_transport.py):

```python
    parallel = get_parallel()
    local_shard = (
        local
        if parallel.attn_tp_size == 1
        else local.tensor_split(parallel.attn_tp_size)[
            parallel.attn_tp_rank
        ].contiguous()
    )
```

## 影响与回归风险
- 影响窗口:正文未给版本号;复现于 pristine upstream `574ead753` 与父提交 `68d70aa8f`(#31700 合并后的旧 main)。
- 回归面:非 EP TBO 的 gather/combine 全路径(`dp_attention.py`、`deepseek_common/moe_transport.py`、`deepseek_v4.py`、`deepseek_v4_platform_stage.py`);`forward_batch_info` 的 TBO 子 batch padding(decode/extend 语义不同);`deepseek_v2.py`/`hunyuan_v4*.py` 中 `hasattr(self, "shared_experts")` 改为 `is not None` 的判断;ROCm 上 combine all-gather 改走 device process group(不再用 aiter 自定义 all-gather)。

## 关联
- https://github.com/sgl-project/sglang/pull/31700(本修复依赖)
- https://github.com/sgl-project/sglang/pull/45507
- https://github.com/sgl-project/sglang/pull/45728
- https://github.com/sgl-project/sglang/pull/46570

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
