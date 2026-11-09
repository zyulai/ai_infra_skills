# PR #23675: perf: add --prefill-only-disable-kv-cache to skip KV pool allocation

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jasperjiaguo |
| 合并 | 2026-05-11 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | run-ci |
| Diff | pack 未提供 diffstat(diff 完整可见,改 4 个文件:memory_pool、kv_cache_mixin、server_args、单测) |
| 链接 | https://github.com/sgl-project/sglang/pull/23675 |

## 一句话结论

新增 opt-in `--prefill-only-disable-kv-cache`,embedding 等 prefill-only 负载跳过 KV pool 物理分配,省 88.9 GB 显存、吞吐 +6.3%。

## 动机

prefill-only 工作负载(如 `--is-embedding` 经 FA3 `fa_skip_kv_cache` 的路径,以及经 `/v1/score` 走 `EmbedReqInput`、内部置 `is_embedding=True` 的 SequenceClassification/RewardModel/cross-encoder scoring)中没有任何层读写 KV cache——attention 直接用原始 K/V(`flash_attn_varlen_func`)。此时自动分配的数十 GB KV buffer 是纯浪费。目标是在这类模式下跳过物理 KV pool 分配。

## 优化点

新增 `NoOpMHATokenToKVPool(MHATokenToKVPool)` 子类:`_create_buffers` 每层只分配 `(page_size, head_num, head_dim)` 占位张量(合计约数百 KB 至 ~112 KB 量级)而非完整 `(size+page_size, head_num, head_dim)` buffer;占位形状按 page-size 设计,使 FA backend 在 `forward_extend` 顶部无条件执行的 `key_cache.view(-1, page_size, head_num, head_dim)` 对任意 `--page-size` 都成功。`_finalize_allocation_log`/`get_kv_size_bytes` 上报零占用;`get_key/value/kv_buffer` 返回占位张量(读到即被丢弃,因为后续分支切换到 raw K/V);`set_kv_buffer` 直接 raise、`move_kv_cache` 为 no-op——误用时在第一次写 cache 就大声失败,而不是悄悄产出错误结果。调度器准入不变(`self.size` 仍为逻辑容量)。配置校验:结构前置条件为 `chunked_prefill_size == -1` 与 `--disable-radix-cache`;当前版本要求 `--is-embedding`、FA 预填充后端(fa3/fa4)、非 FP4 KV dtype,并拒绝 DeepSeekV4/NSA/MLA/SWA/hybrid linear-Mamba/Ascend/out-of-tree 等不支持的 pool 家族、context-parallel attention 与 HiSparse。CausalLM scoring(`max_new_tokens=0`)尚未覆盖,需 FA 后端一行 `fa_skip_kv_cache` 扩展,留待单独 PR。

## 关键改动

```text
python/sglang/srt/mem_cache/memory_pool.py: 新增 NoOpMHATokenToKVPool(占位 buffer/零上报/写保护)
python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py: pool 家族校验 + 按 flag 选 NoOp pool + 防御性断言
python/sglang/srt/server_args.py: 新增 flag 与两段式校验(args 前置条件 + backend 依赖约束)
test/registered/unit/server_args/test_server_args.py: 新增 TestPrefillOnlyDisableKvCache 8 个用例
```

```python
def _create_buffers(self):
    # Total footprint is still on the order of KB vs GBs for a real pool.
    self.k_buffer = [
        torch.zeros((self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype, device=self.device)
        for _ in range(self.layer_num)
    ]
    self.v_buffer = [
        torch.zeros((self.page_size, self.head_num, self.v_head_dim), ...)
        for _ in range(self.layer_num)
    ]
```

## 性能声明

测试条件:H200,Qwen3-0.6B FP8 embedding,生产流量分布,`--is-embedding --chunked-prefill-size=-1 --max-prefill-tokens=32768 --enable-piecewise-cuda-graph --piecewise-cuda-graph-compiler=inductor`,RPS=45,1 item/req,CONSTANT pace,60s x 3 组:

| 配置 | Mean items/s | 空闲显存 |
|---|---|---|
| 无 flag | 37.61 | 46.8 GB |
| 有 flag | 39.97 | 135.7 GB |

+6.3% 吞吐,+88.9 GB 释放显存。PR 说明吞吐提升"推测是 cuda allocator 碎片/缓存余量效应",两种配置下 KV pool 都不在热路径;3 轮实测 38.70/40.21/41.01 items/s。

## 验证与回归风险

新增 8 个 server_args 校验单测(非 embedding、非 -1 chunked prefill、radix cache 开启、attn-cp>1、prefill-CP、hisparse、fp4 dtype 均被拒绝);手工验证项:引擎带 flag 启动、日志出现 `KV Cache skipped (no-op pool)` 与 `avail mem=135.73 GB`(基线 46.81 GB)、`POST /v1/embeddings` 返回有效 embedding、3x60s 基准无调度异常、占位张量随 `--page-size` 缩放。回归面:仅 plain MHA + CUDA + FA 预填充后端 + embedding 模式可用;若未来 pool 选择重构改变分支,boot 时的防御性断言会兜底;不支持模式下的误用会以 RuntimeError 暴露。

## 关联

后续(PR 正文预告):CausalLM scoring(`max_new_tokens=0`)的 `fa_skip_kv_cache` 扩展单独 PR;scoring/MIS 等其他 prefill-only 路径未来可复用同一思路。pack 未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
