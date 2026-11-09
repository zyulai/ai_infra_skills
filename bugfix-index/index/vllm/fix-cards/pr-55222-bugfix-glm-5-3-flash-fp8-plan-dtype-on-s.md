# PR #55222: [Bugfix] GLM-5.3-Flash: fp8 plan dtype on SM90 sparse MLA, and right-size the indexer prefill workspace

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#55222](https://github.com/vllm-project/vllm/pull/55222) |
| 作者 | drakosha |
| 合并日期 | 2026-09-29 |
| labels | bug, ready, deepseek, nvidia, glm |
| 热度 | 👍 0 · 💬 45 |
| 分类 | Quantization |
| bug 类型 | general |

## 一句话结论
修 SM90 sparse MLA fp8 KV 的 plan dtype(uint8→e4m3fn),并把 indexer prefill workspace 按 compress_ratio 缩小,省 5+ GiB 显存。

## 问题现象
GLM-5.3-Flash 在 #53906 合入后的 main 上有两个独立问题。其一:`--kv-cache-dtype fp8` 时 SM90 sparse MLA 后端无法启动,报 `ValueError: MLA kv_data_type torch.uint8 is not supported`,服务器直接起不来。其二:indexer prefill workspace 尺寸单位算错,在 `--max-model-len 1048576`、`index_kpool = 4` 时,memory profile 阶段每 GPU 锁定约 5.16 GiB 显存,直接吃掉 KV cache 预算(由 @nood-co1 在 #53906 报告)。

## 根因
正文披露:问题一,fp8 KV cache 以 `uint8` 存储、`run()` 把它视作 `float8_e4m3fn`,但 `plan()` 被传入的是存储 dtype,FlashInfer wrapper 拒绝 `torch.uint8`。问题二,`get_max_prefill_buffer_size()` 返回的是 token 数(`max_model_len * 40` 条、每条 132 字节),而该 indexer 的 KV 是 pool 粒度(spec 携带 `compress_ratio == index_kpool`),`split_indexer_prefill_chunks()` 接收的是压缩后的序列长度;`deepseek_v4/attention.py` 在同一调用点已经做了除法,GLM-5.3 侧漏了,导致 workspace 按未压缩粒度分配。

## 修复方案
两处独立修改:
1. `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py` 的 `FlashInferMLASparseSM90Builder.__init__`:构造 `_SM90State` 前把 uint8 存储映射为逻辑 dtype:

```diff
+        kv_plan_dtype = (
+            torch.float8_e4m3fn
+            if kv_cache_spec.dtype == torch.uint8
+            else kv_cache_spec.dtype
+        )
```

2. workspace 右移:`vllm/models/glm5next/common/attention.py` 中 `max_total_seq_len = get_max_prefill_buffer_size(vllm_config) // self.index_kpool`;`vllm/v1/attention/backends/mla/indexer.py` 中 `max_prefill_buffer_size` 同样除以 `compress_ratio`(并把 compress_ratio 的计算上移到构造早期)。同时扩充两处单测覆盖 fp8 dtype 与压缩粒度预算。

验证:2x H200 NVL、TP2、`--kv-cache-dtype fp8`、`--max-model-len 1048576`:fp8 修复后 `FLASHINFER_MLA_SPARSE_SM90` 被选中,KV pool 从 bf16 的 2,532,577 tokens 增至 4,503,445;workspace 修复后可用 KV 显存 29.44→32.06 GiB,KV cache 4,503,445→4,904,494 tokens(+8.9%)。两个修复分别在生产环境自 2026-08-29 / 2026-09-01 运行。

## 影响与回归风险
- 影响窗口:两个问题均在 #53906 合入后的 main 上存在(正文指认);fp8 修复影响窗口内 SM90 + fp8 组合完全不可用。
- 回归面:改动一仅影响 SM90 sparse MLA 的 plan dtype 选择,bf16 路径映射后不变;改动二把 indexer prefill 预算与 chunker 的压缩行数单位对齐,DeepSeek 侧调用点本已如此(等价对齐),正文回归测试(needle 1M 提示 4/4、32 并发 118k 全过、驱逐循环复测)未见指标退化。若其他模型路径隐式依赖未压缩预算,行为会收紧,但正文检索确认无其他调用点触碰。

## 关联
- [#55221](https://github.com/vllm-project/vllm/issues/55221)(本 PR 修复的 issue)
- [#53906](https://github.com/vllm-project/vllm/pull/53906)(引入问题的 PR)
- [#51252](https://github.com/vllm-project/vllm/pull/51252)(相关:DeepSeek 侧 chunk 预算除法)
- [#54743](https://github.com/vllm-project/vllm/pull/54743)(相关但不同文件不同故障)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
