# PR #32207: 【NPU】Support EAGLE when PP enabled in prefill nodes

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @cen121212 |
| 合并 | 2026-09-08 |
| 分类 | Speculative Decoding |
| Label | documentation, quant, amd, dependencies, lora, Multi-modal, deepseek, speculative-decoding, hicache, sgl-kernel, blackwell, npu, run-ci, diffusion, model-gateway, apple-silicon, jit-kernel, unified-radix-cache, memory-pool |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用(正文称主改动在 eagle_worker_v2.py、ascend/conn.py、scheduler.py、scheduler_pp_mixin.py 等文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/32207 |

## 一句话结论

让 NPU prefill 节点在 PP 下也跑 EAGLE/MTP 并把 draft KV 传给 decode 节点,MTP 接受长度 2.97→3.28、TPOT 32.9→28.5 ms。

## 动机

PP 与 MTP(EAGLE)此前互不兼容。PD 分离部署中 decode 节点通常要开 MTP;若 prefill 节点开 PP 却不开 MTP,decode 侧会缺失部分 draft_kv,经验实测 decode 侧 MTP 接受率因此下降。因此需要让 prefill 节点在 PP 开启时也支持 EAGLE,并把 draft KV 一并传输。

## 优化点

配置层:`validation_hook.py` 对 NPU 放宽 PP 断言,允许 `speculative_algorithm=EAGLE`(非 multi-layer)且仅限 `disaggregation_mode=prefill` 的节点,非 NPU 保持原禁用;`layer_setup.py` 在 NPU 上跳过 `_assert_pp_mtp_compat`。KV 传输层:`KVArgs` 的 `total_kv_layers` 拆为 `hidden_kv_layers` 与 `draft_kv_layers`;`ascend/conn.py` 的 `get_mla_kv_ptrs_with_pp` 在 PP 末 rank 上把 draft KV 层的指针追加到目标切片中(target KV 按每组的 hidden 层偏移切,draft KV 单独按 `kv_buf_groups * hidden_kv_layers` 起始偏移切);`prefill.py` 中 NPU 仅在 PP 末 rank 准备 draft_kv_pool。模型/索引层:`deepseek_v2.py` 在 NPU 末 rank 也建 `embed_tokens`;NPU DSA 相关的 rotary sin_cos_cache 刷新条件从 `layer_id == 0` 改为 `layer_id == start_layer`,适配 PP 分段后首层非 0 的情形;`deepseek_nextn.py` forward 接受 `PPProxyTensors`。

## 关键改动

```text
python/sglang/srt/arg_groups/validation_hook.py: NPU 下 PP+EAGLE(非 multi-layer)准入断言
python/sglang/srt/disaggregation/ascend/conn.py: PP 切片时追加 draft KV 指针传输
python/sglang/srt/disaggregation/base/conn.py / utils.py: KVArgs 增加 draft_kv_layers
python/sglang/srt/disaggregation/prefill.py: 仅 PP 末 rank 挂 draft_kv_pool
python/sglang/srt/hardware_backend/npu/...: start_layer 适配与 embed_tokens
```

```python
transfer_draft_kv = get_pp_group().is_last_rank and draft_kv_layers
if transfer_draft_kv:
    end_layer = start_layer + src_layers - draft_kv_layers
else:
    end_layer = start_layer + src_layers
# draft kv
if transfer_draft_kv:
    for i in range(kv_buf_groups):
        layer_offset = i * draft_kv_layers + kv_buf_groups * hidden_kv_layers
        sliced_dst_kv_ptrs.extend(dst_kv_ptrs[layer_offset : layer_offset + draft_kv_layers])
```

## 性能声明

prefill cp4pp4(无 MTP)+ decode dp16+mtp3,ais_bench 256 请求(input ~4.19M tokens,output 1024/req):before 接受长度 2.97(98/32.9),TPOT mean 32.9 ms;after 接受长度 3.28(93.7/28.5),TPOT mean 28.5 ms(约 -13%)。after 侧 prefill token throughput 140.60 token/s、output token throughput 399.54 token/s(before 分别为 155.10/420.75,prefill 侧因额外跑 MTP 有回落)。GSM8K 1319 题:Accuracy 0.933,Invalid 0.000。模型/硬件细节(具体 NPU 型号、模型名)PR 未注明。

## 验证与回归风险

验证为 GSM8K e2e 与 ais_bench 压测前后对比;未见新增单测,checklist 各项未勾选。风险面:断言放宽仅限 NPU + prefill 节点 + 非 multi-layer EAGLE;KV 传输路径改动集中在 NPU disaggregation(conn/base/utils/prefill),非 NPU 的 `total_kv_layers` 语义被改名重建,依赖该字段的代码需同步;DSA indexer/attention 的层索引判断改为 start_layer 对比,影响 NPU DSA 模型在 PP 分段下的行为。pack 显示 CI Extra 与 AMD ROCm 7.2 失败。

## 关联

同区域:同批 #32196(PD decode 下 EAGLE DP 图与 token 元数据一致性,非 NPU 侧的 EAGLE/PD 适配)。前序/后续 PR 素材未提供。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
