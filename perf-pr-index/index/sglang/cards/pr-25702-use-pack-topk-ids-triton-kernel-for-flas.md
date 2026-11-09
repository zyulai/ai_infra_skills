# PR #25702: Use pack topk ids triton kernel for flashinfer_trtllm_routed

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @trevor-m |
| 合并 | 2026-06-17 |
| 分类 | Attention / Kernel (tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/25702 |

## 一句话结论

flashinfer_trtllm_routed MoE 后端的 topk ids 打包从逐 tensor PyTorch 位运算换成 flashinfer_mxfp4 已有的 fused Triton kernel,MiniMax-M2.7 TP8 上 Mean TPOT 降 4.4%。

## 动机

`flashinfer_trtllm_routed` 后端在调用 trtllm routed MoE 前,需要把 `topk_ids`(int)与 `topk_weights`(bf16)打包成 FlashInfer 的 int32 格式(`(ids << 16) | weights.view(int16)`)。此前 `flashinfer_trtllm_routed` 路径用 `_pack_topk_for_flashinfer_routed` 的 PyTorch 实现(`topk_ids.to(torch.int32)`、`topk_weights.to(torch.bfloat16)`、移位与或),产生多个中间张量与额外 kernel launch;而 `flashinfer_mxfp4` 后端已有做同样事情的 fused Triton kernel(`PackTopkIds`)。两个后端为同一格式维护两份打包实现,其中一份明显更慢。

## 优化点

机制上是实现复用而非新 kernel:删除 `_pack_topk_for_flashinfer_routed`,fp8、fp4、bf16 三条 routed MoE 路径的 `packed_topk_ids` 一律改为调用 `mxfp4_flashinfer_trtllm_moe` 模块的 `PackTopkIds.execute(topk_ids, topk_weights)`,即 flashinfer_mxfp4 使用的 fused Triton kernel,单次 kernel 完成权重 bf16 转换、id 转 int32 与移位或打包,消除多条 PyTorch 中间张量链。打包格式与语义不变(仍是 FlashInfer 的 int32 [id<<16 | bf16 weight] 布局),调用方接口不变。

## 关键改动

```text
python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py: 删除 _pack_topk_for_flashinfer_routed,fp8/fp4/bf16 三条 routed 路径改用 PackTopkIds.execute
python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py: PackTopkIds(fused Triton kernel,已有,未改动)
```

```python
-            packed_topk_ids = _pack_topk_for_flashinfer_routed(
-                topk_ids=topk_output.topk_ids,
-                topk_weights=topk_output.topk_weights,
-            )
+            packed_topk_ids = PackTopkIds.execute(
+                topk_output.topk_ids, topk_output.topk_weights
+            )
```

## 性能声明

MiniMax-M2.7、`--moe-runner-backend flashinfer_trtllm_routed --dtype bfloat16 --tp 8`:

- bench_serving(random 1024 输入/1024 输出、range-ratio 0.8、80 prompts、max-concurrency 8、request-rate inf):Mean TPOT 8.96→8.58 ms,即并发 8 下 Mean TPOT 改善 4.4%;Mean ITL 同步 8.96→8.58 ms;output token throughput 856.63→894.31 tok/s;Mean E2EL 8363.28→8012.82 ms;TTFT 基本不变。
- GSM8K platinum(8-shot、1209 题、parallel 1209):Accuracy 0.950→0.959,Invalid 0.000,Latency 19.906→19.244 s,output throughput 6234.469→6417.248 tok/s。

## 验证与回归风险

- 验证手段为改前/改后的 GSM8K 与 serving benchmark 对照,未新增单元测试(checklist 各项未勾选)。
- 回归风险:三条 routed MoE 路径(fp8 block-scale、fp4 block-scale、bf16 routed)共享同一替换,打包格式逐位不变,风险集中在 `PackTopkIds` 对非 MXFP4 模型 shape/dtype 的兼容性;两个 backend 的 CI(Base、Extra)显示失败,无法从 pack 确认最终 CI 结论。

## 关联

- 同区域:flashinfer_mxfp4 MoE 后端(PackTopkIds 出处);MoE routing/topk 相关 #40708、#39931。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
