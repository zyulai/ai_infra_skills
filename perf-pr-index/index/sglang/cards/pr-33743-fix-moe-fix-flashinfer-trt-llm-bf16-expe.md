# PR #33743: [Fix] moe: Fix flashinfer TRT-LLM BF16 expert weight reload on refit

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Kh4L |
| 合并 | 2026-10-27 |
| 分类 | MoE(tier: major) |
| Label | bug, documentation, quant, run-ci |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 全量保留(3 个文件:unquant.py、weight_updater.py、新增测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/33743 |

## 一句话结论

修复 flashinfer_trtllm(_routed) BF16 MoE 权重在 refit 重载时 block layout 逆变换缺失导致的形状报错与静默错值,属 RL 权重更新正确性修复。

## 动机

BF16 MoE expert 权重在冷加载时会被转换成 FlashInfer TRT-LLM BlockMajorK 布局,但此前的逆还原(`maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load`)只对 `flashinfer_trtllm_routed` backend 生效,而 `flashinfer_trtllm` 与 `flashinfer_trtllm_routed` 都使用该变换。于是对非 routed backend 做后续 checkpoint-format 拷贝会报 "The size of tensor a (64) must match the size of tensor b (2048)"。且只恢复形状不够:未变化的 expert 值也必须在下一次 pack 前还原到 checkpoint 顺序,否则反复 refit 会静默打乱权重。Fixes #27787。

## 优化点

机制改动有四点:(1) 逆还原的判定从 `is_flashinfer_trtllm_routed()` 改为 `use_flashinfer_trtllm_moe`,两类 backend 一起覆盖;(2) 新增 `_restore_trtllm_bf16_canonical_layout`,原地还原 canonical 形状与值——用 `torch.argsort(permute_indices)` 求逆置换,并先校验置换是双射,非双射时直接 raise 而不是静默错行;(3) 实现 `restore_weights_before_loading`,使 `begin_weight_update` 对绕过 model weight loader 的直接 tensor 写与 P2P/RDMA 写也先还原两个 expert 参数,loader 侧还原保留给 disk reload;(4) post-load pack 加 shape guard:canonical 权重转 kernel 布局,已是 blocked 的保持不变;`end_weight_update` 每 session finalize 一次而非每 bucket 一次。

## 关键改动

```text
python/sglang/srt/layers/quantization/unquant.py: pack/restore 对偶实现 + 双射校验 + restore_weights_before_loading
python/sglang/srt/model_executor/model_runner_components/weight_updater.py: model_load_weights 前先 restore(并行拷贝前恢复)
test/registered/unit/layers/quantization/test_flashinfer_trtllm_bf16_moe_reload.py: 新增 569 行 CPU 单测(10 个方法)
```

## 性能声明

正文明确:"No speed benchmark is claimed. Forward kernels are unchanged." 即无性能数字,本 PR 修的是表示正确性而非速度;canonical restore 与 pack 的开销每个 weight-update session 支付一次,disk reload 保持原 load/postprocess 生命周期。PR 未注明测试条件(无基准)。

## 验证与回归风险

新增 CPU 注册单测覆盖布局往返、bucketed/flattened/distributed 更新、mixed canonical/blocked 参数、无效置换拒绝、LoRA 遍历、连续三次 loader-free refit(direct tensor 与模拟 P2P flat copy);2026-10-05 验证记录:10 方法 + 5 格式子测 + 22 相邻 session/cache-guard 方法全过,对照实验(移除生产 restore hook)可复现三个 pre-fix TP-rank 失败。局限:FlashInfer 的 layout 入口在测试中是 order-changing CPU stand-in,不覆盖 on-device kernel accuracy、真实 RDMA 传输与端到端 RL 验证;CI Extra 与 AMD ROCm 为红。影响路径:`flashinfer_trtllm` / `flashinfer_trtllm_routed` BF16 MoE 的权重更新/重载流程。

## 关联

Fixes #27787。同区域:weight-update session 机制(与 compressed-tensors W4A16 Marlin 的 restore 并列,见 docs 更新);同分类 MoE 其他 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
