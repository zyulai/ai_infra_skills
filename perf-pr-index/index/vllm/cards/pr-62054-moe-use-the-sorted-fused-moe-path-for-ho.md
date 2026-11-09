# PR #62054: [MoE] Use the sorted fused-MoE path for host-resident expert weights

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mushroomcarbon |
| 合并 | 2026-10-22 |
| 分类 | MoE \| tier: major |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/62054 |

## 一句话结论

UVA host 权重改走 sorted 路径，RTX 4090 上 30B 并发 4 的 output 吞吐 +15–19%。

## 动机

PR 正文 Overview：小 batch（`num_tokens * top_k * 4 <= num_experts`，例如 Qwen3 MoE 最多约 4 tokens）时，Triton fused-MoE 路径会跳过 `moe_align_block_size`、给每个 (token, expert) 对分配自己的 block（源自 #29354、#39016），权重在 device memory 时这样能省掉对齐 kernel。但启用 UVA weight offloading（`--offload-backend uva --cpu-offload-params experts`）后 expert 权重是从 host memory 原地读取，被多个 token 路由到的同一个 expert 会按每个 token 各经 PCIe 读一次。

## 优化点

当 expert 权重带有 UVA offload 标记时改走 sorted path，使每个被路由的 expert 每 step 只读一次；device-resident 权重仍保留 naive path。实现上 `_prepare_expert_assignment` 新增 `weights_host_resident`（默认 False）用于禁用 naive block assignment；`TritonExperts.apply` 与 legacy `fused_experts_impl` 传入 `_weights_host_resident(w1, w2)`——当任一权重带 `_vllm_is_uva_offloaded` 标记即返回 true（UVA offloader 设置，`device_loading_context` 在 `process_weights_after_loading` 后重设）；取 "either" 是因为 UVA 按 parameter offload，budget cut-off 可能使 w13 在 host 而 w2 在 device。这是 `moe_forward` custom op 内的 per-forward getattr，结果只依赖参数，故 CUDA-graph capture per size 不受影响；之所以不在 kernel 构建时决定，是因为 `replace_parameter` 安装的新 Parameter 不带标记、setup 时读会是 False。其他 MoE 路径不受影响：Marlin 与 `TritonWNA16Experts` 始终 align，prefetch offloader 先拷到 device 且不设标记；两个 model-specific 调用者（`models/k2_horizon.py`、`models/dots3_note/nvidia/vision_moe.py`）不传该 flag，保持原行为。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/fused_moe.py: 新增 _weights_host_resident()，_prepare_expert_assignment 增加 weights_host_resident 参数，naive_block_assignment 加 not weights_host_resident 条件
vllm/model_executor/layers/fused_moe/experts/triton_moe.py: TritonExperts.apply 传入 _weights_host_resident(w1, w2)
tests/kernels/moe/test_moe.py: 新增 test_host_resident_weights_take_sorted_path
```

```text
+def _weights_host_resident(*weights: torch.Tensor) -> bool:
+    """True when any of the expert weights is read from host memory in place,
+    as UVA weight offloading (``--offload-backend uva``) leaves them."""
+    return any(getattr(w, "_vllm_is_uva_offloaded", False) for w in weights)
...
     naive_block_assignment = (
         expert_map is None
+        and not weights_host_resident
         and num_tokens * top_k_num * 4 <= global_num_experts
```

## 性能声明

RTX 4090（PCIe 4.0 x16）、natural-text chat requests、128 output tokens：Qwen3-30B-A3B-Instruct-2507-FP8 单 GPU offload 16 GiB experts，"+7% / +10% / +15–19% output throughput at 2 / 3 / 4 requests in flight"，1、8、16 in flight 不变；Qwen3-235B-A22B-Instruct-2507-FP8 TP8、每 GPU offload 16 GiB，"+7% / +10% / +18% at 2 / 3 / 4"，1 不变。Base 为 vLLM 0.31.0 wheel（两个被改文件与 base commit 字节一致），仅在 CUDA 上测量，ROCm 的 Triton 路径走同一分支；无 offloading 时无变化（30B TP2 权重全在 VRAM）。Quality：GSM8K（lm-eval 0.4.13、5-shot、greedy、前 250 题、4 in flight 使 decode 走改动路径），30B offload 16 GiB 下两个 build 均为 0.888 flexible / 0.880 strict exact match，且 250 条响应逐字节一致。E2E base vs PR（一个 `vllm serve`，natural-language chat 32 topics × 8 tasks、temperature 0.7、固定 seed、ignore_eos）：30B 1 GPU 16 GiB host 下 in flight 2 22.2 → 23.8 tok/s（+7%）、3 22.6 → 24.9–25.0（+10%）、4 23.1 → 26.5–27.4（+15–19%）；235B TP8 下 2 +7%、3 19.0 → 20.9（+10%）、4 19.7 → 23.2–23.3（+18%）；30B TP2 全在 VRAM 时 1–4 within 0.3%。`vllm bench serve`（sharegpt、sharegpt-output-len 128、ignore_eos、seed 1、30B 单 GPU offload 16 GiB、两臂各两次）：c2 21.3–22.2 → 23.4–23.6（+8%）、c3 +15%、c4 22.2–23.0 → 25.6–26.7（+16%）、c1 0%。

## 验证与回归风险

`pytest tests/kernels/moe/test_moe.py -k "host_resident_weights or naive_block_assignment"` 在 RTX 4090 上 44 passed（原始 base 与 rebase 到 current main 后各跑一次）；新增 `test_host_resident_weights_take_sorted_path` 在 1/2/4 tokens 下用 UVA-mapped 权重（device 权重作对照）分别走 modular `TritonExperts` 与 legacy `fused_moe`，断言 `moe_align_block_size` 恰在 host-resident 时被调用，并与 `torch_moe` 比对（atol=rtol=2e-2），无此改动时其 6 个 host-resident cases 失败。所有适用 pre-commit hook 通过（ruff、mypy 3.10–3.13、check_torch_cuda 等）。回归风险：仅 UVA offloading + 小 batch 场景生效，其他 MoE 路径与 device-resident 权重不变；235B TP8 greedy 输出本身随启动变化（三次 unpatched 三种、三次 patched 两种），无法据此区分 builds。

## 关联

正文 Related 明确列出：naive path 源自 #29354 与 #39016；#34920（closed, stale）曾提议从 config 调节；#55268（open）按 batch size warm up Triton MoE 变体；#57215（open）在 XPU 上保持 experts offloaded；#56177（open）为 shared GPU expert pool；RFC #38256 讨论 offloaded MoE expert caches；#37190、#57794、#57943 决定哪些 experts 留在 VRAM（本 PR 与之独立）。正文称搜索（2026-10-07、2026-10-16）未发现其他实现该改动的 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
