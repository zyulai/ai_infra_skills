# PR #65059: [Bugfix][Kernel] Only set PDL launch attr for MoE top-k gating on SM90+

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @srkim-business |
| 合并 | 2026-11-02 |
| 分类 | Speculative Decoding \| tier: major |
| Label | bug, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/65059 |

## 一句话结论

仅在 SM90+ 为 MoE top-k gating 设 PDL launch attr，修复 pre-SM90 CUDA graph 回放变慢（A100 TP4 decode 快 6.5-13.6%）。

## 动机

Fixes #65055。自 #59737 起 `topkGating` 在所有 CUDA arch 上都带 PDL attribute 启动；pre-SM90 GPU 不能用 PDL，但 stream capture 仍把该 attribute 记为 CUDA graph 中的 programmatic edge，回放这些 graph 变慢。本 PR 只在设备 SM90+ 时设置该 attribute：SM90+ 保持 #59737 行为，老 GPU 不带 attribute 启动（测得与 v0.32 的 `<<<>>>` launch 一致）。standalone 效应在 sm80/sm86/sm89 测得，端到端数字仅 A100。

## 优化点

根因：pre-SM90 上 attribute 对 eager launch 无作用，但 capture 记为 programmatic edge（sm80 上 cudaGraphGetEdges 每 launch 计 1 条），A100 每条 edge 每次 graph replay 加约 1.1 µs GPU 时间。改动在 csrc/moe/topk_softmax_kernels.cu 的 topkGatingLauncherHelper 加函数内 `static const bool use_pdl = get_device_prop()->major >= 9;`（每个模板实例化缓存一次，eager launch 无每调用开销），`config.numAttrs = use_pdl ? 1 : 0`。Python launch sites 已用 current_platform.is_arch_support_pdl() 门控。其它 C++ launchers（topk_softplus_sqrt_kernels.cu、per_token_group_quant.cu）仍在所有 arch 设置 attribute，本 PR 为保持可 cherry-pick 的小改动不动它们。

## 关键改动

```text
csrc/moe/topk_softmax_kernels.cu: topkGatingLauncherHelper 按 SM90+ 门控 PDL attribute（pack 的 diff 仅含此 1 个源文件）
```

```diff
@@ -627,6 +627,8 @@
 #ifndef USE_ROCM
+    // Pre-SM90 cannot use PDL, but the attribute still adds a programmatic edge to captured CUDA graphs.
+    static const bool use_pdl = get_device_prop()->major >= 9;
     cudaLaunchConfig_t config{};
@@ -635,7 +637,7 @@
     config.attrs = &attr;
-    config.numAttrs = 1;
+    config.numAttrs = use_pdl ? 1 : 0;
```

## 性能声明

4× A100、TP4 + EP + MTP（私有 NVFP4 checkpoint of Qwen3.8-Flash-Next）：18 个 prompt×concurrency（1-16）cells 中 17 个每请求 decode 快 6.5-13.6%、第 18 个 2.1%；concurrency 16 时 prose 74.3→80.1、code 88.7→94.8 tok/s。Standalone CUDA（无 vLLM）：attribute 每 launch 加 1 条 programmatic edge，A100(sm80) graph replay 128.7→181.8 µs（+41%、每 edge 1.1 µs）、RTX A4000(sm86) 83.6→+29.9（+36%、0.62 µs）、RTX 4090(sm89) 79.0→+30.0（+38%、0.63 µs）；不带 attribute 时 cudaLaunchKernelEx 与 `<<<>>>` 一致。1× A100 的 router GEMM → topk_softmax CUDA graph replay 链：Qwen3.8-Flash-Next / Qwen3-30B-A3B / OLMoE 在 1/16/48 tokens 下 6.9-9.0% faster（如 Qwen3-30B-A3B 592/672/688 → 542/620/636 µs，-9.0/-7.9/-8.0%）。1× A100 `vllm bench latency` batch 1：OLMoE-1B-7B 2.950→2.932 ms（-0.7%）、Qwen3-30B-A3B-GPTQ-Int4 5.805→5.722 ms（-1.2%），batch 16 within noise。GSM8K 5-shot Qwen3-30B-A3B-GPTQ-Int4 85.90%（base）vs 86.13%（fix），kernel body 相同。SM90+ 按构造不变、未测。测试条件：A100-PCIE-40GB(sm80)、driver 590.48.01、CUDA 13.0、torch 2.14.0+cu130、vLLM v0.33.0（4ba787ec9），模型 Qwen3.8-Flash-Next(NVFP4)/Qwen3-30B-A3B/OLMoE-1B-7B/Qwen3-30B-A3B-GPTQ-Int4，workload vllm bench latency、CUDA graph replay、FULL_DECODE_ONLY、MTP 2 draft tokens、TP4+EP。

## 验证与回归风险

`pytest tests/kernels/moe/test_fused_topk.py` on A100：base 864 passed、fix 864 passed；GSM8K 5-shot 1319 题（Qwen3-30B-A3B-GPTQ-Int4 85.90→86.13%、OLMoE 9.93→10.01%、invalid 0-0.8%）；CI 的 L4(sm89) "MoE Kernels" step 覆盖不带 attribute 的 launch，无新测试（无功能变化，回归检查需 graph-edge inspection / timing）。回归风险：SM90+ 路径按构造不变但未测；sm75 及更老、A100 以外 GPU 的端到端、完整 main build（仅重编 `_moe_C` 覆盖 v0.33.0）未覆盖；其它 C++ launchers 仍在所有 arch 设 attribute；static 假设进程内所有 GPU 同 major arch；TP4 增益未完全定位（48 edges × 1.1 µs 约 0.05 ms/forward，而 decode step 29.5→27.3 ms）；正文称这是对 v0.32.0 的性能回归，希望 cherry-pick 进 v0.33.x。

## 关联

Fixes #65055；#59737（引入该 PDL attribute，#59737 描述里本就写 "Enable PDL for the consumer on SM90+"）、#62912（open，改同文件 top-k selection 且保留 numAttrs=1，后合入者需小 rebase）；正文另引用 3b05805d4（#59737 前最后提交）与 v0.33.0 的对照。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
