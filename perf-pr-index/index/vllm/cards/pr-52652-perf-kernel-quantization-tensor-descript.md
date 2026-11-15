# PR #52652: [Perf][Kernel][Quantization] Tensor-descriptor path for int4 fused_moe_kernel_gptq_awq (XPU default, sm100 opt-in)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @afierka-intel |
| 合并 | 2026-11-13 |
| 分类 | MoE(tier: major) |
| Label | intel-gpu, ci/build, quantization |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/52652 |

## 一句话结论

为分组 `int4_w4a16` 权重的 `fused_moe_kernel_gptq_awq` 增加 tensor-descriptor(TD)路径(XPU 默认、CUDA sm100+ 经 `VLLM_TRITON_USE_TD=1` opt-in):Arc Pro B60 kernel 在 M 41–4096 提速 +28…+31%,B70 上 E2E prefill-heavy +10–11%,单流 decode 从 −3.6% 回正(+0.8%)。

## 动机

#42436 已为 `fused_moe_kernel`(非量化权重)加入 TD 路径(RFC #42545),但分组 int4 `fused_moe_kernel_gptq_awq` 仍走 pointer path;本 PR 把同样的 TD 设计搬到该 kernel,并按 RFC #42545 与 #42436 的先例在同一步完成 XPU 默认开启。

## 优化点

- **TD 路径**:A 用 `tensor_descriptor.gather` 收集;packed int4 B 以字节加载、用 `tl.interleave` 解包(偶 k 取低 nibble、奇 k 取高 nibble,转置后与 pointer 布局一致);B descriptor 在 kernel 内构建(与 `fused_moe_kernel` 的 TD 路径一致)。`int8_w8a16` 保持 pointer path。
- **何时启用**:`VLLM_TRITON_USE_TD` 解析为开(unset:仅 XPU;`1`:也含 CUDA sm100+)且 launch 落到 Triton WNA16 kernel;XPU 上需 `--moe-backend triton`(默认 serving 选 `XPUExpertsWNA16`);sm90 与 ROCm 始终 pointer path。
- **回落 pointer path 的条件**:`VLLM_BATCH_INVARIANT=1`(TD 与 pointer 可能 round 不同,所有 batch size 必须同路径);`BLOCK_SIZE_M < 8`(`gather()` 最小行数);`BLOCK_SIZE_M == 32`(TD 在 Arc Pro B60 上 −42%,其他设备未测,故全线回避);单 token batch(两个 GEMM 都回——down projection 现在拿到 layer 的 token 数 `num_input_tokens`,因为它的 A 有 tokens × top_k 行);`K % BLOCK_SIZE_K != 0`(一次性 warning);`BLOCK_SIZE_K < 32`(packed-byte descriptor 需 ≥16 字节内层块)。
- 默认开启 XPU:`VLLM_TRITON_USE_TD=0` 可关;`envs.py` 注释列出所有消费方。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/fused_moe.py: fused_moe_kernel_gptq_awq 增加 USE_TD constexpr 与 TD 加载分支;invoke_fused_moe_wna16_triton_kernel 增加 use_td 门控与 K-tail 回退;dispatch_fused_moe_kernel / fused_experts_impl 传 num_input_tokens
vllm/model_executor/layers/fused_moe/experts/triton_moe.py: down projection 调用传 num_input_tokens=num_tokens
vllm/model_executor/layers/fused_moe/utils.py: resolve_moe_use_td / warn_if_moe_use_td_ineffective 文档更新
vllm/envs.py: VLLM_TRITON_USE_TD 注释更新(消费者清单与 sm100 opt-in 说明)
tests/kernels/moe/test_moe.py: test_fused_moe_wn16 增加 use_td 参数与 launch spy;新增 forced config、TritonWNA16Experts、TD gate 参数化测试
tests/kernels/moe/utils.py: modular_triton_fused_moe 支持注入 experts_cls
```

```python
+    if USE_TD:
+        m_td = num_valid_tokens // top_k
+        a_desc = tl.make_tensor_descriptor(
+            base=a_ptr,
+            shape=(m_td, K),
+            strides=(stride_am, stride_ak),
+            block_shape=(1, BLOCK_SIZE_K),
+        )
+        # Padding entries (offs_token == num_valid_tokens) gather row m_td,
+        # one past the end: the descriptor fills it with zeros and those rows
+        # are masked at the store.
+        gather_idx = (offs_token // top_k).to(tl.int32)
+        # B is packed 2 nibbles/byte along K: (E, N, K // 2) bytes.
+        b_desc = tl.make_tensor_descriptor(
+            base=b_ptr + off_experts * stride_be,
+            shape=(N, K // 2),
+            strides=(stride_bn, stride_bk),
+            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K // 2),
+        )
```

## 性能声明

- **Kernel 延迟**(`fused_experts`,Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 shapes E=60,N=1408,K=2048,top-k 4,group 128,默认 tiles,7 轮交错中位;正值 = TD 更快;pointer-vs-pointer 噪声 ≤0.52%):
  - Arc Pro B60:M 2–20 +13…+17%(本 head 实测 M=2,4,8,16 为 +13.8%/+12.9%/+14.5%/+13.5%),M 41–4096 +28…+31%;M 1 两 GEMM 均 pointer(−0.1%,噪声),M 21–40 pointer(TD −42%);pointer path 与 `main` 差 ≤0.7%。
  - Arc Pro B70:M 2 −4.8%、M 3 +8.0%、M 4–16 +10…+13%;M 1 pointer;M 21–40 与 41–4096 未测。
  - NVIDIA B200(opt-in):M ≥ 128 +57…+80%;M ≤ 90(此处 `M·top_k/E ≤ 6`)由 `fused_experts` 送 CUDA WNA16 kernel,该 regime 单测覆盖但未 benchmark。
- **E2E**(Arc Pro B70,`vllm bench sweep serve`,graph mode,random dataset,5 reps 交错,Mann-Whitney):prefill-heavy(前一 head a0b1f2262a)Qwen1.5-MoE GPTQ +10.4%、Qwen3-30B-A3B w4a16 +11.4%;本 head Qwen3-30B mc4 +6.0%(TPOT 18.52 → 17.51 ms);batch1 +0.8%(不再回归,前 head 曾 −3.6%);Qwen1.5 mc4 −1.6%(mc4 decode 多在 4 tokens,该点 B70 kernel 层 TD 快 10.5%,E2E 回归原因未查明);Qwen1.5 batch1 TPOT 9.22 → 9.22 ms。
- **B200**(前一 head,基线为 Triton pointer path):prefill_heavy +235%、decode_heavy +161%、mixed +188%、mid32 +179%、mc4 +1.5%、batch1 +2.5%。
- **准确度**:gsm8k(`lm_eval --tasks gsm8k --num_fewshot 5`,全测试集)B70 Qwen1.5-MoE 0.4602/0.4716/0.4655(TD=0/TD=1/TD=0 repeat)、B70 Qwen3-30B 0.8772/0.8711/0.8696、B200 Qwen3-30B 0.8848/0.8802/0.8795——差异在 A/A spread 内,只能排除大于约 1.5–2 点的下降。

## 验证与回归风险

- `pytest -v tests/kernels/moe/test_moe.py -k "wn16 or wna16"`:Intel Arc Pro B60(IGC 2.38.2 / compute-runtime 26.27,triton 3.8.0)608 passed/0 failed;NVIDIA B200(sm100)480 passed/128 skipped(shape 被 `fused_experts` 送 CUDA WNA16 kernel);H200 NVL(sm90,无 TD)406 passed/202 skipped(TD legs)。`main` 在 B200/H200 上通过同样 384 个既有 case。
- `test_fused_moe_wn16` 的 use_td leg 对照 pointer path(容差约 6 bf16 ULP;B200 上曾在旧容差失败后放宽)与 reference,launch spy 断言每个 GEMM 走的路径;`test_wna16_triton_launcher_td_gate` 用 fake kernel 验证全部回落决策(B60 上回退 token gate 会让 down-projection case 与单 token case 失败)。
- 风险:TD legs 无任何 CI job 覆盖(仅上述本地运行);TD 容差曾因 B200 失败放宽;`set_triton_allocator` 在 XPU 上可能不被调用且注册可破坏 torch.compile `split_module`(#61583),graph-mode E2E 未踩到;E2E/精度运行用的 XPU 栈为 IGC 2.34.4/compute-runtime 26.18,`main` 栈(2.38.2/26.27)在模型级崩溃 int4 Triton MoE(与 TD 无关);与 #52577 的 pointer-path B load mask 存在 textual conflict,`K % BLOCK_SIZE_K` 回退会把 ragged-K launch 送进同一个 unmasked load,#52577 应先合;#56154/#56788/#57558 也在改同一 launch site(BLOCK_SIZE_M=32 的 tuned config 将保持 pointer path)。未覆盖:TP > 1、EP、int8、AWQ 与 Quark checkpoints。

## 关联

- #42436(`fused_moe_kernel` 的 TD 路径,镜像来源)、RFC #42545;#63820(sm90 fp8 的 host TensorDescriptor,不读 `VLLM_TRITON_USE_TD`);冲突/叠加:#52577、#56154、#56788、#57558。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
