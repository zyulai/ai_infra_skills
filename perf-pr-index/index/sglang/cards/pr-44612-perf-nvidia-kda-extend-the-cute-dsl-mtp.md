# PR #44612: [Perf] [NVIDIA] KDA: Extend the CuTe DSL MTP verify kernel to SM90

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JackChuang (co-authored-by @HanHan009527) |
| 合并 | 2026-11-16 |
| 分类 | Speculative Decoding(major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/44612 |

## 一句话结论

把融合 CuTe DSL KDA(Kimi Delta Attention)MTP verify 内核从 SM100 扩展到 Hopper SM90:用标量 FP32 数学替换两条 SM100-only 的 packed f32x2 指令,并放宽 gate 到能力 major 9(仅限 CUDA graph 内),使 GLM-5.3-Flash 在 H200 上稳态 decode 吞吐 +2.7–6.5%、Kimi-K3 在 Hopper 不再静默回退 Triton。

## 动机

#32890 加入融合 CuTe DSL KDA verify 内核,#32541/#42905 使其成为 Kimi-K3 / GLM-5-Next 的默认 verify 后端,但内核使用 `cute.arch.mul_packed_f32x2` 与 `cute.arch.fma_packed_f32x2`(每条一次做两个 FP32 乘法/FMA),NVVM 只接受 SM100+(`F32x2 intrinsics are not supported on this architecture` 对 `sm_90a`),且 `_can_run_dspark_cutedsl_mtp` gate 只承认 major 10。结果 SM90 上每个 KDA 层的每次 verify 都静默跑未融合的 Triton 路径(`causal_conv1d_update` + gated delta-rule 递推两个内核),Kimi-K3 启动日志甚至声称已使用 fused kernel 而实际在回退。

## 优化点

- 内核加 `USE_PACKED_FMA` 编译期常量(capability major >= 10 走 packed,SM90 走标量乘加),递推中 4 处调用点(1 处 `mul_packed_f32x2`、3 处 `fma_packed_f32x2`)分支;SM100 代码与 main 逐字一致。
- gate 改读缓存的 `current_arch()`,接受 major 9 与 10;SM90 额外要求 CUDA graph capture 模式(`get_is_capture_mode()`),eager verify 仍走 Triton——eager 下 verify 受 host 启动时间约束,CuTe 启动比 Triton 慢(每层约 680 us vs 300 us host 时间)。
- GLM-5-Next 的 spec 默认 `--linear-attn-verify-backend nv_cutedsl` 在 SM90 上仅当 `--enable-linear-replayssm-spec` 开启时生效。
- 三个既有内核测试改为在 CI `1-gpu-large`(H100)运行;新增 `test_nv_cutedsl_verify_matches_triton` parity 测试与 CPU gate 测试。

## 关键改动

```python
# python/sglang/kernels/ops/attention/kda_decode_mtp.py
+    USE_PACKED_FMA: cutlass.Constexpr[bool],
...
+    USE_PACKED_FMA=cuda_arch.major >= 10,

# python/sglang/srt/layers/attention/linear/kda_backend.py
-        if torch.cuda.get_device_capability()[0] != 10:
+        if cuda_arch is None or cuda_arch.major not in (9, 10):
+            return False
+        # SM90 runs it only in CUDA graphs: an eager verify is bound by the
+        # launcher's host time, and Triton's kernels launch faster.
+        if cuda_arch.major == 9 and not get_is_capture_mode():
             return False
```

## 性能声明

(全部照抄 PR 正文,未独立核实)

- H200, GLM-5.3-Flash, attention DP8 + ReplaySSM, MTP 3/1/4:稳态 decode 吞吐 64 并发 +2.7%(8,492→8,717 tok/s)、128 并发 +6.5%(14,129→15,051);TPOT 7.55→7.32 ms(3.0% lower)与 9.04→8.50 ms(6.0% lower)。负载端 512 请求(63 running/GPU)稳态 TPOT 18.66→16.71 ms(10.5% lower)、节点吞吐 +11.7%;单请求稳态 TPOT 5.1% lower。whole burst 输出吞吐 +1.2–2.5%,median ITL 低 3.3–8.4%。
- H20(8 卡, 同配置):whole-run 输出吞吐 64 请求 +1.7%、128 请求 +1.7–4.8%(MTP 5/1/6 时 +2.5–7.0%)、256 请求 +1.6–5.2%。
- 每 KDA 层内核单独计时(H200, ReplaySSM):1.7–3.0x 快于 Triton(GLM-5-Next 形状 1.67–2.25x,Kimi-K3 Hopper 形状 2.05–2.97x);无 ReplaySSM 时 TP8 形状 N<=16 为 1.27–1.33x,其余 0.97–1.04x。
- 精度:35 个形状输出与 Triton 路径差在 1.03e-2 内;GSM8K 97.08%→97.27%(一个标准误约 0.46 点内)。
- 成本:无新 flag、无内存变化;首次启动一次性 CuTe 编译 18–41 s(ReplaySSM +18 s、TP8 +41 s),后续启动与 main 持平。

## 验证与回归风险

新 parity 测试验证 CuTe 输出与每步 SSM 状态;CPU 测试钉住 gate(SM100 恒 admit,SM90 仅 capture 模式,SM80/SM120 拒绝)与 GLM-5-Next 默认。PR 自测:在 main 上三个内核测试对 sm_90a 编译失败,加本 PR 后四个测试文件全过;CI 560 文件中 14 个失败全部与 main 相同(非本 PR 引入)。风险面:SM90 eager verify 保持 Triton,若调用方绕过 capture-mode 判定可能出现 CuTe 启动开销;SM100 路径不变,回归主要在 gate 逻辑与测试注册变化。

## 关联

#32890(CuTe KDA verify 内核)、#32541(nv_cutedsl 后端)、#42905(GLM-5-Next 默认)、#33102(FlashInfer GDN ReplaySSM 内核,其 `use_packed_fma` 开关是本 PR 同款做法)、#40498/#43659/#44262(测量构建中携带的关联修复)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
