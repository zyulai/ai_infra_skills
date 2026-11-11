# PR #48613: [Perf] kda: Run the fused CuTe DSpARK verify kernel on a BF16 SSM state

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @elvischenv |
| 合并 | 2026-11-10 |
| 分类 | Speculative Decoding(tier: major) |
| Label | run-ci, jit-kernel |
| Diff | 素材未提供 diffstat(diff 摘录完整:4 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/48613 |

## 一句话结论

让 fused CuTe-DSL KDA verify kernel 支持 BF16 SSM state,Kimi-K3.1 DFLASH decode 每 step 的 KDA 时间从 fallback 3.31 ms 降回 0.84 ms(FP32 fused 为 0.77 ms)。

## 动机

`--mamba-ssm-dtype bfloat16` 时,Kimi-K3.1 DFLASH decode 会失去 fused CuTe-DSL KDA target-verify kernel(`fused_kda_decode_mtp_dspark`,nv_cutedsl verify backend):`KDAAttnBackend._can_run_dspark_cutedsl_mtp` 对非 FP32 state 返回 False,备选的 `_can_run_fused_chain_verify` 也是 FP32-only。于是每个 KDA layer 的每个 verify step 回落到 causal_conv1d_update(Triton)+ gated delta-rule 更新(Triton)+ 独立 gated RMSNorm——输出 norm 只融合在 CuTe kernel 里。DFLASH 下 target model 只跑 TARGET_VERIFY,这就是每个 decode step 的 KDA 路径:fallback 每 layer 耗时为 fused 的 1.5-2.3×,69 个 KDA layers 即每 decode step 多 0.7-2.9 ms(8-128 requests/rank)。

## 优化点

只改两个运行时文件,不动 recurrence 语义。kernel 侧(`kda_decode_mtp.py`):shared-memory state stage 改用 state 自身 dtype;128-bit `cp.async` copy atom 对 FP32 搬 4 个、BF16 搬 8 个元素;smem 字节数随 dtype 缩放;recurrence 仍在 FP32 寄存器中运行,BF16 值在 smem→register 加载时 widen,仅 `intermediate_ssm` snapshot store 收窄为 BF16(编译期分支);FP32 state 下 copy atom、layout、smem 大小与 store 与旧代码完全相同。wrapper 接受 FP32/BF16 且 slot 16-byte 对齐(stride/offset 为 4 FP32 或 8 BF16 元素的倍数),要求 snapshot buffer 与 state dtype 一致。dispatch 侧(`kda_backend.py`):gate 接受 BF16 state(8 元素对齐)。无新 flag;BF16 快在两点:每 layer 一个 kernel 而非三个,且 state 字节减半——verify 每 request 读一次 12×128×128 state,BF16 为 384 KB 而非 768 KB,fused BF16 在 32-128 requests/rank 还比 fused FP32 快 7-11%。

## 关键改动

```text
python/sglang/kernels/ops/attention/kda_decode_mtp.py: state stage/cp.async/smem 按.dtype 泛化;snapshot store 编译期收窄
python/sglang/srt/layers/attention/linear/kda_backend.py: _can_run_dspark_cutedsl_mtp 接受 bf16(8 元素对齐)并校验 snapshot dtype
test/registered/kernels/ops/attention/test_kda_decode_mtp.py: TestKdaDecodeMtpBf16State bitwise 测试
test/registered/unit/layers/attention/test_mamba_track_state_dtype.py: dtype gate 单测(CPU)
```

```python
state_dtype = h0.element_type
state_vec = 128 // state_dtype.width
state_copy_atom = cute.make_copy_atom(
    cpasync.CopyG2SOp(cache_mode=state_cache_mode),
    state_dtype,
    num_bits_per_copy=128,
)
```

## 性能声明

GB300 单卡(Kimi-K3.1 shapes,12 KDA heads/rank,ring 模式,CUDA graph of 50 calls,每次前 256 MB L2 flush,µs/call/层):7 draft tokens、128 requests/rank:fallback 107.5 → BF16 fused 65.6(−39%,FP32 fused 70.3);5 draft tokens、128 requests:81.2→52.6(−35%)。每 decode step(×69 层)相对 fallback 省 0.7-2.0 ms(5 draft)/1.0-2.9 ms(7 draft)。模型内(1P2D TP8/DCP8,conc 72):KDA kernel time/step 3.31 ms(fallback)→0.84 ms(fused),FP32 fused 0.77 ms;decode step 31.3→30.5 ms。E2E serving(GB300 NVL72,Kimi-K3.1 FP4 MoE/FP8 KV,1P2D,AIPerf agentic-coding trace,DFLASH 7 draft、AL 3.764):conc 72 输出 17,170→17,395 tok/s/GPU @ P90(+0.4%;fallback 相对 FP32 为 −0.9%),ITL p50 8.65→8.38 ms,fallback 造成约 2.9% 的 P90 tok/s/user 回退被消除;conc 40:10,574→10,763(+0.5%)。BF16 使 prefill worker state slots 440→852、decode worker KV pool 2.45M→2.55M tokens(+4.2%)。

## 验证与回归风险

test_kda_decode_mtp.py(GB300 5 passed/22 subtests):BF16-representable state 下 BF16 kernel 的输出、conv windows、ReplaySSM rings 与 FP32 kernel bitwise 相等,snapshots 等于 FP32 snapshots 的 BF16 舍入;streaming path 同样校验;wrapper 拒绝错位 layout/FP16/dtype 不匹配 snapshot。test_mamba_track_state_dtype.py(CPU 8 passed/11 subtests):gate 接受 FP32/BF16、拒绝 FP16,对齐随 dtype。kernel 数值:BF16 对 FP32 输出 rel Frobenius 约 0.6%(即 BF16 状态舍入本身,fallback 相同),fused 与 fallback 的 BF16 state 差约 1e-4,误差不随轮数增长。GSM8K(1319 题,5-shot,greedy,1P2D TP8/DCP8):FP32 0.968 / BF16 fallback 0.966 / BF16 fused 0.968,均在 ±0.5% 标准误内。限制:仅 DSpARK/DFLASH target-verify 路径;仅 SM100/SM103;fused norm 仅编译 12 heads/rank;精度验证仅 GSM8K,long-context 状态舍入累积未测。

## 关联

#48414(Kimi-K3.1 serving 尚未进 main,测量在其开发分支上进行,本 PR diff 对 main 原样适用);同分类 Speculative Decoding:#42855、#47581。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
