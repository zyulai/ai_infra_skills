# PR #33576: [AMD] Add Work-Centric (Lean) Attention: a persistent-CTA decode kernel for long-context serving

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @valechen |
| 合并 | 2026-08-29 |
| 分类 | Attention / Kernel (major) |
| Label | documentation, amd, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);PR 自述 6 文件约 +1540 行,pack 的 diff 截断(5 个文件未展示) |
| 链接 | https://github.com/sgl-project/sglang/pull/33576 |

## 一句话结论

ROCm/Triton 后端新增持久 CTA + work-stealing 的 Lean decode kernel,长上下文与参差 batch 下 E2E 吞吐最高 1.52x、ITL 最低降 3.6 倍。

## 动机

标准 flash-decode(SplitK)按 `batch × head-tiles × num_splits` 固定网格切分 KV。长上下文低 batch 时,单条/少数长序列产生不了足够多的 KV-split 填满所有 CU,大量 CU 空转,decode 延迟随上下文近似线性增长(occupancy 受限而非带宽受限);64K-128K+ 上下文下 attention 主导 decode step。参差 batch(mixed sequence lengths)下 SplitK 无法把长请求的工作挪到短请求释放的 CU 上,少数长尾请求决定步延迟。

## 优化点

Lean Attention 启动固定为设备 CU 数的 persistent grid(默认一 CTA 一 CU,按 ROCm XCD remap 取整;实测超订 2 CTAs/CU 会劣化),kernel 内部从 `kv_indptr` 在设备端推导 KV-tile 调度并协作拆分长序列(work-stealing),长序列被拆到多 CTA、host CTA 用 atomic spin 等待合并 partial max/sum/acc。调度在 replay 时从 `kv_indptr` 读取,无需 host sync,故可被 CUDA graph capture 且单个 captured grid 自适应真实每步长度与参差度。fp8 KV 通过把 `k_scale` 折进 `sm_scale`、q 转成 K dtype 做 fp8×fp8 MMA、`v_scale` 在输出处反量化;支持 contiguous 与 paged KV、MHA/GQA/MLA(rope-split 288/576 特例小 tile)。auto-gate 双模式:eager 用 `lean_decode_seqlen_gate`(按 query-head 并行度 tiles 分档阈值,GQA base 64K、MHA 16K、重 TP shard 128K,batch 越大阈值越低,4K 下限);CUDA-graph capture 用 `lean_capture_policy`(GQA batch≥16、MLA batch≥8、tiles<4 永不 bake)。未实现的特性(logit cap、attention sinks、xAI temperature、score mod)一律回退标准 kernel,绝不静默出错。

## 关键改动

```text
python/sglang/kernels/ops/attention/decode_attention.py: Lean kernel + launch/dispatch/auto-gate(+822 行)
python/sglang/srt/layers/attention/triton_backend.py: forward_decode 解析 enable_lean_attention 并分配 Mp/Lp/Op/locks scratch
test/registered/kernels/test_lean_attention.py: MI35x per-commit CI 门控测试(bf16/fp8/paged parity + gate 逻辑)
benchmark/lean_kernel_sweep.py: Lean vs SplitK kernel 级 sweep
python/sglang/srt/server_args.py 与 environ.py: --enable-lean-attention、SGLANG_DISABLE_LEAN_ATTENTION kill-switch、SGLANG_FORCE_LEAN_GRID_CU_MULT
```

work-stealing 调度核心(设备端,无 host sync):

```python
tiles_per_khead = tl.load(batch_num_block_n + batch_size - 1)
total_tiles = tiles_per_khead * eff_rows
max_tiles_per_wg = (total_tiles + xcd_programs - 1) // xcd_programs
high_load_wgs = total_tiles - (max_tiles_per_wg - 1) * xcd_programs
```

## 性能声明

测试条件:1×MI355X(gfx950,256 CUs;MI300X gfx942 复测),ROCm 7.2.0、PyTorch 2.9.1+rocm7.2.0、Triton 3.7.0(AMD wheel;stock 镜像的 Triton 3.5.1 会反转 crossover);baseline 为 Triton backend 自带标准 decode kernel(非 AITER),`--attention-backend triton` 同入口 A/B。

- Kernel 级(fp16,page_size=1,batch 1-32 × context 8K-128K):峰值 batch=1/128K 下 **5.33×(Qwen2.5-7B 28Q/4KV)/ 5.92×(Llama-3.1-8B 32Q/8KV)**;短上下文低 batch 低于 1×(0.46×-0.92×),由 gate 关闭。
- E2E GQA(closed-loop,CUDA graphs on,seed 42):26 个测量格全部 tps×≥1.06、无回退;峰值 **1.52× 吞吐**(Llama b32/16K ragged ratio0.1)、median/p99 ITL **3.62×/2.91× 更优**(Llama b32/65K ragged);uniform batch 也全赢(1.06-1.30×)。
- E2E MLA(DeepSeek-R1-MXFP4,TP4,fp8 KV):batch≥8 时 1.05-1.13× tps、ITL 1.33-2.41× 更优;batch 1 回退到 0.83×(gate 正确避开)。MI300X 上 MLA 用 DeepSeek-V2-Lite 代理,b16/b32 赢(最高 1.14×/1.75× ITL)。
- 默认路径(auto,无 flag)三臂验证:auto 在 b≥16 跟随 forced(如 b32 96.28→112.77 tok/s),b8 保持 baseline。

## 验证与回归风险

CI 门控测试 `test_lean_attention.py` 注册在 MI35x AMD per-commit suite:bf16 59 格 cosine=1.0000(bit-identical)、fp8 e4m3 cos>0.99、paged KV(page 16/64)cos>0.999,另覆盖 gate/policy 逻辑;MI300X parity 同样通过。CI 状态 PR Test (Base) 通过。风险面:kernel 不支持 logit cap/sinks/xAI/score_mod 时显式回退;head_dim>128 非 MLA 会 OOR,gate 前置拦截;`seq_lens_sum` 缺失(gpu-only batch、EAGLE draft runner)时 eager gate 保守返回 False;重 TP shard(tiles<4)在 capture 中永不 bake;仅 ROCm/AMD 生效,NVIDIA 走标准 kernel。

## 关联

同区域:ROCm/Triton decode attention(SplitK 分段合并 kernel,本卡的 baseline 与 fallback);AMD MI355X/MI300X kernel 优化系列如 #26894、#45739(pack 未确认编号关联)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
