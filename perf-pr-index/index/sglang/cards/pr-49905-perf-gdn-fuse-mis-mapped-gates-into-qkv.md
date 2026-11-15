# PR #49905: [Perf] gdn: Fuse MIS mapped gates into QKV preparation

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mani1106 |
| 合并 | 2026-11-14 |
| 分类 | Attention / Kernel(tier: major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/49905 |

## 一句话结论

GDN multi-item scoring (MIS) 的 gate 读取与 softplus/sigmoid 计算融进 QKV preparation 单次 Triton launch:canonical preparation 198 token 2.75x、4096 token 1.82x;H200/Qwen3.5-0.8B BF16 serving 吞吐 +1.1–5.5%(正文注明微内核倍数不是端到端加速声明)。

## 动机

减少 GDN MIS 的 preparation overhead,基于 #48585。每个 query/item segment 已有到原始 token 行的 map;baseline 先 gather 对应的 `a`/`b` gate 行(两个 gathered 临时张量)、split/normalize post-convolution QKV,再单独 launch standalone gating。本 PR 在 QKV preparation 内部直接经该 map 读 gate 行:每 segment 的 QKV split 与 Q/K normalization 用同一 preparation kernel(扩展进 gate 计算),`a`/`b` 选择由两次 gathered 临时张量变为 direct mapped loads,`g`/`beta` 计算并入 preparation;convolution、recurrence、state updates 的 MIS 行为不变。

## 优化点

- 新 `fused_qkv_gating_gdn_prefill`:共享现有 split launcher 与 Q/K normalization,返回 contiguous Q/K/V 与 FP32 gates;现有 `fused_qkv_split_gdn_prefill` 的签名/三输出契约不变。
- 数值/布局:复用 standalone 的 softplus/sigmoid 表达式(含 sigmoid 经 `b.dtype` round 后再入 FP32 存储);支持 strided inputs/parameters 与 contiguous int32/int64 maps,地址运算在 stride 乘法前扩宽为 int64;map 内容不在 host 读取,由 caller 保证源行在界内。
- MIS dispatch:CUDA/HIP 且 packed QKV width ≤ 8192 时传入原始 gates 与 segment map;既有 shape/device fallback 仍 gather gates 并用 standalone gating。query/item state ownership、output mapping、padding 与 normalization ownership 不变。
- kernel 注册:`attention.fused_qkv_gating_gdn_prefill`(TRITON backend,CUDA + HIP capability)。
- 测试:覆盖 mapped gates、layouts、large strides 与 fallback;保留继承的 normalization-composition 测试(原 seeded inputs/FP16 bound),另要求 mapped-gating composition 与同一 normalization + standalone gating 严格一致。

## 关键改动

```text
python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py: 新 fused_qkv_gating_gdn_prefill + _prepare_gdn_prefill;kernel 增加 FUSE_GATING 分支(int64 offsets、经 token_indices 直接读 gate 行)
python/sglang/kernels/ops/attention/linear/fused_gdn_gating.py: 提取共享 _gdn_gating_values(softplus/sigmoid 表达式)
python/sglang/kernels/ops/attention/__init__.py: 注册 attention.fused_qkv_gating_gdn_prefill(CUDA/HIP)
python/sglang/srt/layers/attention/linear/gdn_backend.py: _forward_extend_mis 不再预 gather a/b;_forward_mis_segments 改用 mapped-gate 路径并保留 fallback
test/registered/kernels/ops/attention/test_gdn_mis_output.py: 新增 mapped gates / 大偏移 / fallback / composition 测试
test/registered/kernels/ops/attention/kda_flydsl/test_kimi_k3_kda_decode.py: DeprecationWarning 改 FutureWarning(测试侧)
```

```python
if FUSE_GATING:
    source_token = tl.load(token_indices + i_t).to(tl.int64)
    gate_heads = tl.arange(0, BLOCK_GATE_HEADS).to(tl.int64)
    gate_mask = gate_heads < NUM_V_HEADS
    gate_a = tl.load(
        a + source_token * STRIDE_A_T + gate_heads * STRIDE_A_H,
        mask=gate_mask, other=0,
    )
    gate_b = tl.load(
        b + source_token * STRIDE_B_T + gate_heads * STRIDE_B_H,
        mask=gate_mask, other=0,
    )
    gate_g, gate_beta = _gdn_gating_values(
        gate_A_log, gate_a, gate_b, gate_bias, 1.0, 20.0
    )
    tl.store(g + i_t * NUM_V_HEADS + gate_heads, gate_g, mask=gate_mask)
    tl.store(
        beta_output + i_t * NUM_V_HEADS + gate_heads,
        gate_beta.to(b.dtype.element_ty), mask=gate_mask,
    )
```

## 性能声明

- Preparation 微基准(H200,upstream `marker.do_bench`,eager、每次调用前 flush L2、10 warmups、100 timed calls/arm/trial、三次交替 trial;含 split/normalization + gate gather/gating vs mapped preparation;输出 bitwise 一致):packed width 6144 mapped 路径 198 token 21.824→7.936 µs(2.75x)、4096 token 56.544→31.072 µs(1.82x);8192(dispatch boundary)198 token 27.552→8.352 µs、4096 token 69.248→39.232 µs;12288(generic fallback)基本不变(17.344→17.312 / 33.312→33.280 µs);standalone raw split 不变。正文注明这些 microkernel 因子不是端到端加速声明。
- 端到端 score serving(one H200 driver 580.159.04;Qwen3.5-0.8B BF16,checkpoint `2fc0636...`,18 GDN layers、16 Q/K + 16 V heads、head dim 128、packed QKV width 6144;FlashInfer full attention、Triton GDN、MIS enabled,radix cache/chunked prefill/CUDA graphs disabled;每臂 3 个 fresh server,顺序 baseline/PR/PR/baseline/baseline/PR;每 shape/concurrency 10*C warmup + 10*C measured,C1/C8 行每臂 30/240 measured requests;pointwise/setwise 读取任务、2/8 ragged items、短/长 query):16 个 workload 的 median serving throughput +1.11% 至 +5.50%(如 pointwise long 2 items C1 +5.50%、pointwise short 2 items C1 +3.80%、setwise long 8 items C8 +2.33%);single-item MIS 对照 C1/C8 +2.75%/+3.44%(仍激活 MIS)。Percentiles 有波动、并非全部更低;三次 trial(尤其小样本 C1)不能确立可复现的尾延迟回归或普适生产收益。
- Non-MIS 对照(仅去掉 `--enable-mis`,gate fusion 不激活):pointwise short 8 items C1 +3.85% / C8 +1.38%,setwise short 8 items C1 +2.21% / C8 −0.19%,pointwise single item C1 +1.13% / C8 +3.09%——正文注明其变动不应归因于本优化。
- Profiling(实际 score 请求,每 shape 10 warmup + 5 serial captured 请求,/start_profile CPU/GPU):两 workload 的 180 split + 180 standalone gates launches 变为 180 fused preparation、无 standalone gates;每次五请求 MIS capture 消失 360 次 gate gathers 与 180 次 standalone gate launches;total GPU kernel events 4440→3900(short pointwise 2 items)、4305→3765(long setwise 8 items);summed split+gate / fused-preparation GPU time 0.529→0.387 ms / 1.536→1.167 ms。GPU-time sums 是 attribution 测量,不是 server latency 或 overlap-adjusted speedups。

## 验证与回归风险

- `python test/registered/kernels/ops/attention/test_gdn_mis_output.py -f`:14 GPU tests passed(H200,最终文件);`test/registered/attention/test_gdn_prefill_layout.py`:5 GPU tests passed;`QWEN35_GDN_TEST_MODEL=... test/registered/e2e/gdn/test_qwen35_gdn_multi_item_scoring.py -f`:baseline/final 各 3 个 BF16 real-model tests passed;五个改动文件 pre-commit passed;final baseline/PR HTTP matrix 6480 measured requests、零失败。
- 新的 mapped-gating composition 对 BF16 与 FP16 均 bitwise exact(含 state 与 padding);继承的 FP16 composition 容差保持两 ULP、未放宽。BF16 HTTP scores 在 concurrency 1 精确;concurrency 8 最大绝对分差 1.24375921869599e-5,也在 baseline 自身 batching run 内出现——这是数值/输出等价性检查,不是任务质量精度声明。
- 覆盖限制(正文自述):一台 H200、一个 BF16 模型、TP=1;HIP、其他 accelerator/模型、分布式配置与 FP16 serving 未建立;记录的环境替换(CUTLASS DSL 4.7.1、Tokenizers 0.23.3、Humming 0.1.12、sgl-deep-gemm 0.2.3.post2、sgl-flash-mla 1.0.0)意味着这不是 full upstream-lock certification,也不声明 H100 CI 已通过。

## 关联

基于 #48585(正文注明 building on it);正文注明本变更在 GitHub Copilot 协助下开发。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
