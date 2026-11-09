# PR #47695: [Feat] [AMD] quark: Quark int4 W4A16 on ROCm, and gfx1151 kernels for Strix Halo decode and prefill

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hubertlu-tw(co-author @clintg6、@peizhang56) |
| 合并 | 2026-11-07 |
| 分类 | Quantization(major) |
| Label | quant, run-ci, jit-kernel |
| Diff | pack 未提供 diffstat(additions/deletions/files 均缺失),diff 内容可用(pack 注明 13 个文件因超长被省略) |
| 链接 | https://github.com/sgl-project/sglang/pull/47695 |

## 一句话结论

为 ROCm 增加 Quark int4 W4A16 scheme,并为 Strix Halo(gfx1151)补齐 W4A16 GEMM、verify attention 与 GDN prefill 专用 kernel,decode 吞吐最高提升数倍。

## 动机

`amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` 在 main 上无法加载(Quark 无 int4 weight-only scheme,报 `NotImplementedError`;#41389/#47408 只给 MXFP4 MoE 加了 W4A16)。在 Strix Halo APU(gfx1151,Radeon 8060S,RDNA3.5,256 GB/s LPDDR5X)上,单流 decode + DFlash2 drafter 是访存受限的:M<=8 时 W4A16 GEMM 读权重仅 40-70 GB/s;target-verify attention 每个查询块重走一遍 cached prefix,而 split-KV verify kernel 仅支持 gfx95;无 aiter/vllm 时 `GemmaRMSNorm` 退化为 11-op eager 链(一个 8-token verify 步约 2,600 次 kernel launch 中约 1,400 次来自它)。

## 优化点

统一门控:所有 gfx115x 路径经 `use_gfx115_kernels()`(RDNA3.5 检测 + `SGLANG_DISABLE_GFX115_KERNELS` kill switch,一次性读取),每个 kernel 模块有 `covered()` 谓词只接已测形状,其余回退现有代码;决策尽量在 init 时做。四组 kernel/机制:(1) Quark int4 W4A16 scheme(仅 HIP):`_is_int4_w4a16` 选择 `QuarkW4A16Int4`(int4、symmetric、group 32/64/128、weight-only),加载时 AWQ 打包无损 repack 成 `[N, K/8]` unsigned、zero point 8,通用 Triton skinny GEMM 带 gfx1151 配置表;`qwen3_5.py` 融合 GDN projection loader 按 packed 单元切分。(2) gfx1151 专用 kernel:小 M(M<=16)Gluon WMMA GEMM(`(w>>4i)&0x000F000F|0x43004300` 直接把 int4 码变 bf16 `128+q`,权重/预取进寄存器,读速 165-210 GB/s,需 Triton>=3.7,否则回退 skinny GEMM);prefill GEMM(M>=256,同技巧 `tl.dot`,较 skinny GEMM 快 1.12-1.34x);int4 行 pad(128B/行,避免同行别名到相同 L2/MALL 通道,bitwise 等价,bf16 版 #46327 的 int4 对应物);verify attention kernel(每 (request, kv head, split) 程序对所有 draft token 只读一遍 prefix,q 按 32 宽 chunk 重读控制寄存器溢出,fp32 online softmax,仅 page size 1 等受限条件)。(3) GDN prefill:分组三头 `chunk_fwd_o`(三个 value head 共享一次 q·k^T)与 `chunk_gated_delta_rule_fwd_kkt_solve_kernel`/`recompute_w_u_fwd_kernel` 的 head-major 启动顺序。(4) HIP 精确替换:`GemmaRMSNorm.forward_hip` 改调 Triton `gemma_fused_add_rmsnorm`/`gemma_rmsnorm`(新增 `round_sum=True` 保持与 eager 逐位一致);GDN fused split 头比 3 在无 aiter 的 HIP 上启用。通用层(未量化线性、LM head、DFlash 排序)不动。

## 关键改动

```text
python/sglang/srt/layers/quantization/quark/schemes/quark_w4a16_int4.py:
  QuarkW4A16Int4 scheme(ROCm)
python/sglang/kernels/ops/quantization/w4a16_skinny_triton.py: AWQ->skinny repack、
  skinny GEMM + gfx1151 配置表、pad_rows(128B 行 pad)
python/sglang/kernels/ops/quantization/w4a16_small_m_gfx1151.py 与
  w4a16_prefill_gfx1151.py: Gluon WMMA 小 M GEMM / tl.dot prefill GEMM
python/sglang/kernels/ops/attention/verify_attention_gfx1151.py: split-KV
  target-verify attention
python/sglang/kernels/ops/attention/fla/{chunk_o,chunk_fwd,wy_fast}.py:
  分组三头 chunk_fwd_o、HEAD_MAJOR 网格
python/sglang/kernels/ops/layernorm/minimax_m3_rmsnorm.py: ROUND_SUM 开关
```

## 性能声明

测试环境:Radeon 8060S(gfx1151),Windows 11 WSL2,ROCm 7.2.4,Qwen3.8-27B Quark AWQ INT4,triton attention,DFlash 投机(8 draft tokens),CUDA graph 开启,server 进程 taskset 固核。

- E2E(decode tok/s,TTFT 除外):短 prompt 5.64(bf16 main)/ 18.32(int4 + 关 gfx115 kernels)/ 29.82(本 PR);8.7k context 5.22/15.13/30.91;39k context 3.75/6.82/21.49;HumanEval+ 加权 tok/s 7.12/21.12/32.95;prefill 1k/4k/16k:175/138/127 -> 240/119/112 -> 410/367/337 tok/s;TTFT(8.7k prompt)62.3s/71.8s/24.0s。
- Kernel 级:verify attention 每层对 `extend_attention_fwd` 4.3x(1k prefix,0.43->0.10 ms)、14.1x(8k)、15.9x(39k,16.13->1.01 ms);行 pad 使 prefill GEMM 1.10x、8-row verify 小 M GEMM 1.06x(单列最高 1.2x,bitwise 相等);GDN `chunk_fwd_o` 分组三头 3.5x(T=2048,4.53->1.29 ms)至 6.5x(T=8192,30.19->4.65 ms),完整 GDN chunk forward 2.1x/4.9x;GemmaRMSNorm 在 CUDA graph 中 24.5 -> 3.2 us/次。

## 验证与回归风险

精度:GSM8K 5-shot 200 题——main bf16 0.980 / int4 关 gfx115 kernels 0.965 / 本 PR 0.965(0.980->0.965 是 Quark int4 导出的精度代价,gfx1151 kernels 不改变精度;review 后双臂复测一致)。94 个单测在 gfx1151 通过:3 个注册 AMD CI(skinny repack/GEMM 带 pad 与否对比 dequant matmul、Gemma norm 对 eager、ratio-3 split);manual 测试覆盖小 M GEMM 多形状 bitwise、verify attention 对 torch 参考(多组头配置)与 fused qkv 切片拒绝、GDN chunk 前向对默认 kernel。verify attention 与通用 kernel 与 fp32 参考等距(最大偏差 2.4e-4 至 9.8e-4);GDN head-major bitwise 一致,分组 chunk_fwd_o 输出差至多 6e-5。风险:小 M GEMM 依赖 Triton>=3.7 的 `AMDWMMALayout`,旧版回退 skinny GEMM;verify attention 仅限 page size 1、全注意力、无 soft cap/sinks/sliding window 等严格条件,由 `covered()` 兜底;行 pad 改变权重张量 stride;CDNA(MI300)仅跑了注册单测。

## 关联

前序:#41389、#47408(Quark MXFP4 W4A16)、#46327(bf16 行 pad,本 PR 为 int4 对应物)、#28715/#32045(MiniMax-M3 Gemma norm kernel)、#45540(RMSNorm fallback 替换)、#34421/#34859/#44302(GDN ratio-3 fused split)。后续:#29630(KernelSpec 注册 gfx1151 kernels 的 follow-up)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
