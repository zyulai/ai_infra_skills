# PR #42175: [Performance] Optimize Qwen3.8-Flash-Next BF16 decode on Blackwell

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-10-09 |
| 分类 | Speculative Decoding(major) |
| Label | run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数;diff 另有 6 个文件因超长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/42175 |

## 一句话结论

针对 Qwen3.8-Flash-Next 单 token BF16 decode 在 4×B200 上把 8k-input/1k-output 解码速率从 242.443 提到约 271 token/s(约 +12%)。

## 动机

单 token BF16 decode 会分别启动 HC combine/norm 与 MoE shared-gate/finalize/collective 工作,而 packed GDN value tile 在该 shape 下带来较高的寄存器压力;stock 的 decode 计算图有 1432 个 kernel/图,存在可融合与可重叠的开销。

## 优化点

三条机制改动:其一,HC gate 与 block 计算重叠——在专用 CUDA stream 上预先计算 gate partials(`hc_combine_gate`),并把 HC apply 与 grouped Gemma RMSNorm 融合为单个 kernel(`hc_combine_apply_norm`),层间显式传递准备好的 norm 张量,PLE/PP 边界保持原 handoff。其二,扩展 fused MoE finalize + one-shot all-reduce collective,新增 `kGatedShared` 路径:在 routed 值按 BF16 取整后、shared-expert FMA 之前施加 FP32 shared gate,推迟 BF16 MoE finalize 并经带独立通信 buffer 的单 token TP4/EP1 守卫路径与 shared 分支重叠。其三,packed 单 token BF16/FP32-state Blackwell shape 下把 GDN value tile 从 BV=32 缩到 BV=8:原先只发 48 个 CTA、每线程寄存器超 200,缩小 tile 后有 192 个独立 CTA,不改变递推与 one-warp K reduction,不支持 shape 仍走原 dispatch。specialization 要求 hidden 2560、512 experts/top-10、BF16、SM10、TP4/EP1、attention DP1,且禁用 LoRA、投机解码与 two-batch overlap;无需额外命令行开关。

## 关键改动

```text
python/sglang/kernels/jit/csrc/elementwise/hc_combine_decode.cuh: 新增 gate/apply/apply_norm 三段 HC decode kernel(均 PDL)
python/sglang/kernels/jit/csrc/distributed/all_reduce_fusion.cuh: kGatedShared 模板路径,finalize 内 FP32 gate FMA
python/sglang/srt/layers/hyperconnection.py: _gate_stream 上预发 gate partials,combine_and_normalize 融合路径
python/sglang/kernels/ops/attention/fla/fused_recurrent.py: 单 token Blackwell shape 的 GDN BV=32 -> 8
python/sglang/srt/layers/moe/fused_moe_triton/layer.py: qwen4_bf16_deferred 资格判定(2560/512/top-10/BF16)
```

## 性能声明

4×B200,BF16 weights/KV/indexer、FP32 GDN state、TP4、GPU PLE、context 16384、concurrency 1,8k-input/1k-output,3 warmups + 32 measured requests:stock 242.443 tok/s → candidate 270.982–271.759 tok/s,三轮配对比率 1.146404x/1.146941x/1.150134x;对比最快 vLLM GPU-PLE 结果 240.786 tok/s,最慢 candidate 也快 12.54%(该分母含质量不达标的 EP4 结果)。GSM8K stock 与四个 native candidate 均为 126/128、零无效答案、零截断。decode 计算图 kernel 数 1432 → 1241(含 95 HC apply/norm、48 gated finalize/all-reduce、36 packed GDN)。后续在 4×GB300 上:merge 后固定输入 decode BS1 中位时延 +1.38% 至 +1.78%、BS2 +0.39% 至 +0.81%、BS4 0% 至 +0.48%(作者明确为实测小回退而非零开销声明);cleanup head 复测 277.127 → 276.606 tok/s(-0.19%)。

## 验证与回归风险

GSM8K 128 题冻结 workload、全部 128 题判定与 stock 一致;历史验证含 6 HC 单元配置、12 torch-reference GDN 配置、4 rank × 6 collective 配置、90 组 trained HC-chain 比较、324 组 changed-input collective graph 比较、192 组 seeded GDN bit 比较;merge/cleanup 后追加 306 layer-boundary 测试、377 单测、94 gate 测试与大量 bitwise 比较。风险面:specialization 门槛严格(shape/量化/并行/功能开关多重 guard),不满足时回退 stock dispatch;固定输入 logits 非位可复现且来源未定位,作者说明功能证据为 GSM8K 与 kernel/boundary 检查;GB300 上小批 decode 有实测约 +0.4%–1.8% 时延回退。

## 关联

无(素材未列出同区域 PR;reproducer gist 提供基准复现,未列前序/后续 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
