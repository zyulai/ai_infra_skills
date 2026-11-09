# PR #36624: [Cohere Command-A-Plus] Optimize decode and BCG capture on SM10X

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mmangkad |
| 合并 | 2026-09-01 |
| 分类 | Other (tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 记录 +None/-None/None files);diff 可用,涉及 5 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/36624 |

## 一句话结论

SM10X 上 Cohere Command-A-Plus 三处优化(共享 stream、decode 并行块重叠、trtllm-gen 融合 MoE),decode TPOT 最高降 2.28x,BCG prefill capture 内存从 23.57GB 降到 5.34GB。

## 动机

Cohere2Moe 在 SM10X 上跑不起来 breakable CUDA graph(BCG)prefill 后端:`Cohere2MoeSparseMoeBlock.__init__` 每层新建一个 `torch.cuda.Stream()`,32 层共 32 个;caching allocator 按 stream 分池,prefill capture 时 graph pool 涨到 19.02 GiB、另有 23.83 GiB 滞留在 reserved-but-unallocated,超出约 45.7 GiB 余量而 OOM。同时模型本身是并行注意力结构 `y = x + attn(norm(x)) + mlp(norm(x))`,两支相互独立却在同一 stream 上串行执行,decode 时两者都填不满 GPU。

## 优化点

1. **共享 stream 替代每层 stream**: MoE block 不再自建 stream,改为租用全局共享的 `get_stream("alt")`;allocator 不再按 32 个 stream 分池, capture 内存降到 5.34 GB,该架构得以进入 breakable allowlist(`Cohere2VisionForConditionalGeneration` 从 piecewise 名单移到 breakable 名单)。
2. **decode 并行块重叠**:mlp 移到第二支 side stream(`get_stream("alt_mlp")`)形成三条并发链;fork/join 仅在 `forward_mode.is_decode()` 且 capture mode 下生效(prefill chunk 已饱和 GPU,split 只剩 event 开销)。配套 `FusedMoE` 传 `inplace=False`——默认将 routed-expert 输出别名到输入上,会与另一 stream 上 `qkv_proj` 仍在读的 post-layernorm buffer 竞争。
3. **MoE runner 固定为 flashinfer_trtllm**:`auto` 会把 routing、expandInputRows、activation、finalize 拆成独立内核,trtllm-gen 全部折进一个内核。新增 config-time override:SM10X 且 `moe_runner_backend=auto` 时,未量化或 nvfp4-pack-quantized checkpoint 固定到 `flashinfer_trtllm`;compressed-tensors FP8 保持 `auto`(其 `TritonMoeQuantInfo` 会被 trtllm-gen 在首个 forward 拒绝),gate 采用 fail-closed 设计——读不懂的量化配置一律不动。

## 关键改动

```text
python/sglang/srt/models/cohere2_moe.py: alt_stream 改 get_stream("alt");
  decoder 层增加 mlp_stream decode 重叠分支;FusedMoE inplace=False
python/sglang/srt/arg_groups/model_overrides/cohere2_moe.py: 新增 config-time
  override,SM10X+auto 时按量化格式固定 moe_runner_backend=flashinfer_trtllm
python/sglang/srt/configs/model_config.py: Cohere2Vision 从 piecewise 名单
  移到 breakable CUDA graph 名单
test/registered/unit/test_model_overrides.py: 新增 runner gate 测试与
  fail-closed 测试
```

关键片段(diff 摘录,decode 重叠分支):

```python
self.mlp_stream.wait_stream(current_stream)
with torch.cuda.stream(self.mlp_stream):
    mlp_out = self.mlp(hidden_states)
attn_out = self.self_attn(...)
current_stream.wait_stream(self.mlp_stream)
```

## 性能声明

PR 正文,未独立核实。4× GB300 (SM103),tp=4,random 8192-in/1024-out,`--num-prompts 4*C`;两侧均 `--mem-fraction-static 0.75` + `--cuda-graph-backend-prefill breakable`(基线在默认 mem-fraction 下 capture 直接 OOM,无 before 列;0.75 代价是 KV cache 少约 10%:6,362,112 → 5,710,976 tokens):

- **w4a4 (NVFP4)**:TPOT conc=1 从 4.28ms 降到 **1.88ms(2.28×)**,out tok/s 227→504(2.22×);收益随并发升高收敛(conc=256 为 1.43×/1.39×)。
- **bf16**:TPOT conc=1 从 4.39ms 降到 2.78ms(**1.58×**);conc=256 反升至 1.38×。
- **TTFT**:全并发改善 **15–37%**,最优在 c1:w4a4 129→109ms,bf16 209→140ms。
- **BCG prefill capture**:w4a4 23.57GB/22.33s → **5.34GB**/19.23s;bf16 21.36GB/24.04s → **5.21GB**/17.47s。
- **精度(GSM8K,1319 例)**:w4a4 95.43%→95.21%,bf16 95.66%→95.89%,均小于单次运行 std(0.21pp)。

作者注明收益形状符合 overlap 预期:内核欠占用时峰值,GPU 饱和后收敛。自动解析的后端:attention trtllm_mha,MoE runner flashinfer_trtllm,FP4 linear GEMM auto(flashinfer_cutedsl)。

## 验证与回归风险

- 新增 `test_cohere2_moe_runner_gate` 与 `test_cohere2_moe_runner_gate_fails_closed` 两个注册单测,覆盖 NVFP4/bf16 固定 trtllm、FP8 留在 auto、显式 backend 不被覆盖、非 SM10X 不动、不可读量化配置 fail-closed。
- CI 状态(pack 记录):Base 与 Extra 未通过/失败,AMD ROCm 7.2 进行中。
- 回归面:`inplace=False` 略增 MoE 显存拷贝;side-stream 重叠只在 capture/decode 生效,依赖 event sync 正确性(代码注明无需 record_stream);MoE runner gate 若误判量化格式会在首个 forward 崩溃(测试专门钉住该行为)。

## 关联

同区域可参考同 batch 的 SM10X 相关 PR(如 #27053 BCG 相关),pack 未直接给出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
