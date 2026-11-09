# PR #33922: Fix Qwen3.5 GDN multi-item scoring

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @daii-0818 |
| 合并 | 2026-09-10 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +add/-del 行数与文件数;diff 可用,另有 3 个文件因超长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/33922 |

## 一句话结论

修复 Qwen3.5 GDN 多 item 评分(MIS)的状态泄漏:每 item 从 query 末端 conv/SSM 状态分支而非连续递推,MIS 吞吐 +18.84%。

## 动机

Qwen3.5 hybrid 模型同时使用 recurrent GDN 层与 full-attention 层。multi-item scoring(MIS)已经给 full-attention 层提供了 item-branching mask,但 GDN prefill 路径此前把 packed suffix 当作一条连续 recurrent 序列处理——后面的 item 会继承前面 item 的 convolution 与 SSM 状态,属于正确性缺陷(Fixes #31969)。

## 优化点

- Triton chunk gated-delta-rule 路径新增 read-only final-state 模式:`chunk_gated_delta_rule`/`chunk_gated_delta_rule_fwd_h` 增加 `inplace_update` 参数(默认 True 保持原有 in-place 语义),MIS 的 item 分支以 `inplace_update=False` 运行,对 request SSM pool 只读;支持重复的 `initial_state_indices`(多个 item 共享同一 query 末端状态且不写回)。
- GDN 专属 MIS metadata 每 forward 只构建一次:`build_gdn_mis_metadata` 从 request-local MIS delimiters 生成紧凑的 query 段与 item 段(token indices、cu_seqlens、request indices),并校验 delimiter 严格递增、末位 delimiter 必须是请求最后一个 token 等。
- GDN MIS prefill 两阶段执行(`_forward_extend_mis`):第一阶段每 request 的 query 段计算一次并写回 conv/SSM 状态;第二阶段每个 item 从 query 末端的 convolution 与 SSM 状态分支——item recurrence 对 request SSM pool 只读,convolution 状态则使用 per-item 唯一的克隆槽,互不污染。
- 拒绝不支持的组合:非 Triton GDN prefill、page-major layout、cached prefixes、非 prefill 请求(含 target verify)、未声明 MIS 支持的 hybrid linear backend(`MambaAttnBackendBase.supports_mis` + `validate_mis_support`);XPU/CPU/NPU 的 chunk kernel 遇到 `inplace_update=False` 抛 `NotImplementedError`。
- 参考实现改为按 MIS 分段递推(query 末端状态克隆后逐 item 演化),并在测试 fixture 中调高 conv history 权重与 A_log/dt_bias,防止跨 item 状态泄漏被随机 gate 衰减掩盖。

## 关键改动

```text
python/sglang/kernels/ops/attention/fla/chunk_delta_h.py: chunk kernel 的 INPLACE_UPDATE 改为可参数化(read-only final-state 模式)
python/sglang/srt/layers/attention/linear/gdn_backend.py: GDNMISMetadata 构建、两阶段 _forward_extend_mis、MIS 后端校验
python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py: extend dispatcher 透传 inplace_update
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py: supports_mis 声明与启动期校验
python/sglang/test/kits/attention_unittest/attention_methods/gdn_attention.py: MIS 分段参考实现与测试用例
```

## 性能声明

测试条件:单张 NVIDIA H20,`Qwen/Qwen3.5-4B`,query length 120、item length 180、每 request 10 items,10 个 warmup 请求后三次 60 秒运行;两种模式均禁用 CUDA graph、radix cache 与 chunked prefill。

| 模式 | Items/s/GPU | P50 延迟 | P99 延迟 |
| --- | ---: | ---: | ---: |
| Independent pointwise scoring | 52.80 | 189.06 ms | 195.25 ms |
| Multi-item scoring | 62.75 | 158.74 ms | 167.21 ms |

即吞吐 +18.84%,P50 延迟 −16.04%,P99 延迟 −14.36%。PR 未注明其他硬件/负载配置的数字。

## 验证与回归风险

精度:NVIDIA H20,Qwen3.5-4B BF16,FlashInfer full attention + Triton linear-attention prefill。Qwen3.5 MIS 端到端 3 tests 通过(空 query、并发请求、不同 item 数/长度、sibling 修改/重排序);focused recurrent/causal-conv/backend/metadata/validation 套件 27 tests + 46 subtests 通过;完整 recurrent 与 causal-conv 测试文件 275 tests 通过;multi-item 分数与逐 item MIS scoring 一致(atol=2e-2, rtol=2e-2);新增 kernel 级测试验证 read-only 路径下重复 initial_state_indices 不修改 pool。pack 中的 CI 状态显示 Latest PR Test (Extra) 失败、(Base)/(AMD ROCm) pending。风险面:MIS 仅支持 Triton prefill kernel(其他 prefill backend 启动报错)、不支持 target verify、cached prefix、page-major layout;XPU/CPU/NPU kernel 无 read-only 路径;需要 `--enable-mis` 与 delimiter metadata 齐备。

## 关联

Fixes #31969(pack 未给出其他同区域 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
