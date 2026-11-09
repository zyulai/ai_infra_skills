# PR #35499: [AMD] Improve K3 dspark draft attn kernel perf

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @1am9trash |
| 合并 | 2026-08-21 |
| 分类 | Attention / Kernel(major) |
| Label | amd, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/35499 |

## 一句话结论

把 K3 DSpark draft attn 从通用 extend kernel 切到 `verify_shared_kv`,TP8 下 cc=4 的 ITL 提升 4–12%。

## 动机

Kimi-K3 上跑 DSpark 时每步执行 5 层 draft attn(qwen 式、非因果)。trace study 显示 5 层 draft 共 2.368 ms,其中 attn kernel 占 1.244 ms(>50%),原因是该路径回退到 `extend_attention_fwd`,而通用 extend kernel 不适合 shared-KV 形状。draft attn 是公共路径:`--attention-backend` 只选目标验证 kernel,aiter 不在 draft worker 支持列表里,ROCm 上 draft 始终走 triton——因此 `--attention-backend triton` 与 `aiter` 两种部署都会命中本 PR 改的 kernel。

## 优化点

把 DSpark draft attn 路由到 `verify_shared_kv`,并扩展该 Triton kernel 覆盖此前拒绝的两种情形:其一,非因果——prefix loop 本就不加 causal mask,只需给 stage-2 combine kernel 加 `IS_CAUSAL` constexpr(draft query 看全部 key;causal 时仅看自己之前的 draft token),`can_handle` 相应放宽(`not is_causal` 且 custom_mask 为 None 即可,不再一律拒绝);其二,`h_kv > 1`——两个 stage 均加 KV-head offset(`head // kv_group_num * stride_h`),用 `HAS_KV_HEADS` constexpr 保证单 KV head(MLA/MQA)路径的内层寻址与原来逐位一致,既有 Kimi-K3 target-verify 路径编译结果不变。`can_handle` 在 h_kv>1 时要求 kv_group_num 为 2 的幂,`block_h` 必要时收缩到 kv_group_num 以保证一个 head block 落在单个 KV head 内。另加 `head_dim=64` 的调优项(`BLOCK_N=256`,4 warps)与 `model_config.py` 的 `is_dspark_draft()` 门控(`_should_use_verify_shared_kv` 对 DSparkDraftModel + 非 MLA 放行)。

## 关键改动

```text
python/sglang/kernels/ops/attention/verify_mla.py: IS_CAUSAL/HAS_KV_HEADS constexpr、KV-head offset、head_dim=64 调优项
python/sglang/srt/configs/model_config.py: 新增 is_dspark_draft()
python/sglang/srt/layers/attention/triton_backend.py: DSpark draft 模型启用 verify_shared_kv 路由
```

## 性能声明

docker `rocm/sgl-dev:v0.5.17-rocm720-mi35x-20260818`,直接在 main 上测量,Kimi-K3 + DSpark,TP8,`--kv-cache-dtype fp8_e4m3` 等(见 server 脚本):isl=8k/16k/32k、cc=4/32 下,cc=4 时 ITL 提升 4–12%、TTT 提升 2–6%;cc=32 时 ITL 提升 1–5%、TTT 提升 1–2%;收益随输入长度(8k→32k)增大。表中最大:32k/1k、cc=4:TTT 8924.78 tok/s(1.06x)、median ITL 5.15 ms(1.12x)、acc len 5.92。作者估计 aiter 后端下收益更大(aiter 加速 target-verify,draft 占比更高)。

## 验证与回归风险

GSM8K 1319 例:Accuracy 0.955,Invalid 0.001,输出吞吐 1108.066 token/s。PR 未新增单元测试(checklist 未勾选)。风险面:`verify_shared_kv` 的 `can_handle` 契约变化(不再要求单 KV head、非因果放行)影响所有潜在调用方;`custom_mask` 在 HIP graph 内不可读的前提未变,tree mask 等价性仅在 topk==1 成立(调用方须继续 gate);h_kv>1 仅支持 2 的幂 group;`IS_CAUSAL=False` 时 draft 间全可见,依赖 DSpark draft 语义确为 bidirectional。

## 关联

无具体 PR 编号(同区域:Kimi-K3 target-verify 复用同一 `verify_shared_kv` 内核,本 PR 通过 constexpr 保持其编译不变)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
