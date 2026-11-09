# PR #36162: [Mamba] Flashinfer selective_state_update

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @roikoren755 |
| 合并 | 2026-04-14 |
| 分类 | Attention / Kernel（tier: major） |
| Label | ready, ci/build, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files); pack 末尾无 dropped-from-diff 说明(diff 未裁剪文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/36162 |

## 一句话结论

Mamba selective_state_update 引入 FlashInfer 后端与 dispatcher，Nemotron-3-Nano E2E 约 4-5% 提速。

## 动机

为 FlashInfer 的 `selective_state_update` kernel 增加 wrapper，带一个运行时 dispatcher，并连到一个 config 字段，用于在该 kernel 与现有 triton 实现之间选择。按 @tdoublep 在 #35753 中的建议，本 PR 引入 `MambaConfig`，后续可把与 Mamba 相关的 config 字段迁入其中。PR 正文未详述改动前的性能瓶颈。

## 优化点

新增 `MambaConfig` 与 `--mamba-backend` 配置（triton / flashinfer）；`ssu_dispatch.py` 做运行时分发，在 FlashInfer 与 triton 两条 `selective_state_update` 实现间选择；`mamba_mixer.py`、`mamba_mixer2.py`、`mamba_ssm.py` 接入新 kernel，并联动 `vllm/config`、`vllm/engine/arg_utils.py` 与 GPU model runner；Nemotron 的 gsm8k eval 配置补 `--mamba-backend flashinfer`。正文称配套新增 dispatcher 功能测试、使用 FI kernel 的 Nemotron 3 Nano e2e 测试，并对两种后端做 benchmark 对比。

## 关键改动

```text
vllm/model_executor/layers/mamba/ops/ssu_dispatch.py: selective_state_update 的运行时 dispatcher，在 FlashInfer 与 triton 实现间选择
vllm/config/mamba.py: 新增 MambaConfig，承载 mamba backend 等配置
vllm/engine/arg_utils.py 与 vllm/config/*: 增加 --mamba-backend 参数并接入配置体系
vllm/model_executor/layers/mamba/ops/mamba_ssm.py、mamba_mixer2.py: 接入 FlashInfer selective_state_update kernel
tests/kernels/mamba/test_ssu_dispatch.py: 新增 dispatcher 功能测试
```
```text
   --enable-expert-parallel
   --mamba-backend flashinfer
   --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
```

## 性能声明

模型 nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16，TP4、max-model-len 8192，`vllm serve ... --mamba-backend {triton|flashinfer} --trust-remote-code`，benchmark 为 `vllm bench serve --num-prompts 500 --request-rate inf --input-len 256 --output-len 256`。汇总表（Triton → FlashInfer）：Duration 5.88 s → 5.64 s（-4.1%）；Request throughput 85.02 → 88.60 req/s（+4.2%）；Output throughput 21765.03 → 22682.25 tok/s（+4.2%）；Peak output throughput 31129 → 32938 tok/s（+5.8%）；TPOT median 17.90 → 17.26 ms（-3.6%）；TPOT P99 19.41 → 18.60 ms（-4.2%）；ITL P99 69.00 → 63.94 ms（-7.3%）。正文结论为 "~4-5% speedup in most all metrics"，并称 e2e GSM8K 两后端分数相同。测试条件：模型见上、TP4、input 256 / output 256、500 prompts、request-rate inf；PR 未注明 GPU 型号。

## 验证与回归风险

正文 Test Plan 列：新增 dispatcher 功能测试文件、为 Nemotron 3 Nano 增加使用 FI kernel 的 e2e 测试、对两种后端做 benchmark；Test Result 称测试通过、Nemotron-3-Nano-30B-A3B-BF16 的 e2e GSM8K 两后端分数相同，并给出上述 serving benchmark 对比。回归风险：改动经 config 字段选择后端，正文未注明默认后端值；涉及 Mamba mixer、mamba_ssm、plamo2、GPU model runner 等多处；`MambaConfig` 为本 PR 新引入，正文指出后续可能把更多 Mamba 相关 config 字段迁入（#35753 建议）。

## 关联

正文明确 #35753（@tdoublep 建议引入 `MambaConfig`）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
