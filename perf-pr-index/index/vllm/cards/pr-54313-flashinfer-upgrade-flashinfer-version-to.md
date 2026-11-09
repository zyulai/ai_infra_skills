# PR #54313: [Flashinfer] Upgrade Flashinfer version to 0.6.18

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wzhao18 |
| 合并 | 2026-08-30 |
| 分类 | Attention / Kernel \| tier: major |
| Label | ready, ci/build, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/54313 |

## 一句话结论

把 FlashInfer 依赖与镜像版本由 0.6.17 升到 0.6.18。

## 动机

正文 Purpose 仅 "Upgrade Flashinfer version to 0.6.18"，未说明升级动因、修复项或目标收益，故具体动机 PR 未注明。从 diff 可见版本钉在多处：Dockerfile 的 `ARG FLASHINFER_VERSION`（安装 flashinfer-jit-cache，注释指出需 CUDA-version-specific index URL）、`docker/versions.json` 的 `FLASHINFER_VERSION.default`、`requirements/cuda.txt` 的 `flashinfer-python` / `flashinfer-cubin`；后者注释说明 flashinfer-cubin 自 0.6.14 起不在 PyPI，setup.py 把它从 install_requires 排除，以免已发布 wheel 带上无法解析的 pin。

## 优化点

主体是版本号 0.6.17 → 0.6.18 的同步更新（Dockerfile、versions.json、cuda.txt）。此外 diff 在 `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` 的 `fi_chunk_gated_delta_rule` 中，把 `cu_seqlens` 显式转为 `torch.int64` 后再传给 `chunk_gated_delta_rule_fi`；并把 gpt_oss eval 配置 `gpt-oss-20b-flashinfer-mxfp4-bf16-cutlass.yaml` 的 `server_args` 增加 `--gpu-memory-utilization 0.85`。正文未说明这些附带改动与新版本的因果关系，故机制细节 PR 未注明。Dockerfile 的 diff 上下文显示 `FLASHINFER_VERSION` 用于 `uv pip install --system flashinfer-jit-cache==${FLASHINFER_VERSION}`，`requirements/cuda.txt` 通过 `--extra-index-url https://flashinfer.ai/whl/` 拉取 wheel；`docker/versions.json` 的 `FLASHINFER_VERSION.default` 与 requirements 里的 `flashinfer-python`、`flashinfer-cubin` 三者必须同步，本 PR 正是把这三处一并从 0.6.17 抬到 0.6.18。

## 关键改动

```text
requirements/cuda.txt: flashinfer-python / flashinfer-cubin 由 0.6.17 升到 0.6.18
docker/Dockerfile、docker/versions.json: FLASHINFER_VERSION 由 0.6.17 升到 0.6.18
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py: cu_seqlens 转 int64 后传给 chunk_gated_delta_rule_fi
tests/evals/gpt_oss/configs/gpt-oss-20b-flashinfer-mxfp4-bf16-cutlass.yaml: server_args 增加 gpu-memory-utilization 0.85
```

cu_seqlens dtype 适配片段（摘自 qwen_gdn_linear_attn.py 的 diff）：

```diff
     fi_state = initial_state.to(torch.float32)
     fi_g = g.to(torch.float32)
     fi_beta = beta.to(torch.float32)
+    if cu_seqlens is not None:
+        cu_seqlens = cu_seqlens.to(torch.int64)
     result = chunk_gated_delta_rule_fi(
```

## 性能声明

PR 正文未给出定量性能数据，pack 的 "perf claims detected in body (regex, unverified)" 标注 "(none detected by regex)"；正文没有 benchmark 数字，也没有模型 / 硬件 / 精度 / workload 等测试条件，测试条件 PR 未注明。

## 验证与回归风险

Test Plan 仅 "Full CI nightly test"；Test Result 为空，checklist 的目的、测试计划、测试结果、文档更新四项均未勾选。PR 未注明回归风险与受影响路径；从 diff 看涉及 FlashInfer JIT cache 镜像、CUDA requirements、GDN linear attn 的 `cu_seqlens` dtype，以及 gpt_oss eval 的显存参数。正文的 checklist 四项全部未勾选，也没有给出升级前后的任何对比数据；`Full CI nightly test` 是唯一列出的验证手段，且 Test Result 一节为空，因此本卡无法列出具体通过数量或失败项。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
