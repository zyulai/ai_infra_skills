# PR #27868: fix(qwen3.5): keep CUDA dual-stream overlap (regressed by #25885)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @YAMY1234 |
| 合并 | 2026-06-15 |
| 分类 | Overlap / Scheduler (tier: major) |
| Label | run-ci, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 数;diff 本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/27868 |

## 一句话结论

修复 #25885 把 Qwen3.5 三个 CUDA 双流 overlap 误关的回归:恢复后 8xH200 上 decode 吞吐 377 -> 434 tok/s(约 +15%),回到 #25885 之前的基线。

## 动机

#25885("[AMD] Support alt stream for Qwen3.5 on AMD platform")在为 AMD 接入 alt stream 环境变量开关时,把 `qwen3_5.py` 里三个原本在 CUDA 上无条件启用的双流(`alt_stream`)overlap 也一并 gating 掉了:`_gdn_use_alt_stream` 与 `_qknorm_use_alt_stream` 都要求 `_is_hip`(及对应环境变量),MLP/shared-expert 的 `alt_stream` 在默认的 shared-experts-fusion 路径被强制 `None`。这些 overlap 因此在所有 CUDA GPU 上被静默禁用且无环境变量可重新打开,造成 Qwen3.5 decode 吞吐回归;H200 只是 CI 上最先观测到,Blackwell/Ampere 同样受影响。作者用 `git bisect` 在 Qwen3.5-FP8 conc4/isl8192/osl1024 工作负载(H200, TP8/EP8)上定位出 #25885 的 `7f919ed` 为首个坏 commit。

## 优化点

以最小条件修补恢复 #25885 之前在 CUDA 上的行为,同时保留 AMD 的新环境变量 gating:三处布尔条件从"仅 AMD 开关"改为"或上 `_is_cuda`"——`_gdn_use_alt_stream = _is_cuda or (env && _hip_use_alt_stream)`、`_qknorm_use_alt_stream = _is_cuda or (...)`、MLP 与 shared-expert 的 `alt_stream` 改为 `(alt_stream if (_is_cuda or _disable_shared_experts_fusion()) else None)`。即 CUDA 上恢复无条件启用 GDN QKVZ/BA 投影、QK norm、shared-expert MLP 三条双流 overlap;AMD 上仍由 `SGLANG_GDN_QKVZ_BA_ALT_STREAM`/`SGLANG_QK_NORM_ALT_STREAM`/`SGLANG_ALT_STREAM` 控制。不改变前向计算本身,只恢复流级并行。

## 关键改动

```text
python/sglang/srt/models/qwen3_5.py: 三处 alt_stream 条件恢复 _is_cuda 或上 AMD env gating(GDN input_proj、QK norm、两处 MLP/shared-expert 构造)
```

关键片段:

```python
_gdn_use_alt_stream = _is_cuda or (
    get_bool_env_var("SGLANG_GDN_QKVZ_BA_ALT_STREAM", "False") and _hip_use_alt_stream
)
_qknorm_use_alt_stream = _is_cuda or (
    get_bool_env_var("SGLANG_QK_NORM_ALT_STREAM", "False") and _hip_use_alt_stream
)
```

## 性能声明

测试条件:Qwen3.5-397B-A17B-FP8,8xH200,TP8/EP8,flashinfer attention,random isl=8192 / osl=1024,concurrency=4,`--quantization fp8 --kv-cache-dtype fp8_e4m3`:

- Before(current main):Output 377 tok/s,Mean TPOT 9.97 ms
- After(this PR):Output 434 tok/s,Mean TPOT 8.60 ms
- 约 15% decode 吞吐恢复,与 #25885 之前基线一致(通过在同一镜像中 checkout 父 commit 验证);profiling 显示 `memcopy` 等恢复到双流路径(PR 附截图)

## 验证与回归风险

GSM8K(`sglang.test.run_eval --eval-name gsm8k --thinking-mode qwen-3 --max-tokens 16384`,chat API + thinking,1319 例,Qwen3.5-397B-A17B-FP8/H200):main 0.979 vs 本 PR 0.977,在 run-to-run 方差内(改动只恢复流重叠、不改前向计算)。未注明新增单元测试;CI Base 通过、Extra 失败标记。回归面:仅 Qwen3.5 模型文件的 alt_stream 条件;AMD 路径语义保持(环境变量仍生效);CUDA 上 GDN/QK-norm/MLP 三条双流 overlap 重新启用,需留意这些流路径在 CUDA 上的既有正确性(它们在 #25885 之前即为 CUDA 默认行为,本 PR 属于恢复)。

## 关联

直接前序:#25885(引入 AMD alt stream 支持并造成本回归)。同区域为 Qwen3.5 双流 overlap 调度;pack 未列出其他同分类 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
