# PR #43345: [Perf] [XPU] server_args: Enable fused QK-norm + RoPE by default

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mehersai07 |
| 合并 | 2026-10-19 |
| 分类 | Attention / Kernel(major) |
| Label | documentation, intel, xpu, run-ci |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 覆盖 5 个文件(server args 字段、platform hook、两处文档、白名单测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/43345 |

## 一句话结论

把 `--enable-fused-qk-norm-rope` 在 XPU 上改为默认开启,使 #40190 实测 +4.7-7.0% 的 Qwen3-MoE 融合 kernel 收益无需显式 flag 即可生效。

## 动机

#40190 已在 XPU 上为 Qwen3-MoE 接通 `sgl_kernel.fused_inplace_qknorm_rope`,但其收益(q/k-norm + RoPE 阶段 device 时间 -60%)藏在 opt-in flag 后面。该 flag 在 CUDA 上默认关闭有明确理由:CUDA 的 unfused 路径会把 RoPE 与 KV-cache 写(`fused_set_kv_buffer`)融合,fused QK-norm 路径要放弃这一点;而 `enable_fused_set_kv_buffer` 仅限 CUDA/HIP,XPU 的 unfused 路径没有可放弃的东西,融合在 XPU 上是"纯赢"。因此本 PR 把 XPU 的默认值打开,其他平台不动。

## 优化点

无 kernel/模型代码改动,纯默认值与参数形态调整:`enable_fused_qk_norm_rope` 从 `bool = False` 改为三态 `Optional[bool] = None`(BooleanOptionalAction、resolvable),与 `enable_dsa_fused_indexer` 同构;`--enable-fused-qk-norm-rope` 语义不变,新增 `--no-enable-fused-qk-norm-rope` 显式关闭。XPU 的 `handle_xpu_backends` platform hook 在用户未选择时 declare `enable_fused_qk_norm_rope=True`。非 XPU 平台字段保持 `None`,其两个读者(`qwen3_moe.py`、`mellum.py`)只在 `and` 链与真值判断中使用,`None` 行为等价旧 `False`。唯一可见差异:off-XPU 上 `/get_server_info` 从报 `false` 变为报 `null`。

## 关键改动

```text
python/sglang/srt/arg_groups/fields/exec_.py: 字段改三态 Optional[bool]=None,BooleanOptionalAction
python/sglang/srt/arg_groups/platform_hook.py: handle_xpu_backends 默认 enable_fused_qk_norm_rope=True
docs/docs/advanced_features/server_arguments.mdx: flag 行文档更新(XPU 默认开)
docs/docs/hardware-platforms/ascend-npus/reference/support_features.mdx: Ascend 支持表同步
test/registered/unit/test_model_overrides.py: server_args 白名单加入新字段
```

## 性能声明

本 PR 自身无新 benchmark;引用 #40190 的 A/B 结果(Qwen3-30B-A3B,TP=4,4x Intel Arc Pro B60):+4.7-7.0% output throughput(bench_serving,input 1024/output 256,conc 1/8/32 分别 +7.0%/+6.3%/+4.7%),q/k-norm + RoPE 阶段 device 时间 -60%(5,778/102.67ms → 1,938/41.32ms,40 profiled decode steps),TPOT 降 4.6-6.6%。PR 未注明新测试条件,数字均沿用 #40190。

## 验证与回归风险

无新数值路径:kernel 与调用点即 #40190 所测,GSM8K 0.9414 vs 0.9460。参数解析通过 `prepare_server_args` + `resolve_once()` 表格验证:xpu 无 flag→True、显式开/关→True/False;non-XPU 无 flag→None(off)、显式开/关→True/False。nightly `test/registered/xpu/llm_models/test_xpu_qwen3_30b_a3b.py`(TP=4 GSM8K,阈值 0.90)现在默认跑 fused 路径。风险:XPU 用户若依赖 unfused 行为需显式 `--no-enable-fused-qk-norm-rope`;`/get_server_info` 消费方需兼容 `null` 值;CI Base 未运行(:no_entry_sign:),Extra 与 AMD 快照失败。

## 关联

直接前序 PR #40190(XPU 融合接线与性能数据来源)。同区域其他 XPU/attention kernel PR 本卡未收录,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
