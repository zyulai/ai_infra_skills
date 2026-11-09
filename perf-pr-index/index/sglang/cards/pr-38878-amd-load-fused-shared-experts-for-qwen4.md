# PR #38878: [AMD] Load fused shared experts for Qwen4-Exp and Qwen3.5 MTP

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yichiche |
| 合并 | 2026-09-17 |
| 分类 | Speculative Decoding(major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 显示 2 个文件:qwen4_exp.py(+26/−2)、qwen3_5_mtp.py(+13/−8,正文注明) |
| 链接 | https://github.com/sgl-project/sglang/pull/38878 |

## 一句话结论

修复 Qwen3.8-Flash-Next 在 shared-expert fusion 下共享专家权重被静默丢弃的问题,MTP accept length 恢复 +3.1%(等比吞吐收益)。

## 动机

`Qwen3.8-Flash-Next` 在 shared-expert fusion 激活时加载,其共享专家权重处于未初始化状态,主模型与 MTP draft 各有一处独立实例。主模型:fusion 开启后模型没有独立 `mlp.shared_expert.*` 参数——共享专家住在 fused MoE 的路由槽 `num_experts`,而 checkpoint 仍按独立张量发布,`Qwen4ExpForConditionalGeneration.load_weights` 没有对应映射,每 TP rank 丢弃 144 个张量(3 projections x 48 层)、TP8 共 1152 个,服务器照常启动并生成流畅文本,但 48 层共享专家贡献全是垃圾。MTP draft:`qwen3_5_mtp.py` 在 stacked-params 循环内第一次见到 fused 张量时把 `expert_params_mapping` 永久重绑为 `fused_expert_params_mapping`,该 checkpoint 交错两种形式(49 个 fused + 196 个独立共享专家张量),导致其后所有共享专家张量查错 mapping 而被丢弃。ROCm 镜像默认 `SGLANG_USE_AITER=1`,fusion 开箱即开,该模型的每次 ROCm 运行都命中两处。

## 优化点

权重加载修复,不是 kernel 优化(收益体现为 MTP accept length 恢复带来的等比吞吐提升):(1) `qwen4_exp.py` 从已构建模块发现 `num_fused_shared_experts`,用 `num_experts + num_fused_shared_experts` 构建 `expert_params_mapping` 覆盖路由槽 `num_experts`,并在 stacked/expert 循环之前把 `mlp.shared_expert.` 重写为 `mlp.experts.{num_experts}.`,复用 `qwen3_5.py` 对同一问题的既有做法(`FusedMoE.make_expert_params_mapping(num_experts=513)` 已能发出所需条目,如 `('experts.w13_', 'experts.512.gate_proj.', 512, 'w1')`)。(2) `qwen3_5_mtp.py` 改为对每个权重计算 `is_fused_expert` 并据此选择 `current_expert_params_mapping`,不再在循环内重绑共享局部变量,使一个权重的 mapping 选择不影响下一个。两处 loader 都以 `_use_aiter`(`SGLANG_USE_AITER and is_hip()`)为门:CUDA 上 `enable_cuda_shared_expert_fusion` 对这些模型默认 False,故对 CUDA 影响可证明为零;作者明确说明精确谓词应是 `num_fused_shared_experts > 0`,若将来有人为这些模型开启 CUDA fusion,两处 gate 需要一并放开。

## 关键改动

```text
python/sglang/srt/models/qwen4_exp.py: 主模型 loader 覆盖 fused shared-expert 槽位
  (mapping 扩到 num_experts + num_fused_shared_experts)并重写 mlp.shared_expert. 名字
python/sglang/srt/models/qwen3_5_mtp.py: MTP draft loader 改为 per-weight 选择
  current_expert_params_mapping,修复 mapping 永久重绑
```

关键摘录(MTP mapping 选择):

```python
is_fused_expert = (
    "experts.gate_up_proj" in name or "experts.down_proj" in name
)
current_expert_params_mapping = (
    fused_expert_params_mapping
    if is_fused_expert
    else expert_params_mapping
)
```

## 性能声明

无 kernel/稳态加速声明(权重加载只跑一次)。可测效果为 MTP accept length 恢复:3.164 vs 3.067(+3.1%),等比吞吐收益;PR 明确注明 reverted 变体的吞吐未采集,仅引用 accept-length 差值作证据。本次 GSM8K 运行本 PR 测得 1608 tok/s 输出吞吐(仅绝对值,无对照)。gate 与 ungated 在 ROCm 上等价(accept length 3.164 vs 3.157,运行间噪声内),gate 无成本。

## 验证与回归风险

8x MI355X(gfx950),BF16 Qwen3.8-Flash-Next,`--tp-size 8 --attention-backend aiter --page-size 32 --chunked-prefill-size 16384`,EAGLE MTP(3 steps, topk 1, 4 draft tokens),GSM8K 200 例逐部分消融:本 PR 0 warning / GSM8K 0.980 / accept length 3.164;仅 revert qwen3_5_mtp.py 改动时 24 个 warning / 0.975 / 3.067;两者都 revert(现状)时 1152+24 个 warning、指标不可评分。作者指出 MTP 行是关键证据:draft 共享专家被丢时 accept length 掉 3.1% 而 GSM8K 不变——EAGLE 对 target 的 verification 保证输出分布正确,broken draft 只损失吞吐不损精度,精度 gate 抓不住这类问题。18k-token prompt 检索检查通过。未新增单元测试(作者提出可补 CPU 单测);CI Extra 与 AMD ROCm 7.2 失败(pack 记录)。

## 关联

同区域:复用 `qwen3_5.py` 对同一问题的既有机制;#23620、#23611 为同 ROCm/AMD MiniMax 系列优化(不同模型)。pack 内未列出其他直接关联。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
