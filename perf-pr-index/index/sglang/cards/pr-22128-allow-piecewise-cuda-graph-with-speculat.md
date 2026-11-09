# PR #22128: Allow piecewise CUDA graph with speculative decoding

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @narutolhy |
| 合并 | 2026-04-17 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, deepseek, speculative-decoding, run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 行数与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/22128 |

## 一句话结论

解除 PCG 与 speculative decoding 的互斥:MTP decode 加速之上叠加 PCG prefill 加速,rate=5 TTFT 253→147 ms(-42%)。

## 动机

#16331 在 piecewise CUDA graph(PCG)转为默认开启时,把"开了 speculative decoding 就整体禁用 PCG"作为保守安全措施加上(原始 PCG 实现 #10062 并无此限制),但从未实际测试过两者是否兼容。事实上 PCG 与 spec decode 工作在**相互独立的 forward 路径**上:PCG 捕获/重放的是 prefill/extend(`ForwardMode.EXTEND`,spec_info=None)的图;speculative 的 draft/verify 走 decode CUDA graph(`TARGET_VERIFY`)。一刀切禁用使所有 spec 用户白白损失 PCG 的 prefill 加速。

## 优化点

改动是"解除互斥 + 补上真正需要的隔离 guard"。其一,`server_args.py` 删除 `speculative_algorithm is not None → disable_piecewise_cuda_graph` 的整段禁用(后续禁用项编号顺移)。其二,`model_runner.init_piecewise_cuda_graphs` 对 draft worker 提前返回——draft 模型只用 decode CUDA graph,不捕 PCG 图。其三,`piecewise_cuda_graph_runner.can_run` 增加两条拒绝条件:TARGET_VERIFY 模式不得使用 PCG 捕获的图(PCG 以 EXTEND + spec_info=None 捕获,verify 的 spec_info 与 capture_hidden_mode 不同);batch 需要的 `capture_hidden_mode` 与 runner 捕获时不一致(如 spec 需要 FULL)时拒绝重放,否则会返回错误/缺失的 hidden_states。这两条 guard 把"PCG 只服务 extend 路径"的边界显式化,替代原先的整体禁用。

## 关键改动

```text
python/sglang/srt/server_args.py:
  _handle_piecewise_cuda_graph 删除 spec decoding 禁用分支
python/sglang/srt/model_executor/model_runner.py:
  init_piecewise_cuda_graphs 对 is_draft_worker 提前返回
python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py:
  can_run 拒绝 target_verify 与 capture_hidden_mode 不匹配的 batch
test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py(新):
  MTP/EAGLE3/STANDALONE/NGRAM 四类 PCG 共存 e2e 测试
```

```python
# TARGET_VERIFY 有不同的 spec_info 和 capture_hidden_mode,不得重放 PCG 图
if forward_batch.forward_mode.is_target_verify():
    return False
if forward_batch.capture_hidden_mode != self.capture_hidden_mode:
    return False
```

## 性能声明

Benchmark(Qwen3.5-35B-A3B FP8,TP2,H100,全部配置 `--mamba-scheduler-strategy extra_buffer`):baseline(无 overlap、无 PCG、无 MTP)rate=1 ITL 21.95 ms;extra_buffer + MTP:rate=1 ITL 3.00 ms、rate=5 TTFT 253 ms、rate=5 ITL 6.06 ms;extra_buffer + PCG + MTP(本 PR):rate=1 ITL 2.89 ms、rate=5 TTFT 147 ms、rate=5 ITL 5.56 ms。即在 MTP decode 加速之上,PCG 贡献 prefill 加速:TTFT(rate=5)253 → 147 ms(-42%)。未注明请求长度/workload 细节。

## 验证与回归风险

Accuracy(GSM8K,每配置 50 题):EAGLE/NEXTN(Qwen3.5-35B-A3B FP8 TP2)PCG 0.980 = no-PCG 0.980,acceptance 3.46;EAGLE3(Qwen3-30B-A3B TP2)PCG 1.000,acceptance 4.23;STANDALONE(Qwen3-8B FP8 TP2)PCG 0.380 vs no-PCG 0.400(差异归因于弱 draft 模型 Qwen3-0.6B,与 PCG 无关)。新增 e2e 测试覆盖四种 spec 算法与 PCG 共存(含 avg_spec_accept_length > 1.5 断言);作者注明 CI 项未完成([ ])。风险面:guard 之外的隐式耦合(capture_hidden_mode 相同但 spec_info 形状不同的扩展路径)可能被未来的 forward mode 绕过;PCG 默认开启意味着所有 spec 用户现在默认走进新组合路径,NGRAM 等算法的覆盖主要靠新测试;非 CUDA 硬件、DP attention、torch compile 等原有禁用项不变。

## 关联

本 PR 解除的前序限制来自 #16331(PCG 默认化时的安全禁用),PCG 原始实现为 #10062;同区域:本批次 #32219(spec-v2 host-seam)、#32017(startup 阶段 CUDA graph capture 重叠)。pack 未列出其他直接关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
