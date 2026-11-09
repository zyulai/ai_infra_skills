# PR #61640: [Perf][MRV2] Skip empty draft-token resets on request admission

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jzakrzew |
| 合并 | 2026-10-14 |
| 分类 | Speculative Decoding |
| Label | performance, ready, mrv2 |
| Diff | diffstat 缺失(pack 未提供 +/− 统计,仅含源码 diff) |
| 链接 | https://github.com/vllm-project/vllm/pull/61640 |

## 一句话结论

MRV2 请求接纳时跳过零宽 draft_tokens 行的 zero_,GH200 短输入 embedding 吞吐 +10.6~28.2%。

## 动机

GH200 上短输入 embedding 推理在高并发下由 EngineCore 主线程 CPU 瓶颈:吞吐平台约 5–6k req/s(~85–100k tok/s),GPU SM 利用率仅 ~9–18%(对照:1024-token 输入可达 ~400k tok/s)。`RequestState` 的 `draft_tokens` 形状为 `[max_num_reqs, num_speculative_steps]`,admission 代码无条件执行 `self.draft_tokens[req_idx].zero_()`——即使该行零元素(Model Runner V2 零 speculative steps)也会做一次 PyTorch 索引加 op。

## 优化点

在 `add_request` 中加 guard:`num_speculative_steps > 0` 才执行 indexing + `zero_()`。对 pooling 与普通非 speculative 生成,请求接纳路径上这一不必要的 PyTorch op 被完全移除;凡 buffer 有列的场景(包括 speculative decoding 与 diffusion 模型)均保留 #39833 引入的 stale-slot 清零语义。收益机制不在 GPU:该 op 会释放 GIL,主线程随后在 `pthread_cond_timedwait` 中等待重新获取,导致其他线程在主线程本可等待 GPU 的时间窗内抢占——去掉它主要是降低 GIL 争用,让 CPU/GPU 工作更好重叠。

## 关键改动

```text
vllm/v1/worker/gpu/states.py: add_request 中 draft_tokens 清零加 num_speculative_steps guard
```

```python
-        self.draft_tokens[req_idx].zero_()
+        if self.num_speculative_steps > 0:
+            self.draft_tokens[req_idx].zero_()
```

## 性能声明

复用既有 pinned GH200 embedding benchmark(每配置六次运行、两次反向顺序 pass):短输入 conc=512 request throughput ~4.66k→~5.15k req/s(**+10.6%**);conc=2048 ~5.57k→~6.96k / ~7.14k req/s(**+25.0% / +28.2%**,两 pass)。配置:`nvidia/Nemotron-3-Embed-1B-BF16`、FP8 量化、v0.31.0 aarch64 container、单 GH200、MRV2 同步调度、16 个 API server、随机唯一短文本;engine cores 4–5、API cores 8–39、client cores 40–71 钉在 GPU 0 的 NUMA 节点。stock 基线重复区间 4741–4965 req/s,per-patch 原始波动 PR 注明不可得。

## 验证与回归风险

运行 `pytest tests/v1/worker/test_gpu_model_runner_v2.py -v -rs`、pre-commit、`git diff --check`,PR 未附精度评测。风险极低:零 speculative steps 时原 zero_ 作用于空行本就无效果,行为等价;非空 buffer 的清零语义不变(speculative/diffusion 路径不受影响)。

## 关联

#39833(其 stale-slot reset 语义被本 PR 保留)。同分类( speculative decoding)还有 #54687,但属 kernel 层不同区域。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
