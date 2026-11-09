# PR #55331: [Perf] Read VidCom2 frame budgets once

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Levius-Fubuki |
| 合并 | 2026-09-04 |
| 分类 | Model-specific Fast Path \| tier: major |
| Label | ready, multi-modality |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/55331 |

## 一句话结论

VidCom2 把 T 次 `item()` 降为 1 次 `tolist()`，RTX 4090 全矩阵几何均值加速 1.181x。

## 动机

VidCom2 的 token-selection 路径用 `ks[i].item()` 逐帧读取动态 per-frame Top-K budget，再用一次标量 reduction 读选中 token 数；对 `T` 帧共引入 `T + 1` 次 host-visible 标量读取。budgets 已被 clamp 到至少 1，原先的非正 budget 分支不可达。作者搜索 `VidCom2 topk`、`VidCom2 frame budget`、`video prune synchronization` 未发现重复工作。

## 优化点

把小 budget 向量一次性拷成 Python list（`budgets = ks.tolist()`），循环改为 `for i, k_i in enumerate(budgets)`，保留每帧原有的一维未排序 `torch.topk` 操作不变；count reconciliation 处把 `int(flat_mask.sum().item())` 换成 `sum(budgets)`，复用预算和。改动保持单一可移植 PyTorch 路径，无自定义 kernel、架构门控、设备分支或设备特定调优。

## 关键改动

```text
vllm/multimodal/video_prune/vidcom2.py: compute_retention_mask 用 ks.tolist() 一次读取预算、sum(budgets) 复用，移除逐帧 item() 与 mask.sum().item()。
tests/multimodal/test_vidcom2.py: 新增 test_mask_reads_frame_budgets_once，monkeypatch 拒绝 Tensor.item() 并断言恰好一次 (4,) 向量传输。
```

```python
budgets = ks.tolist()
mask_2d = torch.zeros(T, tokens_per_frame, dtype=torch.bool, device=device)
for i, k_i in enumerate(budgets):
    _, idx = torch.topk(similarity[i], k=k_i, largest=False, sorted=False)
    mask_2d[i].scatter_(0, idx, True)

flat_mask = mask_2d.view(-1)
flat_sim = similarity.view(-1)
current = sum(budgets)
```

## 性能声明

完整函数基准（NVIDIA RTX 4090、driver 580.76.05、CUDA 13.2、PyTorch 2.13.0+cu132，baseline `c615b1fd`，warmups 25 / iterations 100，dtype FP16/BF16/FP32、T=1–128、tokens/frame=48/196/576、q=0.25–0.9 共 288 配置）：p50 speedup 举例 BF16 T=1/48/0.50 为 1.04x、BF16 T=32/196/0.50 为 1.26x、FP32 T=64/576/0.25 为 1.52x、BF16 T=128/576/0.75 为 1.44x；全 288 配置原始几何均值 1.181x、中位数 1.142x，所有 mask 逐位一致。T=1 有 1 个原始 timing 离群点，5 组交替顺序 500 迭代复测中位数 1.041x（1.037–1.041x），无超过 3% 的稳定回归。profiler（每 20 次完整 mask 调用）显示 item 调用 60/180/660 → 0，CUDA memcpy 事件 60/220/700 → 20/60/60，Top-K 与 scatter 次数不变。Qwen3-VL-2B-Instruct 配对评测：8 帧 operator 1.061→0.836 ms（1.27x）、E2E 647.60→645.29 ms（1.004x）；32 帧 operator 1.549→1.196 ms（1.29x）、E2E 644.90→644.57 ms（1.001x），mask 与生成 token/text/logprob 一致。

## 验证与回归风险

回归测试：`.venv/bin/python -m pytest -q tests/multimodal/test_vidcom2.py`，Test Result `33 passed`；`ruff-format`、`ruff-check`、typo、SPDX-header、forbidden-import、new-`torch.cuda` API、`git diff --check` 本地通过；`pre-commit run --files tests/multimodal/test_vidcom2.py vllm/multimodal/video_prune/vidcom2.py`。限制（正文自述）：宽矩阵基于单张 RTX 4090 上的合成 post-ViT embeddings，两个 Qwen3-VL 用例只验证输出稳定性，不证明端到端吞吐提升，优化 helper 仅占总延迟很小一部分；未对其他架构或后端作性能声明。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
