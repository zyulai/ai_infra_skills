# PR #27988: [Experimental] Full Cuda Graph Support for Prefill

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Oasis-Git |
| 合并 | 2026-07-07 |
| 分类 | CUDA Graph / Compiler |
| Label | run-ci, bypass-fastfail, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 含 8 个文件,另有 1 个文件因超长被丢弃 |
| 链接 | https://github.com/sgl-project/sglang/pull/27988 |

## 一句话结论

把 "full" CUDA graph 后端首次扩展到 prefill(此前 decode-only):每个 num_tokens bucket 捕获一张整 forward 图,小 prefill forward 约 1.7x 提速,收益全部来自 launch 开销。

## 动机

prefill 是变长 workload,此前 `full` 后端(整 forward 单图捕获)在配置校验层被拒绝,只允许 breakable / tc_piecewise。但小型 prefill(短 extend)在快 GPU 上的瓶颈是每 forward 的 kernel launch 与 python 开销,而非 GPU 计算,整图捕获正好消除这部分开销。本 PR 定位 experimental(opt-in),让 operator 自行权衡 padding 浪费。

## 优化点

Token 轴:replay 把 num_tokens pad 到最近的已捕获 bucket(与其他 prefill 后端同一 ladder)。Request 轴:每张图按固定 slot 数(`FULL_CG_PREFILL_REQ_SLOTS=16`,或 `full_prefill_max_req` 自动取 `chunked_prefill_size // 512`)捕获;真实 bs <= slots 时用零长度 sentinel 请求 pad metadata(每次 replay 前清零旧值),一张图服务 bs∈[1,slots];更大 batch 经 `can_run_graph` 回退 eager。捕获只覆盖 transformer body(`layer_model.forward`),LM head + logits_processor 在 eager tail 以 raw bs 运行,使图不受 `(req_slots, vocab)` buffer 绑定。Attention metadata 走 decode 式 2-step 契约(`init_forward_metadata_out_graph` 在 capture/replay 前):FlashInfer 实现为专用 `use_cuda_graph=True` wrapper 组(各 bucket 共享,2GB 级独立 workspace,按 split-kv worst case 公式加 1.25x margin 计算)——split-kv 必须保持开启,其 `block_valid_mask` 是固定捕获网格中 padded/stale tile 唯一的提前退出手段;FlashAttention(fa4)实现为专用 buffer,`cu_seqlens_q` 不与 `cu_seqlens_k` 别名(否则捕获时永久焊死破坏 prefix replay),`max_seq_len_q/k` 以 bucket 上界烘焙;仅支持 page_size=1。其他 attention 后端在启动捕获时明确报错。

## 关键改动

```text
python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py: FullCudaGraphBackend 接入 prefill runner,slot-pad / sentinel 清零 / eager tail 以 raw_bs 运行
python/sglang/srt/layers/attention/flashattention_backend.py: _init_full_cg_prefill_metadata(fa4,独立 buffer,专用 cu_seqlens_q)
python/sglang/srt/layers/attention/flashinfer_backend.py: _create_full_cg_prefill_wrappers + _full_cg_prefill_workspace_bytes(split-kv worst-case 定容)
python/sglang/srt/model_executor/cuda_graph_config.py: Phase.PREFILL 允许 Backend.FULL,新增 full_prefill_max_req 旋钮
python/sglang/srt/server_args.py: cuda_graph_backend_prefill 接受 "full",experimental 警告
test/registered/cuda_graph/full_prefill/test_full_cuda_graph_prefill.py: 集成测试(pinned flashinfer)
```

## 性能声明

- 小 prefill forward(30 tok,Qwen3-8B,B200):fa4 4.07 ms / flashinfer 4.41 ms vs BCG(breakable)7.35 ms,约 **1.7x**;GPU busy 时间相同——收益全部来自 launch 开销。
- 精度(Qwen3-8B,B200,mgsm_en 1319 例,threshold 0.80):flashinfer 0.836-0.84,fa4 0.852;默认配置无回归(0.848)。
- 反面数据:split-kv 若不开启,每次 replay 会按捕获尺寸重跑 attention(实测约 6.5 ms/layer);开启后 padded/stale tile 提前退出(实测 165x 提前的退出收益,PR 原文 "measured 165x")。

## 验证与回归风险

新增 `test/registered/cuda_graph/full_prefill/test_full_cuda_graph_prefill.py` 集成测试(镜像 breakable 集成测试,pinned flashinfer 以保证 CI 硬件可移植;fa4 需 Blackwell、fa3 未测需 SM90 硬件)。风险面:experimental,生产 workload 建议 breakable / tc_piecewise(启动时有 warning);仅 flashinfer 与 FlashAttention(fa4)两个后端实现 extend 捕获契约,其他后端启动时失败;fa4 仅 page_size=1;`can_run_graph` 对 bs > slots、input_embeds、replace_embeds 等场景回退 eager;捕获时烘焙 max_seq_len 上界依赖真实 extent 从 device buffer 读取的正确性。

## 关联

同区域:PR #29458(Breakable CUDA Graph 转 default)、PR #30206(多请求 prefill CUDA graph batch 捕获修复)、PR #33723(elastic-ep scale-up 后重捕获 decode CUDA graphs)。复用既有 `FullCudaGraphBackend`(decode 侧)未改动。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
