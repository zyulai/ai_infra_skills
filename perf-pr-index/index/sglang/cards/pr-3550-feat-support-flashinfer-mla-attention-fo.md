# PR #3550: feat: support flashinfer mla attention for deepseek v3

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zhyncs |
| 合并 | 2025-02-14 |
| 分类 | Attention / Kernel |
| Label | 无 |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/3550 |

## 一句话结论

接入 FlashInfer 0.2.1 的 MLA attention 用于 DeepSeek V3,为首个集成该内核的开源推理引擎。

## 动机

FlashInfer v0.2.1 发布了新的 MLA attention 内核。SGLang 与 FlashInfer 团队协作完成集成并修复了若干问题,成为首个把 FlashInfer MLA attention 接入 LLM 引擎的开源框架,为 DeepSeek V3 的 MLA decode 提供新的高性能 attention 后端选择(此前 MLA 模型默认被切到 triton 后端)。

## 优化点

新增 `--enable-flashinfer-mla` 开关。flashinfer_backend 检测到 DeepseekV3ForCausalLM 且开关打开时,decode wrapper 换成 FlashInfer 的 `BatchMLAPagedAttentionWrapper`(backend="fa2"),支持 CUDA graph 捕获(qo_indptr/kv_indptr/kv_indices/kv_len_arr 预分配 buffer);decode plan 以 sm_scale=1/sqrt(192)、kv_lora_rank 512、rope 64 维调用 wrapper.plan,decode 前向把 MLA KV buffer 视为 [T,1,head_dim],按 v_head_dim 拆开 nope/rope 部分传入。prefill/extend 路径用 ragged prefill wrapper(head_dim_qk=192、head_dim_vo=128)走 forward_normal,不再做 paged prefill 的 cached-part plan;absorb/normal 的模式选择也随之调整(flashinfer_mla 下 extend 一律 forward_normal,decode 走 forward_absorb)。model_runner 中 MLA 模型的默认后端逻辑改为:开 flashinfer_mla 用 flashinfer,否则维持 triton。依赖升级 flashinfer_python>=0.2.1.post1,CI 安装源切到 flashinfer-python 包,新增 ninja 依赖;新增 set_cuda_arch 工具按设备设 TORCH_CUDA_ARCH_LIST(sm90 加 +PTX)。

## 关键改动

```text
python/sglang/srt/layers/attention/flashinfer_backend.py: decode 换 BatchMLAPagedAttentionWrapper,extend 走 ragged MLA
python/sglang/srt/models/deepseek_v2.py: forward 按 enable_flashinfer_mla 选择 normal/absorb
python/sglang/srt/server_args.py: 新增 --enable-flashinfer-mla
python/sglang/srt/model_executor/model_runner.py: MLA 模型默认后端按开关分流
python/sglang/srt/utils.py: 新增 set_cuda_arch
```

## 性能声明

PR 正文未提供任何吞吐/延迟/准确性数字(checklist 相关项未勾选)。声明仅为定性:首个集成 FlashInfer MLA attention 的开源 LLM 引擎;使用方式为 `--enable-flashinfer-mla --disable-radix-cache`,prefix cache 支持留待后续更新。

## 验证与回归风险

PR 正文无新增测试或评测结果。回归面:改动集中于 flashinfer backend 与 deepseek_v2 前向分发,非 MLA 或未开开关的路径保持原逻辑;flashinfer 版本要求提升(0.2.0.post2 到 0.2.1.post1),旧版本会被 assert_pkg_version 拒绝;当前限制 DeepSeek V3 架构、需 disable radix cache;EAGLE 测试仅调整 cuda_graph_max_bs 参数。

## 关联

依赖 FlashInfer v0.2.1 的 MLA 内核(flashinfer-ai/flashinfer release);正文注明后续更新将加入 prefix cache 支持;同区域后续有其他 FlashInfer MLA/TRT-LLM MLA 后端优化,素材未给出编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
