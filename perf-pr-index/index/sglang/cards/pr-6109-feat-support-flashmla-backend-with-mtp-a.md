# PR #6109: [Feat] Support FlashMLA backend with MTP and FP8 KV cache

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @quinnrong94 |
| 合并 | 2025-05-15 |
| 分类 | Speculative Decoding |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 + / - 数字);diff 含 8 个文件(两个 attention backend、eagle_worker、cuda_graph_runner、docs、测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/6109 |

## 一句话结论

让 FlashMLA backend 支持 MTP 投机解码与 FP8 KV cache:MTP + FP8 KV cache 组合约 30% 加速,KV cache 占用减半。

## 动机

FlashMLA backend 此前不支持投机解码(deepseek 文档注明 "MTP support for the FlashMLA backend ... is still under development"),也不支持 FP8 KV cache;DeepSeek 系模型在 Hopper 上可选的 attention backend 中,FA3/FlashMLA 是 MLA 主力,缺 MTP 意味着大 batch 下无法用投机解码加速,缺 FP8 KV cache 意味着显存受限时并发与上下文长度受限。FlashMLA 本身具备处理 `seq_len_q > 1` 的能力,可直接用于 target verify 的多 query 行,此前未被利用(旧代码 `Q_LEN = 1` 硬编码)。PR 正文未附逐项 profile,加速比以截图形式给出。

## 优化点

两条线:(1) MTP——`FlashMLABackend.init_forward_metadata` 新增 `is_target_verify` 分支:seq_lens 加 `num_draft_tokens` 重建 KV indices,`get_mla_metadata` 以 `num_draft_tokens * num_q_heads` 作为 q 头数;CUDA graph 的 capture/replay 同步支持 target_verify;新增 `FlashMLAMultiStepDraftBackend`,按 `speculative_num_steps` 包装多个 `FlashMLABackend`(每步一个,共享 kv_indptr 缓冲),供 EAGLE/NEXTN draft decode 多步使用(仅支持 topk=1);`cuda_graph_runner` 对 flashmla 传 `max_bs` 而非 `max_num_token`。(2) FP8 KV cache——`forward_decode`/`forward_extend`(target-verify 路径)在 `kv_cache_dtype fp8_e4m3` 时把 q cast 成 fp8 调 `flash_mla_with_kvcache`,传 `descale_q`/`descale_k`(当前为全 1);flashinfer MLA backend 的 `data_type` 从 `kv_cache_dtype` 改为 `model_runner.dtype`,fp8 KV 时在读取 k buffer 处 `.to(q.dtype)` 转换。另修 `create_flashmla_kv_indices_triton` 的 off-by-one mask(`<=` → `<`),更新 backend 支持矩阵文档。

## 关键改动

```text
python/sglang/srt/layers/attention/flashmla_backend.py: target_verify 分支 + FP8 q cast 路径 + FlashMLAMultiStepDraftBackend
python/sglang/srt/layers/attention/flashinfer_mla_backend.py: data_type 改 model_runner.dtype,fp8 KV buffer 读取时转换
python/sglang/srt/speculative/eagle_worker.py: flashmla 接入 draft backend 分支
python/sglang/srt/layers/attention/utils.py: kv indices 生成 mask 修正
python/sglang/srt/model_executor/cuda_graph_runner.py: flashmla 的 cuda graph 状态按 max_bs 初始化
test/srt/test_flashmla.py: 新增 TestFlashMLAMTP(GSM8K + accept length)
```

## 性能声明

PR 正文仅一句量化:"The speedup of MTP + FP8 KV cache is about 30% with KV cache usage reducing by 50%"(附一张截图,未在正文中给出模型/硬件/batch 等 测试条件)。PR 未注明该数字对应的基准与配置细节,按原文引用。docs 更新声明 FlashMLA 支持 MTP 与 `--kv-cache-dtype fp8_e4m3` 用法(DeepSeek-R1,TP=8)。

## 验证与回归风险

新增 `test/srt/test_flashmla.py::TestFlashMLAMTP`:dsv3 测试模型 + EAGLE(num_steps=1、topk=1、draft_tokens=2)+ flashmla + torch compile,GSM8K 200 题 accuracy > 0.60,`avg_spec_accept_length` > 1.8;既有 latency/throughput 测试保留。风险点:FP8 路径的 `descale_q`/`descale_k` 硬编码全 1(未接真实 scale),数值行为依赖 FlashMLA kernel 内部处理(deepseek-ai/FlashMLA#56 为已知 FP8 问题);draft backend 仅 topk=1,其他配置 raise;`cuda_graph_runner` 的 flashmla 特判是按 backend 名分支,后续 backend 泛化需重构;mask off-by-one 修正影响所有 flashmla 调用点。

## 关联

同区域投机解码 + MLA:投机解码侧 #3582(NextN/MTP)、#6606(EAGLE draft extend CUDA graph)、#6081(MTP DP attention);MLA backend 侧 #34580(AMD triton MLA)、#34647(12-head MLA aiter)、#30971(minimax FP8 attention)、#41533(rocm fused MLA absorb rope)。文档同页提及的 CutlassMLA MTP 支持当时仍 under development。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
