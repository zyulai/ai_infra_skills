# PR #53906: [Model] add GLM-5.3-Flash support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ZJY0516 |
| 合并 | 2026-09-03 |
| 分类 | Model-specific Fast Path |
| Label | new-model, rocm, speculative-decoding, ready, torch.compile, ci/build, multi-modality, kv-connector, nvidia, quantization, mrv2, kimi, k3, scheduler, kv-cache-manager, glm |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计;pack 仅保留部分 diff,88 个文件被裁剪) |
| 链接 | https://github.com/vllm-project/vllm/pull/53906 |

## 一句话结论

为 zai-org/GLM-5.3-Flash 提供 day-0 支持,含 kpool KV 压缩、sparse MLA decode、ROCm/NVIDIA 双平台内核等配套,属模型接入而非单点提速。

## 动机

GLM-5.3-Flash 发布时的 day-0 支持需求(主要工作由 @JaredforReal 完成)。官方运行指引见 recipes.vllm.ai 的 GLM-5.3-Flash 配方(建议用 docker image)。PR 声明只有第一个 commit 经过完整验证。

## 优化点

围绕 GLM-5.3-Flash(`glm5next` 架构)接入一整套执行路径:`gather_and_maybe_dequant_cache` 新增 head_dim 512 分支(CALL_GATHER_CACHE_512);indexer 侧引入 kpool 压缩机制——每 16 个 token 的 K/score 经 softmax(gate+ape) 加权求和、Hadamard-128 变换、per-vector FP8 absmax 量化后写入 indexer K cache,decode 用 batched update 内核(按请求内位置顺序迭代,保证 pool 完成时的 read-after-stash 依赖);sparse MLA decode 走 FlashInfer `trtllm_batch_decode_with_kv_cache_mla`(`sparse_mla_top_k`,支持零宽 RoPE 尾),SM90 上用 FA3 wrapper 支持 BF16 query 对 FP8 cache(无 KPE);ROCm 侧 AITER mHC 融合 RMSNorm、kpool cache 布局为 16x16 重排;GDN prefill 传 int64 cu_seqlens。模型主体实现在 pack 中被裁剪,仅见测试与少量 csrc。

## 关键改动

```text
csrc/libtorch_stable/cache_kernels.cu: gather_and_maybe_dequant_cache 支持 head_dim=512
vllm/models/glm5next/{nvidia,amd}/ops/kpool_compress.py: kpool 压缩/decode 更新内核(ROCm 与 CUDA 两套)
tests/kernels/test_kpool_decode_update_batched.py: 批量 decode 更新内核的对拍测试
tests/kernels/attention/test_flashinfer_mla_decode.py: sparse MLA / SM90 FP8 decode 用例
```

## 性能声明

PR 未提供 benchmark 数字(无吞吐/时延对比);本 PR 定位为模型支持与配套内核引入,性能收益未在 PR 内量化。

## 验证与回归风险

新增测试:kpool decode 批量更新内核对纯 torch 参考实现的逐字节对拍(含 fuzz、非均匀 padding、leading invalid sentinel 等边界)及与 prefill writer 的一致性校验(测试注释指出曾有 stash-gating bug 靠这种对拍才能发现);FlashInfer MLA decode 测试按 SM90/SM10x 拆分并新增 sparse 无 RoPE、FP8 cache 用例;ROCm mHC 融合回退路径测试。风险面:day-0 接入,PR 自述仅首 commit 完整验证;diff 规模大(pack 裁掉 88 个文件),kpool/sparse MLA/页布局多路径耦合,对 CI 矩阵(H100/H100 ROCm)有对应扩充。

## 关联

同区域:GLM 系列(glm label)前序支持 PR;MiniMax-M3 sparse indexer(共享 sparse MLA 生态);同批 PR #52737(DeepSeek-V4 mHC AITER 融合)属相邻的 ROCm mHC 线。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
