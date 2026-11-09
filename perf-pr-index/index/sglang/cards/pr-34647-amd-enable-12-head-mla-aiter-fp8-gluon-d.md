# PR #34647: [AMD] Enable 12-head MLA aiter fp8 Gluon decode (batched bh16bn128).

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @chien-an-chen |
| 合并 | 2026-09-01 |
| 分类 | Quantization(major) |
| Label | amd, run-ci, jit-kernel, amd-aiter-not-ready, run-ci-extra |
| Diff | 素材未注明 diffstat(diff 片段可见 5 个文件,含 2 个新增) |
| 链接 | https://github.com/sgl-project/sglang/pull/34647 |

## 一句话结论

在 MI355 上为 Kimi-K3 TP8(每卡 12 头 MLA)启用 aiter fp8 Gluon decode,长上下文输出吞吐提升 94%–168%。

## 动机

Kimi-K3 在 TP8 下每卡只暴露 12 个 MLA head,而 aiter 快速 persist decode ASM 要求 head 数能被 16 整除,且此前没有原生 fp8 12-head 路径,fp8 服务只能 zero-pad 到 16 头走较慢的内核;在 decode CUDA graph 下的长上下文场景尤其痛苦。aiter 已有带原生 12-head masked MFMA tile、面向 fp8 KV(bh16bn128)的 Gluon 内核,本 PR 把它接进来。

## 优化点

新增 `aiter_mla_gluon.py` 薄封装:运行时探测 aiter `mla_gluon` 可导入且 Triton Gluon 提供 `cga_layout`(需 Triton >= 3.7),任一条件不满足即返回 None,由调用方回退到原有 zero-pad + `mla_decode_fwd` 路径,旧容器行为不变。`aiter_backend.py` 把 12 头纳入合法 head 集合并引入 `head_pad_mode`(4/8 头 repeat、12 头 zero、其余 none);decode 路由改走 `_forward_mla_decode`,在 `max_q_len==1`、h12、fp8 KV 同时成立时优先调用 Gluon 内核,并针对 h12+fp8 关闭慢速的 16 头 persist(PS)内核;`min_kv_seq_len` 在 CUDA graph 捕获期间取 `max_context_len` 以避免 GPU→CPU 同步。prefill 侧仅将 h12 的 Q zero-pad 到 16 头后再走 aiter MLA prefill 内核,并保持 Q 为 bf16(Gluon 要求)。新增环境开关 `SGLANG_AITER_MLA_GLUON`(默认开),置 0 可强制回退。

## 关键改动

```text
python/sglang/srt/layers/attention/aiter_mla_gluon.py: 新增;Gluon decode 封装 + 能力探测(import/cga_layout),失败返回 None 触发回退
python/sglang/srt/layers/attention/aiter_backend.py: h12 head 校验与 head_pad_mode、decode 路由 _forward_mla_decode、graph-safe min_kv_seq_len、h12+fp8 禁用 PS 内核、prefill Q zero-pad
python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py: fp8 KV + aiter 后端时 prefill 保持 bf16 Q
python/sglang/srt/environ.py: 新增 SGLANG_AITER_MLA_GLUON 开关
test/registered/attention/test_mla_gluon_h12_fp8.py: 新增;CPU 单测覆盖探测门控与回退分发
```

```python
def prefer_mla_gluon_decode(*, head_pad_mode, num_head, kv_cache_dtype) -> bool:
    if not _mla_gluon_enabled():
        return False
    if head_pad_mode == "zero" and num_head == 12 and kv_cache_dtype == fp8_dtype:
        return _gluon_runtime_ok()
    return False
```

## 性能声明

- 精度:MI355 + Kimi-K3,`--kv-cache-dtype fp8_e4m3 --attention-backend aiter`,decode CUDA graph 开启(rocm720-mi35x-20260811 镜像 + aiter fb82bf7e),GSM8K 95.5–95.6%。
- 长上下文(ISL>50K,radix cache ON,主收益):output token/s/gpu 相对 baseline 提升 94%–168%。Baseline 为 triton attention backend + bf16 KV(rocm720-mi35x-20260805 镜像);本 PR 栈为 fp8 KV + Gluon decode + aiter prefill(rocm720-mi35x-20260811 镜像)——即整栈对比,非本 PR 单独贡献。
- 8k1k(non-Dspark,disable-radix-cache):c=4/8/16/32 时 TTT 提升 4–10%;TP8 下并发 32 时 TTT per GPU 741.1 tok/s(0.805 基线的 1.10×),median ITL 为基线 0.85×。PR 注明与 0805 镜像对比是因为 0811 镜像 triton 在高并发下 TTT 回退约 10%。

## 验证与回归风险

新增 `test_mla_gluon_h12_fp8.py` 为纯 CPU mock 测试(注册进 `base-a-test-cu` 套件),覆盖 env 关闭、import/cga 缺失时的探测结果与 `_forward_mla_decode` 的回退分发。依赖未在本仓库 pin 的外部组件:ROCm/aiter #4480(fp8 Gluon bh16bn128 batch>1,已合并)、#4555(decode CUDA graph KV-split 预算)、Triton >= 3.7,label 亦带 `amd-aiter-not-ready`;pack 内 CI 状态显示 PR Test 与 AMD ROCm 7.2 均为失败态。范围限定 decode(`max_q_len==1`、h12、fp8 KV);gfx950 上 12 头模型的 fp8 PS-ASM prefill 因内存错误被关闭、改走 flash-attn 回退,影响该拓扑的 prefill 路径;另含 target_verify 处 `draft_num` 取值与 `seqused_k` 计算的小改动,影响 speculative 相关路径。

## 关联

同区域优化:PR #34619(同系列 12-head MLA Aiter fp8 Gluon decode);PR #34837(h=12 KV write,已在 main)。外部依赖:ROCm/aiter #4480、#4555;参照 vLLM #50371(zero-pad 风格)与 #50563(bf16 Q)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
