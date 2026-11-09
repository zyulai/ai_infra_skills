# PR #47180: [Perf] [AMD] dsa: Replicate the DCP query projection on the owned route

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @EricKing626 |
| 合并 | 2026-11-01 |
| 分类 | Speculative Decoding (tier: major) |
| Label | amd, deepseek, run-ci, jit-kernel |
| Diff | 素材未提供 diffstat;diff 涉及 5 个文件(含 1 个新增测试文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/47180 |

## 一句话结论

DSA DCP 路径复制 Q 投影并"先投影后合并",去掉每层 Q all-gather、a2a 载荷减半,extend 快 13-21%。

## 动机

#44322 建立的 owned DSA DCP 路线上,每个 MLA 层在 decode 和每次 DSA extend 前都要 all-gather 全头 query(`[T, H, 576]`),且 a2a 合并按 `kv_lora_rank`(512)宽传输每个头的部分结果。本 PR 在不改变 attention 计算结果的前提下移除 Q all-gather 并把 a2a 载荷减半:一是 Q 复制——每个 rank 持有整个 DCP 组的 `q_b_proj` 与 `w_kc`,本地投影所有 query 头,`--dcp-replicate-q-proj` 的覆盖范围从 decode 扩展到 HIP 上的 DSA extend;二是"先投影后合并"——LSE 合并对部分输出是线性的,先把每个部分结果 V 投影,a2a 只传 `v_head_dim`(256)而非 `kv_lora_rank`(512)。

## 优化点

三处机制:其一,`model_runner` 在 CUDA Graph capture 前一次性 all-gather `q_b_proj` 与 `w_kc`,本地 `q_b_proj` 权重 alias 到 replica 中自己那一段,复制只增加 (W-1)/W 的权重而非翻倍;GLM 在 gfx95 上的 per-tensor FP8 `w_kc`/`w_vc` 先反量化为 bf16 再 gather,而不是跳过该层。其二,a2a backend 且 `o_proj` 未量化时,每个 rank 用复制的 `w_vc` 对所有组头的部分结果先做 V 投影,合并结果直接进 `o_proj`。其三,token-major absorb:absorb bmm 直接写出 `[T, H, kv_lora_rank]`,省去 head-major 结果的转置拷贝;≤288 token 的批次用小型 Triton per-head bmm(`head_bmm`),更大批次用转置视图上的 `torch.bmm`。

## 关键改动

```text
python/sglang/srt/model_executor/model_runner.py: capture 前 all-gather q_b_proj/w_kc/w_vc 并 alias 本地切片
python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py: is_dcp_q_replicate_phase 覆盖 HIP 上的 DSA extend
python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py: 先投影后合并路径与 token-major absorb bmm
python/sglang/srt/models/deepseek_v2.py: 新增 w_vc_qrep 成员
test/registered/kernels/ops/attention/test_dcp_qrep_hip.py: 新增 project-before-merge 与 head_bmm 数学一致性测试
```

```python
def _dcp_project_before_merge(self) -> bool:
    return (
        get_parallel().dcp_comm_backend == "a2a"
        and self.w_vc_qrep is not None
        and isinstance(self.o_proj.quant_method, UnquantizedLinearMethod)
        and not is_kv_b_lora_active(self)
        and not _SGLANG_EXPERIMENTAL_LORA_OPTI
    )
```

## 性能声明

MI355X,`amd/GLM-5.2-MXFP4`,TP4/EP4 + DCP4 + MTP(模拟 AL 3.61),与 #44322 同日同脚本对比。固定长度探针(131K 缓存前缀):extend 2K TTFT 331 → 287 ms(**-13%**)、4.6K 584 → 485 ms(-17%)、12K 1267 → 1007 ms(**-21%**)、32K 3509 → 2844 ms(-19%);fresh 131K prefill 12.87 → 10.30 s(-20%);decode 16 用户 84.7 → 87.1 tok/s/user(+2.8%)、24 用户 70.0 → 72.4(+3.4%)。InferenceX agentic 回放(1 h/点):TPUT per Chip c16 17,196 → 18,125、c24 23,416 → 25,151、c40 24,740 → 26,583、c48 10,774 → 12,819(1.19x);P90 Interactivity 每点均为 #44322 的 1.15-1.25x(如 c16 41.19 → 51.38 tok/s/user)。收益主要来自 prefill 提速 13-21% 从而缩短其后的 decode 停顿,decode 本身约快 3%。

## 验证与回归风险

新增 `test_dcp_qrep_hip.py`:`head_bmm` 在 absorb 与 V 投影 shape 下匹配 `torch.bmm`,project-then-merge 与 merge-then-project 相对误差 < 1e-2;另跑既有 DCP kernel/单元测试、GLM-5.2 DCP4 端到端 sanity(passkey 至 96K、NLL)、GSM8K(400 题 0.953 → 0.945/0.940,波动 ±0.02;1,319 题 0.935 → 0.933)、MTP accept length 3.23 → 3.26、NLL 1.232 → 1.236/1.227。风险:复制带来每层 (W-1)/W 的 `q_b_proj` 与 `w_kc`/`w_vc` replica 显存开销;先投影后合并仅在 a2a backend、`o_proj` 未量化、无 kv_b lora 时生效,不支持的 `w_kc` 回退到逐层 Q all-gather 并打警告;不开 DCP 或 `--no-dcp-replicate-q-proj` 均走原路径,CUDA 保持 decode-only 行为。pack 记录 CI:Base 通过、Extra 失败、AMD ROCm 10 通过。

## 关联

前序:#44322(owned DSA DCP route,本 PR 直接基于其代码与 benchmark 表)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
