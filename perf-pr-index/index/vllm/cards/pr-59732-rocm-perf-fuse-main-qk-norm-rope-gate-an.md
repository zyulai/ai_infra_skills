# PR #59732: [ROCm][Perf] Fuse main QK-norm/RoPE/gate and KV-cache write into the AMD QSA prepare launch

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mjkvaak-amd |
| 合并 | 2026-10-09 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | rocm, ready, ci/build, qwen |
| Diff | diffstat 缺失(diff 全文已提供,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/59732 |

## 一句话结论

ROCm 上把 Qwen4Exp QSA 主注意力的 QK-norm/RoPE/gate 与 K/V cache 写入并入 QSA prepare 单次 launch,MI355X TP2 输出吞吐最高 +2.9%。

## 动机

NVIDIA 侧已在 #57097 中把 Qwen4Exp 主注意力的 QK-norm、RoPE、gate 拆分和 K/V cache 写入融合进 QSA prepare launch,但 AMD 路径仍执行未融合算子链(split、Gemma RMSNorm、层自身的 `rotary_emb`、`reshape_and_cache_flash`),各环节单独 launch,带来额外 kernel 启动开销和中间张量访存。本 PR 把同样的融合带到 AMD QSA 路径,消除这部分开销。

## 优化点

`amd/ops/qsa_pre_indexer.py` 更名为 `qsa_prepare.py`,在原 Triton kernel 的 indexer 工作之后追加主注意力 program 段:grid 末尾每 (token, Q/KV head) 一个 program,一次完成 Gemma RMSNorm(带 `+1.0` weight 约定)、RoPE(含 MRoPE 三维 position 按 rot%3 区分 h/w)、gate 拆出与 K/V 写入 paged cache(slot<0 跳过)。gate 经新 custom op `qwen4_exp_qsa_prepare_with_output` 写入调用者的 `gate_out` buffer,`* sigmoid(gate)` 仍留在编译图里由 Inductor 融合;cudagraph padding 行的 gate 被 op 置零以保持 `forward_qsa` 的零输出。融合资格(层有 output gate、indexer 支持 fused pre-indexer、主 head dim 为 2 的幂)在 init 时判定,热路径无 CPU/GPU 同步。

## 关键改动

```text
vllm/models/qwen4_exp/amd/ops/qsa_prepare.py: 由 qsa_pre_indexer.py 改名;Triton kernel 追加主注意力段(QK-norm/RoPE/gate/KV 写)
vllm/models/qwen4_exp/amd/indexer_qsa.py: _fused_prepare 接收 attn/qkv/slot_mapping/gate_out,forward 返回 (indices, main_outputs)
vllm/models/qwen4_exp/amd/qsa.py: 新增 qwen4_exp_qsa_prepare_with_output custom op,forward 走融合路径
vllm/config/compilation.py: 把新 op 加入分割算子表
```

核心 kernel 分支节选(每个 program 处理一个 token/head):

```python
if pid >= num_index_work:
    # Main attention: one program per (token, Q or KV head)
    main_pid = pid - num_index_work
    token = (main_pid // (MAIN_HQ + MAIN_HK)).to(tl.int64)
    head = main_pid % (MAIN_HQ + MAIN_HK)
    ...
    if is_k:
        slot = tl.load(main_slots_ptr + token).to(tl.int64)
        if slot >= 0:
            _store_rotated(dst, y, o1, o2, main_k_scale)
```

## 性能声明

MI355X TP2,`amd/Qwen3.8-Flash-Next-Quark-MXFP4`,与不含本改动(#57947)的同镜像做单因素 A/B,两臂轮换 GPU 对、各 2 次:

- ISL 8192 / OSL 1024 / conc 4:输出吞吐 462.0→475.5 tok/s(+2.9%),TPOT 7.39→7.12 ms(−3.7%)
- 同 workload conc 16:+1.2%;conc 64:+0.9%;ISL 60000 / OSL 600 / conc 4:+1.8%
- 4 个测点 × 2 次共 8 个 cell 全部更快(+0.9% 到 +2.9%)

精度:GSM8K 5-shot(flexible-extract)6 对对照,均值 0.9638→0.9605,配对差 −0.33 点,PR 称在 run-to-run 噪声内(单次标准误约 0.005);gate bit-identical,kernel 数值对 fp64 参考不劣于未融合链。

## 验证与回归风险

依赖 `tests/models/qwen4_exp/test_qsa_amd.py` 与 `test_qsa_prepare_amd.py`(由 test_qsa_pre_indexer_amd.py 改名):8 个 bf16 配置下主注意力与未融合链在 bf16 容差内一致(含 MRoPE),gate 逐位相等;gfx950 上 24 passed / 8 skipped——FP8 用例因 AMD QSA cache 仅支持 bf16 而跳过。CI 挂在 AMD `models_basic` 步骤。风险面:仅影响 AMD Qwen4Exp QSA 路径;不满足 eligibility 时回退未融合链;NVIDIA kernel 是独立拷贝不受影响;新 custom op 需留在 torch.compile 分割算子表中。

## 关联

前序:#57097(NVIDIA 侧主注意力融合,本 PR 复用其 kernel 程序)、#57947(two-masked-load state select)。同区域为 Qwen4Exp QSA AMD 路径的持续融合优化;同分类其他 PR 素材未列出编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
