# PR #38935: [PD][HeteroArch]Fix accuracy issue with CPU_ATTN as Decoder and Flash_ATTN as prefiller

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38935](https://github.com/vllm-project/vllm/pull/38935) |
| 作者 | xuechendi |
| 合并日期 | 2026-04-09 |
| labels | intel-gpu, ready, v1, cpu, kv-connector |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
CPU 作 decoder 收到的 FlashAttn plain KV 缺 CPU_ATTN 所需 packing;NixlConnector 收包后按 CPU 方法 pack KV 修复精度。

## 问题现象
Hetero Arch PD 分离场景:CPU 作 decoder、CUDA/Intel GPU/Intel Gaudi 等作 prefiller 时出现精度问题(正文称修复 #38710 报告的错误)。测试以 Qwen/Qwen3-0.6B 经 NixlConnector(kv_role=kv_both, kv_buffer_device=cpu)做 prefill(CUDA/XPU)/decode(CPU)分离,chat completion 正常返回,并用 lm_eval gsm8k(limit 256)验证精度。

## 根因
正文披露:CPU KV layout 需要额外的 `packing` 步骤,而 PD 场景下从远端收到的是 FlashAttn 的 plain KV,未按 CPU_ATTN 的 KV packing 方法处理,导致精度问题。

## 修复方案
1. `nixl_connector.py`:新增 `post_process_device_kv_on_receive_heterogeneous_attn()`,对接收到的 KV block 调用 `current_platform.pack_kv_cache()`;握手元数据 `NixlAgentMetadata` 增加 `attn_backend_name` 字段并校验远端 backend;
2. `enable_heterogeneous_attn_post_process` 仅在本地 backend 为 `CPU_ATTN` 且远端不同 时开启(HMA 场景直接报错不支持);
3. `platforms/cpu.py` 新增 `pack_kv_cache()`:KV 做 permute 后调用 `cpu_attn_reshape_and_cache` 写入 packed cache;
4. `get_finished()` 中对完成接收的请求触发上述后处理。

## 影响与回归风险
- 影响窗口:正文未说明(对应问题报告见 #38710)。
- 回归面:Nixl 握手 payload 新增 `attn_backend_name` 字段(新旧版本握手兼容性需对端同步);`get_finished` 收包完成后新增 packing 后处理步骤,仅在 CPU_ATTN 接收非 CPU_ATTN KV 时激活;CPU 平台新增 `pack_kv_cache` 方法;单架构(同 backend)部署不触发新路径。

## 关联
- #38710(修复其报告的错误)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
