# PR #41233: [Bugfix][Hybrid][NemotronH] Fix mamba_cache_mode=all + speculative decoding crash

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#41233](https://github.com/vllm-project/vllm/pull/41233) |
| 作者 | roikoren755 |
| 合并日期 | 2026-05-18 |
| labels | bug, ready, v1 |
| 热度 | 👍 1 · 💬 6 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
`mamba_cache_mode=all` + 投机解码时块表未预留 spec 槽位导致 NemotronH 等 hybrid Mamba2 崩溃;三处簿记修复并回退 #40454 的临时规避。

## 问题现象
prefix caching 的 `all` 模式与投机解码(speculative decoding)组合时,NemotronH 及其他 hybrid Mamba2 模型崩溃——这是 #39809 报告的三个 bug 的链条。4×GB200 上的端到端复现命令:`vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 --enable-prefix-caching --mamba-cache-mode all --trust-remote-code --tensor-parallel-size 4 --speculative-config '{"method":"mtp","num_speculative_tokens":1}'`。修复前两个新增单测在 main 上失败(`assert torch.Size([4, 16]) == (4, 18)`、`assert torch.Size([4]) == (3,)`);修复后 server 正常启动、gsm8k 全矩阵(PC-off/align/all × CUDA graphs/enforce-eager)约 92% 精度,重复发送 >4224-token prompt 时 `vllm:prefix_cache_hits_total` 正常增长(8448/10926 cached tokens),align/all 两种模式前缀缓存命中率均约 75%。

## 根因
PR 正文披露:三个问题共享同一根因——`mamba_cache_mode='all'` + spec decode 下,kernel 每个请求读写 `1 + num_speculative_blocks` 个连续 state 槽位,但周边簿记把块表(block table)与每请求索引当作没有预留投机槽位来处理。具体表现为三个 bug:(1) `state_indices_tensor_d` 尺寸少了 spec 槽位;(2) CUDA graph capture 时 `block_idx_last_*` 持久缓冲按 `num_decode_tokens` 而非 `num_reqs` padding,与 kernel 索引方式不符;(3) 接受的 draft 少于调度数时,`(num_computed - 1) // block_size` 不再对应上一步实际写入的槽位(因为 num_computed 只反映被接受的 token),下一步读锚点错误。

## 修复方案
1. `vllm/v1/attention/backends/mamba_attn.py`:`state_indices_tensor_d` 尺寸改为 `cdiv(max_model_len, block_size) + num_speculative_blocks`;新增 `block_idx_last_scheduled_token_prev_step` 缓冲,`_compute_common_metadata` 中以 `prev_last_scheduled_idx`(无效时 fallback 到 `(num_computed_tokens - 1) // mamba_block_size` 并 clamp)计算上一步写入槽位;CUDA graph capture 时 `block_idx_last_*` 缓冲 padding 到 `padded_bs` 并把多余槽位清零。
2. `vllm/v1/kv_cache_interface.py`:`MambaSpec.max_memory_usage_bytes` 按新尺寸加 `num_speculative_blocks * page_size_bytes`;`utils.py` 的 `mamba_get_block_table_tensor` docstring 同步。
3. `vllm/model_executor/layers/mamba/mamba_mixer2.py`:spec 分支下 input/output indices 用 `block_idx_last_scheduled_token_prev_step_d` / `block_idx_last_scheduled_token_d` 加 `_decode_state_offsets`(arange(1+num_spec)) gather 整段状态。
4. `vllm/v1/worker/gpu_model_runner.py` + `mamba_utils.py`:新增 `mamba_prev_last_scheduled_idx` CpuGpuBuffer 与 `preprocess_mamba_all_specdec`(把每请求上一步最后调度块索引填入缓冲),`postprocess_mamba` 按 cache mode 分派,`all`+spec 分支在 `postprocess_mamba` 中记录每请求 `(seq_len-1)//block_size`。
5. `vllm/model_executor/models/config.py`:回退 #40454 的临时规避——不再在 prefix caching + spec decode 时强制默认 `align`。
新增回归测试 `tests/v1/attention/test_mamba_update_block_table.py`(2 个用例)。
注:pack 中 diff 在 40000 字符处截断(截断点位于 mamba_utils.py 之后),按文件粒度可能还有未展示的文件改动(如新增测试文件内容),以原 PR 为准。

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围(问题由 #39809 报告、#40454 曾以强制 align 规避)。
- 回归面:mamba cache `all` 模式 + 投机解码的整条路径(块表尺寸、显存估算增加 num_speculative_blocks 个 page、CUDA graph capture padding、spec 分支的 gather 逻辑);不再强制 `align` 后,hybrid Mamba2 + PC + spec decode 默认回到 `all`;`align` 模式行为仅被重构为分派但逻辑保持(按 diff 如实)。显存占用在 `all`+spec 下会略增。

## 关联
- Issue #39809(三个 bug 的报告)
- PR #40454(本 PR 回退的临时规避)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
