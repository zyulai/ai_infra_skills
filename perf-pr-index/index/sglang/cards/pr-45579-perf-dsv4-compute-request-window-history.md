# PR #45579: [Perf] dsv4: Compute request-window history rows once per batch

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hnyls2002 |
| 合并 | 2026-10-21 |
| 分类 | Other (tier: major) |
| Label | deepseek, run-ci |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/45579 |

## 一句话结论

把 DeepSeek V4 request-window 的 history 源行索引从每层重算改为每 batch 算一次,削掉调度线程上每层 1.7-2.9% 的 `_history_src` 开销。

## 动机

DeepSeek V4.1 的 encoder SWA bounded replay 路径中,`RequestWindow.buffer` 在每一层都要通过 `_history_src` 重算每个 history slot 的源行(`row_loc`:按 slot 所属 request 与位置定位,无效 slot 取 zero 行)。同一 batch 内所有层读取的行完全相同,重复计算纯属浪费;profiling 显示 `row_loc` 与 `_history_src` 在调度线程上占比可观(每层 4.1-4.8% 与最高 2.9%)。

## 优化点

把源行计算前移到 metadata 构建阶段:`window_layout()` 新增必选参数 `zero_row`,在构建 `WindowLayout` 时用 `torch.where(history_valid, history_req * capacity + history_pos % capacity, zero_row)` 一次性算出 `history_src` 并存进结构体;`RequestWindow.buffer(layer)` 直接读 `layout.history_src`,删除每层重算的 `_history_src`。`WindowLayout.copy_` 同步复制新字段,保证 graph replay 时原地刷新的 captured layout 携带当前 batch 的行。两个 `window_layout` 调用方(CUDA 与 HIP 的 DeepSeek V4 backend)及相关测试改为传入 `zero_row`。行为无变化,仅作用于启用 encoder SWA bounded replay 的场景。

## 关键改动

```text
python/sglang/srt/mem_cache/dsv41_request_window.py: WindowLayout 新增 history_src 字段,copy_ 复制;window_layout 一次性计算;删除 _history_src
python/sglang/srt/layers/attention/deepseek_v4_backend.py: 调用方传入 zero_row
python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py: 同上
test/.../test_dsv41_request_window_copy.py 等四个测试: 适配新签名
```

```python
history_src = torch.where(
    history_valid, history_req * capacity + history_pos % capacity, zero_row
)
```

## 性能声明

PR 正文给出的是剖析占比与一次合并测量(DeepSeek V4.1 Flash,GB300,DEP4):4K tokens/rank 时 `row_loc` 与 `_history_src` 分别占调度线程 4.1-4.8% 与最高 2.9%;640 并发会话的 multi-turn prefill workload 中分别为 3.2% 与 1.7%(均为 `get_swa_raw_buffer` 下每层占比)。本改动与 #45577(V4.1 index-K slot snapshot)合并后,4K step 从 106-110 ms 降到 98.7 ms(每臂单次运行);本改动单独的收益未测量。

## 验证与回归风险

既有测试适配新签名并扩展:`test_dsv41_request_window_copy.py`、`test_q8kv8_sparse_prefill_backend.py`、`test_dsv4_compressed_pools.py`、`test_encoder_swa_replay.py`(引入 fake window 的 `_ZERO_ROW` 常量)。PR 声明无行为变化,风险面小:影响范围限于 DeepSeek V4 CUDA/HIP 后端的 request-window SWA replay 路径;`window_layout` 的 `zero_row` 变为必选参数,任何新增调用方都必须显式提供;CI 状态显示 PR Test (Base) 与 AMD ROCm 10 为 no_entry_sign、Extra 未通过。

## 关联

#45577(V4.1 index-K slot snapshot,与本改动合并测量);同区域为 DeepSeek V4 request-window/SWA 相关优化,正文未列其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
