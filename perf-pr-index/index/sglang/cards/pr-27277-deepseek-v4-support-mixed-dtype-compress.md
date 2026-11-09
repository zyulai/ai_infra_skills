# PR #27277: Deepseek v4: support mixed dtype compression states

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @DaZhUUU |
| 合并 | 2026-06-17 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | deepseek, run-ci, jit-kernel, release-highlight |
| Diff | diffstat 缺失(pack 记录 +None/-None/None files);diff 可用(另有 2 个文件因超长未包含在 pack 中) |
| 链接 | https://github.com/sgl-project/sglang/pull/27277 |

## 一句话结论

DeepSeek-V4 C4/C128 压缩状态支持 BF16 存储(env 开关,默认仍 FP32),削减 sparse attention 压缩状态的带宽与存储开销,大 batch C128 decode 单核最高 1.365x。

## 动机

DeepSeek-V4 sparse attention 路径中,压缩状态(compress state)缓冲此前固定 FP32 存储与更新,其内存带宽与存储开销在该路径占比较大。本 PR 引入混合 dtype 支持:`SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16` 时 C4/C128 压缩状态缓冲可用 BF16 分配与更新;不设该环境变量时保持原 FP32 默认路径不变。目标是在状态缓冲访存主导的场景(尤其大 batch C128 decode)降低开销。

## 优化点

1. **内核模板拆分**:C4/C128 JIT kernel 的模板参数从 `<InFloat, OutFloat>` 改为 `<BufferFloat, InputFloat, OutFloat>`,把"状态缓冲存储 dtype"与"输入 dtype"解耦;混合 dtype 分支从 BF16 缓冲 load 后逐元素 `cast<InputFloat>` 到 FP32 寄存器再做 online softmax(softmax 计算仍为 FP32,精度仅损失在存储侧)。`std::is_same_v<BufferFloat, InputFloat>` 的 constexpr 分支保证同 dtype 时走原路径、零开销。
2. **写路径适配**:`c4_write_decode`/`c128_write_decode`/`write_*_prefill` 增加 BF16 store 分支;write kernel 中 `PDLTriggerSecondary` 移到 cast 完成之后再触发,保持 PDL 依赖正确。
3. **运行时配置**:`model_runner_kv_cache_mixin` 的 `_get_dsv4_compress_state_dtypes()` 解析环境变量(fp32/bf16),`SGLANG_OPT_USE_ONLINE_COMPRESS=1` 时拒绝 bf16(online c128 state 必须 FP32)并报错;`state_dtype` 拆为 `c4_state_dtype`/`c128_state_dtype`,memory pool 按 ratio 选 dtype 分配状态页。
4. **容量计算更新**:`pool_configurator` 的 bytes-per-full-token 计算改用各 state 的实际 dtype 大小,使 KV 容量估算与 BF16 存储一致。
5. **JIT 缓存键**:`_jit_compress_module` 增加 `dtype_buffer` 参数,不同 buffer dtype 编译不同模块实例。

## 关键改动

```text
python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh: c4_forward/write_decode/
  write_prefill 增加 BufferFloat/InputFloat 模板与混合 dtype 分支
python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh: 同上,decode/prefill
  kernel 与 TensorMatcher dtype 校验改为 buffer/input 分离
python/sglang/jit_kernel/dsv4/compress.py: _jit_compress_module 增加
  dtype_buffer 参数;compress_forward 传入 kv_score_buffer.dtype
python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py: state_dtype 拆分为
  c4/c128 两份,按 ratio 分配
python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py +
  pool_configurator.py: 环境变量解析(含 online compress 互斥检查)与
  容量字节数计算
python/sglang/srt/environ.py: 新增 SGLANG_DSV4_COMPRESS_STATE_DTYPE(默认
  "float32")
```

关键片段(diff 摘录,混合 dtype 载入分支):

```cpp
} else {  // mixed dtype
    const auto kv_tmp = gmem_buffer.load(src);
    const auto score_tmp = gmem_buffer.load(src + Trait::kScoreOffset);
    for (int32_t k = 0; k < kTileElements; ++k) {
        kv[i][k] = cast<InputFloat>(kv_tmp[k]);
        score[i][k] = cast<InputFloat>(score_tmp[k]);
    }
}
```

## 性能声明

PR 正文,未独立核实。单核 operator benchmark(FP32 us vs BF16 us):

- **136 个 preset shape 汇总**:speedup >1.02x 有 41 行,>1.05x 有 15 行;0.98–1.02x 有 77 行;<0.98x 有 18 行。
- **最佳案例**:Flash ratio=128 decode bs=128(state 128×128×1024):23.27→17.05µs(**1.365x**);同 shape bs=64 为 1.086x;Pro ratio=128 prefill 1.046x;Flash ratio=4 decode bs=2 1.069x。
- **服务 trace 运行时 shape**:多为 1.024–1.050x 的小幅收益;部分 shape 退化(0.989x / 0.953x / 0.945x)。

作者结论:收益在状态缓冲访存主导时最明显(大 batch C128 decode),部分 prefill/runtime replay shape 只有较小收益或回退;FP32 仍是默认路径。

## 验证与回归风险

- 精度评测(bf16 开启,服务配置 DeepSeek-V4-Flash,tp8/dp8 + dp-attention、marlin MoE runner、EAGLE 3-step 投机):GSM8K(1314 有效例)0.969559;MMLU(14042 请求)0.886;GPQA(32×8,thinking mode)均值 0.91796875;LongBench(128 例)0.609375。PR 未附 FP32 基线的同套评测数字。
- Checklist 各项已勾选(含 unit tests / docs);CI 状态(pack 记录):Base 与 Extra 均失败。
- 回归面:混合 dtype 分支覆盖 decode/prefill/write 三类 kernel 的 overlap 与非 overlap 路径,边界条件(buffer_len、overlap 页、invalid plan)较多;`SGLANG_OPT_USE_ONLINE_COMPRESS=1` 与 bf16 的互斥靠启动时 raise 兜底;容量估算变化影响 mem-fraction 分配。默认关闭(env 缺省 float32),不影响未开开关的用户。

## 关联

#24041(branch PR,PR 正文开头链接)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
