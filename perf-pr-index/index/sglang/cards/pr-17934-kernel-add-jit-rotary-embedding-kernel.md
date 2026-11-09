# PR #17934: [Kernel] Add JIT rotary_embedding_kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @pansicheng |
| 合并 | 2026-02-13 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | run-ci |
| Diff | diffstat 缺失(素材未提供统计);diff 实际涉及 5 个文件(3 新增 2 修改) |
| 链接 | https://github.com/sgl-project/sglang/pull/17934 |

## 一句话结论

为 rotary_embedding 新增 JIT 编译 CUDA kernel,H20 上较 sgl_kernel AOT 版快 1.07x–1.16x。

## 动机

PR 正文对动机只有一句:"Add JIT-compiled CUDA kernels for rotary_embedding"。此前该算子走 AOT 编译的 `sgl_kernel.rotary_embedding`,本 PR 将其纳入 sglang 正在建设的 `jit_kernel` 体系,实现与 vLLM `_custom_ops` 完全相同的逻辑(diff 注释明言"same implementation"),以 JIT 按需编译替代预编译库路径。

## 优化点

新内核移植自 vLLM 的 `pos_encoding_kernels.cu`:每个 thread block 负责一个 token,按 `scalar_t × IS_NEOX` 模板实例化,支持 float32/float16/bfloat16 三种 dtype 与 GPT-NeoX/GPT-J 两种旋转风格。加载 cos/sin 时统一走 `SGLANG_LDG`(非 ROCm 上为 `__ldg`,只读数据缓存路径);ROCm 上退化为普通解引用。Python 侧通过 `load_jit` 按需编译 `elementwise/pos_enc.cuh`,并拆成 `rotary_embedding_with_key` / `rotary_embedding_without_key` 两个注册的 torch custom op(分别声明 mutates_args)。最后把 `rotary_embedding.py` 中 CUDA/HIP 路径的 import 从 `sgl_kernel` 切换到 JIT 版本,并留 TODO 待多设备验证。

## 关键改动

```text
python/sglang/jit_kernel/csrc/elementwise/pos_enc.cuh: 新增 JIT rotary kernel(host 分发 + device 实现,313 行)
python/sglang/jit_kernel/pos_enc.py: load_jit 封装 + 注册 custom op(with_key / without_key)
python/sglang/jit_kernel/include/sgl_kernel/utils.cuh: 定义 SGLANG_LDG 宏(CUDA 用 __ldg,ROCm 直读)
python/sglang/srt/layers/rotary_embedding.py: CUDA/HIP 路径改 import sglang.jit_kernel.pos_enc
python/sglang/jit_kernel/tests/test_pos_enc.py: 正确性 + 性能测试(48 用例)
```

```cuda
// 每个 thread block 负责一个 token;cos/sin 走只读缓存
__global__ void rotary_embedding_kernel(...) {
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
  apply_rotary_embedding<scalar_t, IS_NEOX>(query, key, cache_ptr, ...);
}
```

## 性能声明

H20,与 `torch.ops.sgl_kernel.rotary_embedding` 对比(各规模 100 次迭代计时):

| 规模 | JIT | SGL | 加速 |
| --- | --- | --- | --- |
| Batch=1, Tokens=1 | 0.0153ms | 0.0168ms | 1.10x |
| Batch=4, Tokens=64 | 0.0212ms | 0.0240ms | 1.13x |
| Batch=8, Tokens=512 | 0.0209ms | 0.0243ms | 1.16x |
| Batch=16, Tokens=2048 | 0.0216ms | 0.0243ms | 1.13x |
| Batch=32, Tokens=16384 | 0.0786ms | 0.0846ms | 1.08x |
| Batch=64, Tokens=65536 | 0.3006ms | 0.3214ms | 1.07x |

均为 kernel 级微基准,PR 未注明 dtype 与模型配置。

## 验证与回归风险

新增 `test_pos_enc.py` 共 48 个用例全部通过:正确性对 vLLM torch native 实现 `torch.testing.assert_close`(bf16/fp16 atol=rtol=1e-2,fp32 1e-5),并覆盖 key 为 None、多 dtype、多 head 配置;性能测试同时附带精度校验。回归风险:`rotary_embedding.py` 把 CUDA 与 HIP 全部切到 JIT 路径(代码注释留有 "Test on different devices" 的 TODO),若某设备 JIT 编译失败或行为差异会影响所有走该层模型的 rotary 计算。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
