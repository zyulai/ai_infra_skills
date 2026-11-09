# PR #48174: [Perf] [NVIDIA] moe: Re-land the JIT moe_finalize_fuse_shared with the shared row read after the PDL wait

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @DarkSharpness |
| 合并 | 2026-11-06 |
| 分类 | MoE(tier: major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(素材未提供增删行统计,diff 本身可用;含 1 个文件因超长被省略) |
| 链接 | https://github.com/sgl-project/sglang/pull/48174 |

## 一句话结论

重落地 JIT fused MoE finalize 内核,把 shared expert 行读取移到 PDL wait 之后,修复 CUDA graph 下 20/20 全错的竞态并保住内核收益。

## 动机

#47784 引入的 JIT `moe_finalize_fuse_shared` 在 #47994 被 revert,revert 怀疑的根因正是过早加载 `shared_output`:`shared_output` 由 shared expert 在 `alt_stream` 上写出、经 `wait_stream` 汇合后再进 finalize;在 CUDA graph 中,这个跨 stream 汇合进入 PDL launch 也会成为 programmatic edge,若 shared expert 提前触发依赖,finalize 可能在该行写出前启动——只有 PDL wait 能保证顺序。在 B300 上用"入口即触发、自旋 300us 再写"的 stand-in writer 复现:#47784 在 CUDA graph + PDL writer 下 20/20 全错,本 PR 0/20。

## 优化点

相对 #47784 的唯一修复:shared-expert 行改在 `PDLWaitPrimary` 之后读取;routing 张量在 `routing_ready=True` 时仍可在 wait 前加载(调用方保证生产内核在自己 griddepcontrol.wait 之后才触发依赖)。同时删掉 #47784 加的 device guard。内核重写为 header-only(`.cuh` 替代 `.cu`):每个线程持有 kVec 个 hidden 元素、一次性发出全部 K 个 gather;route 按 lane 摊到 warp 上以保持 top-k decode 单波驻留;超过两波后用 persistent 变体跨 token 步进并预取下个 token 的 route。另按 Blackwell `fma.rn.f32x2` 增加 packed fp32x2 运算。

## 关键改动

```text
python/sglang/kernels/jit/csrc/moe/moe_finalize_fuse_shared.cuh: 新 header-only 内核,PDL wait 后读 shared 行,含 persistent 变体
python/sglang/kernels/jit/csrc/moe/moe_finalize_fuse_shared.cu: 删除(被 .cuh 取代)
python/sglang/kernels/ops/moe/moe_finalize_fuse_shared.py: JIT 按权重 dtype/top_k/hidden/has_shared/use_pdl 特化;新增 out 与 routing_ready 参数
python/sglang/kernels/jit/include/sgl_kernel/math.cuh: 新增 fma_f32x2/mul_f32x2 packed 运算
test/registered/kernels/ops/moe/test_moe_finalize_fuse_shared.py: 对 fp32 参考的全组合正确性测试
test/registered/kernels/benchmark/moe/bench_moe_finalize_fuse_shared.py: 新增基准(6 个模型配置)
```

关键片段(shared 行严格在 wait 之后):

```cuda
if (!p.route_before_wait) PDLWaitPrimary<kUsePDL>();
uint32_t token = blockIdx.y;
token_t cur;
cur.load_route(p, token, lane_id);
if (p.route_before_wait) PDLWaitPrimary<kUsePDL>();
cur.load_shared(p, token, safe_chunk);
```

## 性能声明

测试条件:B300,hidden 7168,top-k 8,bf16 输出/fp32 权重,CUDA graph 挂在真实 PDL producer 后(producer 在最终 store 前触发);finalize 开销 = pair 时间减 producer-only 时间,2×15 次 replay 取中位数。

- vs #47784:warm(shared 行刚写、在 L2)在 1-8192 tokens 内相差 ±0.04us 以内;cold(轮换 buffer)64/128 tokens 慢 +0.12/+0.23us,其余规模约 0;无 PDL 时约 0。
- vs pre-#47784 的 main 内核:warm 下 1 token -0.65us、32-128 tokens -1.5 至 -2.9us、4096-8192 tokens 持平;cold 下 32-128 tokens -2.6 至 -6.5us。
- 在 wait 后立即发 shared load 与按使用顺序(gather 之后)一样快或更快:persistent 内核最多快 0.6us。

## 验证与回归风险

`test_moe_finalize_fuse_shared.py` 于 B300:132 passed、6 skipped(与 #47784 一致);SASS 检查确认 shared 的 `LDG.E.128` 位于 grid-dependency wait 之后且无栈溢出。作者注明合并前需对 nightly `test_dsa_glm52_hisparse.py`(8-gpu-b200)执行 `/rerun-test`,正是该测试此前捕获了回归。竞态复现需要专门构造的早触发 writer 内核,故未注册为常规测试。风险路径:PDL/CUDA graph 下 shared expert 的跨流汇合语义;`routing_ready=False` 语义需生产者不提前触发。

## 关联

前序:#47784(首次引入,被 revert)、#47994(revert)。同分类:trtllm deferred finalize / MoE 输出后处理相关 PR;新基准覆盖 kimi_k3、deepseek_v3/v4、glm5、qwen3_5_moe 等 trtllm deferred-finalize 模型配置。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
