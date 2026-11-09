# PR #35735: [kernel] Split the custom all-reduce communicator into push/pull planes

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @DarkSharpness |
| 合并 | 2026-08-26 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci, jit-kernel, bypass-fastfail, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数;diff 另有 27 个文件因超长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/35735 |

## 一句话结论

重构 JIT custom all-reduce 的通信对象为 push/pull 两个 plane,主路径 1shot_push 内核快 2.7%–6.2%,SASS 指令数 512→344。

## 动机

`csrc/distributed` 与 `csrc/kimi_k3/comm` 的每个 JIT collective 都接收一个扁平 `CommunicatorObj`,携带所有 buffer:2×world_size 个 push slot、pull workspace、per-block semaphore、rank-local counter 与三个 multicast 地址之一。后果:另两个 multicast 地址在对象外传递(`ws_mc_base` 作为 int64 参数在 5 个 call site 重复,pull_sem_mc_ptr 走 3 个模块重复的 Python dict);push-only 使用方(minimax_m2 的 fused qk-norm)被迫分配占位 1 KB pull buffer 与 semaphore;只有 `custom_all_reduce.cuh` 读取 pull 字段,一半是死重;且它把 helper 定义在 `namespace sglang` 作用域并被 3 个 K3 文件跨目录 include,重命名会静默破坏下游内核。

## 优化点

按生命周期把对象拆为三层:`PushPlane`(zero-fill 对称 push slot、rank-local phase counter、multicast base)、`PullPlane`(对称 pull buffer + per-block semaphore,含两个 multicast 地址,任一半可为 0 元素张量)、`Communicator`(两个可选 plane + pull launch 宽度,即所有内核接收的对象)。固定 multicast 地址移入 plane,删除 5 个入口的 `ws_mc_base` 参数与 3 个模块的 `_PULL_SEM_MC_MAP`;per-call multicast 地址仍是 `intptr_t`。共享 PTX 词汇移到 `include/sgl_kernel/distributed/ptx.cuh`(删除三份 multimem flag increment、三份 semaphore multicast alias、`ptx_sys.cuh`),五个通信内核不再含 inline PTX;`Barrier`/`McBarrier`/`LamportTrait` 成为每个协议一份实现。`tp_qknorm` 的 8B volatile 对改为 `.sys` scope relaxed 访问。1shot_push 提速来自 `LamportTrait` 的 4-byte-atom marker 测试替代逐 bf16-half 测试,SASS 从 512 减到 344 条指令。PR 还记录了关键地址形式经验:shard offset 必须折叠进基指针(32 位 typed 向量索引,保持 peer table 在 uniform datapath),否则每线程多花约 12 个寄存器、损失约 10% 吞吐。

## 关键改动

```text
python/sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh: PushPlane/PullPlane/Communicator 定义(按素材描述;diff 收录部分为旧 communicator.cuh 删除)
python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh: 拆分 AllReducePushParams/AllReducePullParams 与 1shot_push/1shot_pull/2shot_pull 内核
python/sglang/kernels/jit/include/sgl_kernel/distributed/ptx.cuh: 共享 inline PTX 词汇(素材描述)
```

## 性能声明

8×B200(SM100),内核时间(CUDA graph 重放 20 次链式调用,排除 host dispatch,三轮交错,组内散布 <1%):`1shot_push`/eager(≤768 KB)**−2.7%…−6.2%**;`2shot_pull`/multicast −0.03%…+0.39%;`1shot_pull` eager/multicast −1.1%…+0.6%;`2shot_pull`/eager +0.5%…+0.9%。33 点中位数 +0.32%、最差 +0.92%;在调优后 SM100/world-8 配置实际派发的路径上中位数 **−3.30%**、最差 +0.39%。寄存器:所有内核零 spill;`1shot_push` bf16 60→63 寄存器但更快;`2shot_pull` eager bf16 53→51。正确性:291 个测试全部通过(含 8 GPU 的 custom_all_reduce 180、ar_fusion 58、tp_qknorm 42 等)。

## 验证与回归风险

291 个测试(7 个测试文件)+ 全部 11 个受影响 JIT 模块构建 + SM120 nvcc 直接编译验证 + `cuobjdump -res-usage` 寄存器配对审查;clangd 零诊断(除两个既有 cutlass include-path gap)。风险面:重构触及所有 JIT collective 的参数布局,PR 记录了两类会静默损失 ~10% 的地址形式陷阱(已在代码注释标注);`2shot_pull` eager 在早期修订曾回退 7%–18%,经 bisect 定位为地址形式而非抽象层;行为语义(pull 家族借 plane 仅做 barrier、K3 原地 reduce)有注释但依赖调用方正确传 plane。

## 关联

无(素材未列出前序/后续 PR 编号;涉及 minimax_m2 fused qk-norm push-only 使用方与 K3 pull 内核等区域)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
