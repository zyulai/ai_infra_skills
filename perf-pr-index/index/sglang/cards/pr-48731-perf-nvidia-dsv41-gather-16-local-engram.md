# PR #48731: [Perf] [NVIDIA] dsv41: Gather 16 local Engram rows per program on SM103 and SM107

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @dmtri35 |
| 合并 | 2026-11-10 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | jit-kernel |
| Diff | 素材未提供 diffstat(diff 摘录完整:2 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/48731 |

## 一句话结论

engram_gather 本地 kernel 每 program 批 16 行(SM103/SM107),32K tokens/rank 时比 main 快 6.6x,端到端 prefill 约 +1% tokens/s。

## 动机

`_engram_gather_kernel` 此前对每个被查的行跑一个 Triton program,每个 program 是一条依赖加载链(先 id,再 payload 与 scales),最后 512-byte store。DeepSeek-V4.1 DP prefill 一次 forward 会做两次这样的调用、共 `ranks x tokens x 24` 行:4 ranks、每 rank 32K tokens 时达 3.1M programs——瓶颈是 per-program latency 而非带宽。4x VR200、32K tokens/rank 下每次调用 1.67 ms,约占 forward 的 1.5%。peer-read kernel(#46180)已经 8 行/program;本 PR 对本地 gather 采用同样的批行思路,16 行/program。main 上本地 gather 服务于非 peer-read 的 lookup:all-reduce、reduce-scatter、all-to-all lookup、host-table placement 与跨节点 TP groups。

## 优化点

`_engram_gather_kernel` 每 program 处理 ROWS 行:行号组成 2-D `[ROWS, DIM]` tile,经既有的 `_dequant_rows` 解码,最后一个 program 以 mask 处理 ragged 尾部;MXFP4 decode、目标行排列(`sort_ids`)与 shard 边界逻辑不变,输出与逐行版本 bitwise 相同。行数由新的 `_gather_rows_per_program()` 决定:SM103 与 SM107 取 16(实测过的两块卡),其他 GPU 保持 1 行——grid 与每 program 的工作量在那些卡上完全不变。收益来源是同一 program 内各行(id、payload、scales)的依赖加载相互重叠,摊薄了每行的延迟链。

## 关键改动

```text
python/sglang/kernels/ops/embeddings/engram_gather.py: kernel 改为 [ROWS, DIM] tile + ragged mask;新增 _gather_rows_per_program() 按 SM103/SM107 门控
test/registered/kernels/ops/embeddings/test_engram_gather.py: 新增 2/8/16 行 per program 与 1 行的 bitwise 参数化测试
```

```python
def _gather_rows_per_program() -> int:
    """Rows per program of the local gather: 16 on SM103 and SM107, where one row
    per program is bound by its chain of dependent loads; 1 on GPUs not measured."""
    cuda = current_arch().cuda
    return 16 if cuda is not None and cuda.capability in ((10, 3), (10, 7)) else 1
```

## 性能声明

VR200(SM107)单卡,CUDA graph of 20 calls,fp8 表 DIM 256 block 32,ids 均匀分布(约四分之一 owned),us/call:32K tokens/rank(3,145,728 行)main 1508.9 → 本 PR 228.9,即 6.6x,比 8 行/program(311.7)再快 27%;8,192 tokens 378.6→59.0;decode 尺寸(1-8 tokens)相对 main 最多贵 0.2 us/call。GB300(SM103,median of 15 graph replays):32K tokens/rank 1584.6→378.2,4.2x,比 8 行快 3%;1.5K 行以下最多贵 0.4 us/call(每 forward 两次调用)。端到端(较早构建):4x VR200,DeepSeek-V4.1-Flash,DEP4 prefill + engram all-reduce lookup,126,976 cached + 4,096 new tokens/request,4K-64K tokens/rank 约 +1% tokens/s/GPU。

## 验证与回归风险

改动为 bitwise,无需模型级精度跑分。VR200:8/16 行与 main 的 1 行 kernel 逐 bit 相等(bf16 按 int16 比较),覆盖 fp8 与 packed MXFP4 表、sorted/unsorted ids、零 e8m0 指数、1 至 196,613 行(4M 行表的 1M 行 shard),0 mismatch;新单测 test_engram_gather_rows_per_program_bit_exact(2/8/16 行,1-4099 行,含 shard 外 id);`TRITON_INTERPRET=1` CPU 复核逐 bit 相等;GB300 每个尺寸 bitwise 校验。风险:行数提升仅门控在 SM103/SM107(未测的卡保持旧行为;作者判断 B200/Hopper 大概率同样受益,可凭数据放宽);默认单节点 DEP 部署的 lookup 走 peer_read,本 PR 不改变该路径。

## 关联

#46180(前序:peer-read kernel 8 行/program,本 PR 将同一批行思路应用到本地 gather)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
