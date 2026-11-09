# PR #51011: [ROCm][MLA] [K3] Fix fp8 KV cache decode on the AITER MLA backend

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51011](https://github.com/vllm-project/vllm/pull/51011) |
| 作者 | fanxingran |
| 合并日期 | 2026-08-10 |
| labels | rocm, ready, verified, k3 |
| 热度 | 👍 2 · 💬 25 |
| 分类 | Quantization |
| bug 类型 | general |

## 一句话结论

Kimi-K3 TP8(每 rank 12 头)+ fp8 KV cache 在 ROCm AITER MLA 后端下断言/KeyError/静默出错;重写 decode 路由五处修复,GSM8K 74.00%→97.19%。

## 问题现象

Kimi-K3 在 TP8 下每 rank 只有 12 个 MLA head,配 `--kv-cache-dtype fp8` 用 ROCm AITER MLA 后端(MI355X, gfx950)无法正确服务。未修改的 main 上跑满 1319 题 GSM8K 只有 74.00%,其中 285 个答案退化;打上本 PR 后 97.19%、0 退化。正文披露后端按 head 数和 query 长度有三种失败方式:1) head 数能整除 16 时,小 head 解码被路由到 Gluon 的唯一 fp8-KV regime `bh16bn128`,它断言 `batch_size == 1`,第二个请求一到即把服务打挂(batch 1 的冒烟测试测不出来);2) 投机解码 query 长度下抛 `KeyError`——白名单外的 drafter 拿到 qlen=1 的元数据,而路由放行到 `1 + 2*num_spec`,fp8 的 `get_block_n_fp8[num_heads*qlen]` 表只含 `{8,16,24,32,48,64,128,256,384,512}`,16 头时 qlen 5..7、9..15 均为 KeyError,运行中途随机崩溃;3) head 数不整除 16 时静默输出错误(Kimi-K3 即此例):21.6% 的请求先输出一两个貌似合理的 token 然后复读单 token 到长度上限(如 `We!!!!!!...`),是 NaN 进入 sampler 的特征;qlen==1 时不报错,易被误读为量化损失。

## 根因

正文明确披露三条根因链:Gluon 的 fp8 唯一 regime 断言 batch 1 且 head 数 <16 一律被路由进 Gluon(与 dtype 无关);`_mtp_decode_qlen` 来自硬编码方法名白名单 `("mtp", "deepseek_mtp")`,其他 drafter 的元数据按 qlen=1 分配;以及 `use_persistent_metadata` 以 `num_heads >= 16` 为门,12 头 rank 补 padding 到 16 跑 asm decode 却拿不到持久化调度,bf16 尚能容忍、fp8 不行。

## 修复方案

五处改动全部在 decode 路由,不动 kernel/aiter:1) `_mtp_decode_qlen` 改由 `reorder_batch_threshold`(decode 可收到的最大 qlen)推导;2) 量化 KV cache 一律不路由到 Gluon decode;3) verify 分支抽成 `use_gluon_verify` 谓词并排除 fp8;4) 持久化元数据门改为跟随路由(`not use_gluon_decode and not use_gluon_verify`)而非 head 数;5) `_kv_cache_dtype_str` 保留在 builder 上供 `_build_decode` 决策。关键 diff:

```diff
+        # Gluon's only fp8-KV regime, bh16bn128, is a bf16-query kernel with a
+        # hardcoded scale that asserts batch_size == 1, so it cannot serve a
+        # decode batch. Checked before the mode knob: an explicit "gluon"
+        # request under fp8 would assert immediately.
+        if is_quantized_kv_cache(kv_cache_dtype):
+            return False
```

fp8 能离开 Gluon 依赖 q-row fold:补齐的 16 heads x qlen 8 折叠到 `nhead=32, qseqlen=4` kernel(包内 `.co`)。单元测试 425 passed;新文件 `test_rocm_aiter_mla_fp8_decode_routing.py` 钉死各谓词。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:按 diff,只影响 ROCm AITER MLA 后端的路由判定:`use_gluon_decode`/新增 `use_gluon_verify` 签名变化并加 dtype 参数;bf16 除数 head 在 gfx950 仍走 Gluon(不损失性能),bf16 非 4 以内 qlen 的 padded rank 在无 gfx950 fold 时回落非持久化入口;`VLLM_ROCM_AITER_MLA_ASM_PADDING` 旋钮不能把 fp8 强推进 Gluon(作者论证其不能替代本修复)。作者注明改动 4 对 bf16 无精度收益(非持久化 bf16 约 1% 更快),仅由 fp8 正确性辩护。生产配置(TP8、fp8、FULL_AND_PIECEWISE)下引擎正常捕获 35 piecewise + 19 full graphs 并正常服务。

## 关联

- https://github.com/vllm-project/vllm/pull/50578 (VLLM_ROCM_AITER_MLA_ASM_PADDING 旋钮,非替代方案)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
