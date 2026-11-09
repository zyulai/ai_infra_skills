# PR #28925: fix(nightly-precision): pin flashinfer allreduce-fusion backend for TP-partial capture contract

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28925](https://github.com/sgl-project/sglang/pull/28925) |
| 作者 | JustinTong0323 |
| 合并日期 | 2026-07-02 |
| labels | (none) |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
#23402 改变 allreduce fusion 默认值致 nightly 精度测试捕获契约失效,在 nightly serve 参数中显式钉住 trtllm 后端修复。

## 问题现象
nightly 精度回归测试(`test_nightly_precision_regression`,suite `nightly-precision-8-gpu-h200`)自 2026-06-17 起在 `zai-org/GLM-5.1-FP8` 上持续失败:`FAILED - tensor=non_intrusive__model.layers.8.inputs.1 rel_diff=0.7537`;33 个捕获层中 30 层失败,`rel_diff` 全部落在 [0.7497, 0.7839],仅 layer 0 通过。

## 根因
PR 正文披露:二分定位到 #23402(`32685874f3`,"Reenable MNNVL backend for FlashInfer allreduce fusion")。该 PR 将 SM90 从 fusion 自动启用列表中移除(SM90 mnnvl 需要 H200 节点不具备的 NVLink multicast fabric),使 H200 上的门控从 `enable_flashinfer_allreduce_fusion`(GLM 系自动 True)变为 `flashinfer_allreduce_fusion_backend is not None`(H200 上为 None)。nightly serve 参数不传 fusion 标志,fusion 由开变关,翻转了 dumper 的捕获契约:fusion 开时上一 MLP 层跳过 `postprocess_layer`、标记 tensor 延迟 allreduce,dumper 在下一层入口捕获 TP-partial 和(各 rank 不同);fusion 关时立即 allreduce,各 rank 捕获到相同的全量和。比较器按 `--override-dims "...:bs h[tp:partial]"` 对 8 个相同全量和张量再求和,得到 8 倍幅值(abs_mean 0.00556 → 0.04443,恰为 8 倍),`rel_diff = 1 - 16/65 = 0.7538`,与观测的 0.7537 吻合。layer 0 通过是因为其 `inputs.1` 是 embedding 输出,先于任何 MLP allreduce。

## 修复方案
测试基建修复(非前向精度改动,模型 forward 不变,只改变 dumper 记录的是 partial 还是 full 张量):
- `test/registered/debug_utils/test_nightly_precision_regression.py`:新增常量 `PRECISION_FUSION_BACKEND = "trtllm"`,在 nightly serve 参数中显式加入 `--flashinfer-allreduce-fusion-backend trtllm`,恢复 TP-partial 捕获契约;显式 `trtllm` 而非 `auto`,因为 `auto` 在 SM90 单节点会解析为 `mnnvl`;
- `fusion_backend` 纳入 `_capture_signature` 捕获签名与 HF push 元数据,使未来 fusion 默认值再变化时重建基线而非静默错配 partial/full。

关键 diff 摘录:

```diff
+PRECISION_FUSION_BACKEND = "trtllm"
@@
         "--disable-cuda-graph",
         "--disable-piecewise-cuda-graph",
         "--disable-radix-cache",
+        "--flashinfer-allreduce-fusion-backend",
+        PRECISION_FUSION_BACKEND,
```

## 影响与回归风险
- 影响窗口:由正文可知,失败自 2026-06-17(#23402 合并后)开始至本 PR 修复前;仅影响 nightly 精度测试基线采集与比较。
- 回归面:仅 `test_nightly_precision_regression.py` 的 serve 启动参数、捕获签名与上报元数据;模型前向与生产路径零改动。验证:H200 8 卡 TP8 对 2026-06-16 pre-#23402 基线(run-2dd449c)33/33 通过、max_rel_diff=0.000000(bit-exact),证实恢复原捕获契约、既有基线可复用。

## 关联
- [#23402](https://github.com/sgl-project/sglang/pull/23402)(引入契约翻转的 PR)
- [#28190](https://github.com/sgl-project/sglang/pull/28190)(正文提及:修复 `_select_latest_run` 不把失败 run 提升为比较基线,独立但应同期合入)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
