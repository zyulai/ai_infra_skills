# PR #52228: [Model Runner V2][Spec Decode] Adaptive Verification without a confidence head

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @TheEpicDolphin |
| 合并 | 2026-09-14 |
| 分类 | Speculative Decoding (tier: major) |
| Label | speculative-decoding, ready, mrv2, dflash |
| Diff | diffstat 缺失(素材未提供 additions/deletions/文件数);diff 全文含约 11 个文件(1 个超长文件被省略),新增 acceptance_estimator.py 约 501 行 |
| 链接 | https://github.com/vllm-project/vllm/pull/52228 |

## 一句话结论

用在线接受率估计器替代 trained confidence head,把自适应验证(AV)扩展到所有 draft-model 投机解码,并发 64 下平均吞吐 +14.6-18.5%。

## 动机

此前 adaptive verification 仅对带 confidence head 的 DSpark 可用;MTP、EAGLE3、DFlash 1/2 等所有 draft-model 投机器都无法受益于"按接受概率裁剪验证预算"。高并发下 fixed-k 验证的算力浪费(与本库 #47808 同一问题)在这些投机器上无解。前置条件是基础模型的 attention backend 支持 varlen verification,由依赖 PR #52992 扩展(Inkling sconv/rel-attention、FLASH_ATTN、FLASH_ATTN_MLA)。

## 优化点

新增 `OnlineAcceptanceEstimator`:对每个 draft 位置拟合在线逻辑回归 `p_accept(k) = sigmoid(w * logit(q_max) + b[k])`,`w` 为全局共享斜率、`b[k]` 为逐位置截距,服务期间以 IRLS 持续重拟合(每 REFIT_INTERVAL=100 步解一次 Newton 步,共享斜率使信息矩阵为 arrowhead 结构、求解塌缩为标量方程;更新按样本数阻尼 `n/(n+50)`,L2=1e-3)。特征选 `logit(q_max)`(draft 分布最大概率的 log-odds):在约 4.5M 标注 draft 的留出集上 AUC 0.8336 / log-loss 0.4284 / ECE 0.0100,优于 entropy、top-2 margin 等 lossless 特征;刻意不用排名更高的 `logit(q_sampled)`(AUC 0.8553)——它对被裁剪的 token 本身打分,破坏拒绝采样的 losslessness,会使输出分布偏向 drafter 自信 token。预测在 `sample_draft` 内完成(draft logits 本来就要物化,MTP/EAGLE3/DFlash 零投机器改动;DFlash2 在等价的 selector walk 点打分),概率同时写入持久 request-state 槽(供 refit)与当前 batch 位置(供 AV 索引)。裁剪完全复用既有 DSpark AV 路径:`record_confidences` 异步 D2H、CPU 侧按落地副本定全局预算、GPU 侧按 survival probability(逐位置置信度 running product)做全局 top-k 录取,因 survival 单调递减,录取自动是每请求的前缀。配置上 `enable_adaptive_verification` 对所有 draft-model 投机器放开,仅与 `use_local_argmax_reduction` 互斥(后者不物化 logits);DSpark 有 confidence head 时优先用 head、关掉估计器。

## 关键改动

```text
vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py: OnlineAcceptanceEstimator + 4 个 Triton kernel(accumulate/refit/local_max_sumexp/predict)
vllm/v1/worker/gpu/spec_decode/speculator.py: sample_draft 接入 _maybe_predict_acceptance 与 observe_verification
vllm/v1/worker/gpu/model_runner.py: sample_tokens 中对 DraftModelSpeculator 调 observe_verification
vllm/v1/worker/gpu/spec_decode/dspark/speculator.py: confidence head 与估计器二选一(use_confidence_head)
vllm/config/speculative.py: AV 仅与 use_local_argmax_reduction 互斥校验
```

## 性能声明

条件:temp 1.0 / top-p 0.95、concurrency 64、≤2K in / 2K out、natural EOS,AV OFF vs ON 各两跑。4xGB200 TP4:MiMo-V2.5-Pro-FP4 + DFlash (nspec 8) 平均 +19.0% tok/s(speed-bench low-entropy +16.8%、high-entropy +34.0%);Inkling-NVFP4 + MTP8 平均 +17.6%(high-entropy +32.3%);1xGB200:Muse-Glimmer-30B + DFlash2 (nspec 7) 平均 +14.7%;TPOT 相应 -13.4% 至 -19.2%。Kimi-K2.5 + DFlash,8xH200 TP8:gsm8k +29.8%、humaneval +50.1%、mt-bench +53.5%(均值 +44.5%;gsm8k 受估计器 300-step warm-up 影响,pass 2 为 +46.0%)。DSpark 上估计器 vs trained head(均开 AV,TP4 GB200,nspec 5):tok/s +2.1%、median TPOT -3.9%、accepted/drafted +4.70pp——估计器与 head 基本持平且无需训练权重。精度:GSM8K 配对设计(同题同种子),四模型 pooled -0.08pp(5,276 对,McNemar p=0.769),无显著变化。

## 验证与回归风险

测试:新增 `tests/v1/spec_decode/test_acceptance_estimator.py`——predict 对 slot 序与 batch 序双写一致(用与 kernel 不同路径的 logsumexp 参考实现交叉验证)、cudagraph padding 行(slot -1)不污染 live 请求、refit 后系数更新等;既有 speculator 测试补 `use_acceptance_estimator=False`/`acceptance_estimator=None` 属性。风险面:AV 现在影响所有 draft-model 投机器的 batch 形状(裁剪后 varlen),依赖 #52992 的 backend 支持(AttentionCGSupport.ALWAYS 翻转仅涉及 Inkling 两个 builder);估计器系数初始为冷启动中位数(w=0.5, b=-0.2),极端负载下前几个 refit 周期预测欠佳(裁剪仍安全,退化为按深度均匀截断);Kimi 基线数据带 300-step warm-up(已移除);DFlash2 在该分支需上游 #53435 的修复才能加载。

## 关联

直接依赖:#52992(varlen verification 的 attention backend 支持,Inkling/FLASH_ATTN/FLASH_ATTN_MLA)。前序:#47808(DSpark confidence-head AV,本 PR 是其"无 head 泛化")。同区域:MTP/DFlash/DFlash2/EAGLE3 投机器、rejection sampler、`KIMI_BENCHMARK_RESULTS.md` 基线记录。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
