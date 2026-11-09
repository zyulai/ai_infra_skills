# RFC #51005: [RFC]: Let Platforms customize fused MoE Router construction

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51005](https://github.com/vllm-project/vllm/issues/51005) |
| 作者 | freyfwt |
| 创建日期 | 2026-08-04 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | MoE |

## 一句话结论
在 fused MoE 工厂选中 Router 分支后加一个 Platform 构造器钩子,让 OOT 平台替换为兼容的设备特定实现而不动上游选择逻辑。

## 提案动机
上游 fused MoE factory 掌握选择 Router 的优先级规则,但 OOT 加速器有时需要所选分支之外的设备特定 Router 实现。缺少构造扩展点时,插件只能 patch 已构造 Router 的方法、patch factory,或复制其不断增长的挑选逻辑与构造参数。初版 RFC 提出的 qualified Backend 类、仅子类结果与按类缓存设计,已被本修订版取代:现在采用一个直接的 Platform 构造器钩子(draft #51006),并允许兼容的 sibling Router。

## 提出的方案
1. Platform 构造器契约:工厂分支选定具体 Router 类后调用 `Platform.get_fused_moe_router_constructor(selected_cls)`;默认实现返回 `selected_cls` 保持现有构造;OOT Platform 可返回任意 callable——接受该分支相同的构造参数、产出 `FusedMoERouter`,可以是子类、兼容的 sibling Router 或原类不变。钩子拿到的是选定的类而非 factory 原始路由配置;Core 验证结果可调用、以既有参数调用、检查实现 `FusedMoERouter` 基类契约,不要求与所选具体类有子类关系。
2. 选择与生命周期顺序固定:上游先按既有优先级规则选分支(含 CPU 分支)→ 向 Platform 请求构造器 → 传入分支原始参数 → serving 前验证返回的 Router。模型初始化中每个 Router 构造可各触发一次钩子;无 Backend 发现、无按类缓存;Router forward 执行只用构造对象的普通方法,不再有 Platform 查找。构造器必须保持所选分支的外部相关路由语义、可选状态与接受的构造参数;Core 能验证基类型,数值等价义务在 OOT 集成测试。
3. 与 EPLB 解耦:钩子不接收 EPLB policy、load window、communicator、mapping 或 transfer 对象;若上游 factory 已传可选 `eplb_state`,它作为构造参数原样保留;EPLB 状态与 per-layer 设备操作由 #49702 和 draft #49700 独立提供。
- 文中还列出 5 个被拒替代方案(替换实例方法、让 Platform 提供完整 factory、Backend 类+缓存、仅允许子类、全局/forward 时回调)及对应风险缓解:构造期校验非 callable/非 Router、要求 OOT 路由与模型输出等价测试、钩子应用于每个最终选择分支、不在热路径加注册表或查找。

## 优化点与收益
正文未给出量化数字。收益为扩展性:OOT 平台(动机案例为 Ascend,目前需在 vllm-project/vllm-ascend#12804 中适配已构造的 Router 实例)可改用此构造器钩子替代对私有实例方法的 patch,同时完整保留上游工厂的 Router 选择优先级与构造参数。

## 现状与落地
仍在讨论/等待评审(open)。修订版反馈期为两周、至 2026-09-29;已有 draft 实现 #51006。验收要求包含:Fake Platforms 演示默认构造、兼容 sibling 与未定制分支不变;契约测试拒绝非法构造器;内置 Router 套件保持数值行为;以及一个下游 OOT 集成在真实加速器硬件上证明等价路由与模型输出后才算就绪。

## 关联
- #51006(draft 实现)
- #49702(EPLB device-operation RFC,与本 RFC 分离)
- #49700(EPLB 状态与设备操作钩子 draft)
- [vllm-project/vllm-ascend#12804](https://github.com/vllm-project/vllm-ascend/pull/12804)(动机:Ascend 当前的实例方法适配)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
