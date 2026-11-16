# ACT Residual PPO 训练约定

## 当前目标

- 算法固定为Residual PPO。初始 checkpoint 由启动命令明确指定；**round14 起用户指定从离线基座 epoch4000（outputs/act/RoboDojo-stack_blocks-arx_x5-ee-0/2026.11.14-06.48.15/policy_epoch_4000_seed_0.ckpt）全新训练**，round10 useful checkpoint 已废弃（软链接已按用户指示移除，权重文件保留）；此后逐轮按有效性判据链式推进；不得自动选择最新权重。
- 先训练 seed 0 布局集合的 **14–23**，在 `configs/act_online.yaml` 的 `rollout_layout_ids` 配置，按回合轮转覆盖。seed 0 实际有 55 个布局，编号 **0–54**。
- 课程阶段连续 3 批独立评测、每批 10 回合的最终得分全部至少 15 后，再把训练布局扩展至 0–54；扩展后的结果另行验证，不能由课程阶段推断泛化。
- **用户已允许 residual actor/critic 使用积木位姿。** 冻结 ACT 仍只接收 `cam_head` RGB 和 16 维本体状态；残差输入为归一化本体、基座动作、冻结视觉特征，以及按 `block_0/1/2` 排列的场景相对位置和 wxyz 四元数。训练和评测使用相同输入。
- 外部得分始终为 RoboDojo 的 `0/15/100`。15 分是部分完成，不是成功；成功仍要求官方 100 分条件。
- 历史对照（round7–13）：旧势能 `20*max_block_lift − 10*nearest_wrist_block_xy_distance` 加 score_delta，训练零任务分。round14 起改用下述阶段化势能，不回退旧式。
- **阶段化势能（round14 起，用户指定的 7 步分解）**：总目标布局 14–23 稳定 ≥15。"识别→靠近→抓起→放置第一块→再抓起第二块→叠到第一块上方→第三块叠到 2 倍高度"实现为**同一个纯物理状态势能 Φ**（无阶段状态机、无历史依赖）："识别"由残差输入的积木位姿承担，不设奖励项；官方无"指定区域"，放置用块间相对几何。四项：3D 抓取点接近（wrist+0.143·R(q)x̂，tanh，随链进度门控）；有界抬升（每块 15mm ramp/140mm plateau/180mm zero，两块门控求和，防单块高举刷分）；**有向上升链 pair 计分**（六排列取最大两条正 dz 边之和，禁止共享中间块重复计数；宽核对齐+高度窗只做运输代理）；实测释放（官方同款归一化的实测夹爪关节，0.8 即满开；仅在 ≤20mm 对齐、块线速度 ≤0.01m/s、角速度 ≤0.1rad/s 的"落定对"上生效）。落定+释放后才对 reach 施加 0.9m 已放置排除（大于实测最大塔→第三块距离 0.8084m，保证转移单调；携带中不排除）。速度落定是瞬时物理代理，不是接触/静置时长证明；稳定性的权威判定仍是官方评分与独立评测。PBRS 输入均为物理状态或 reset 常量（初块高），不含当前动作；terminated 强制 Φ=0（Grzes），截断保留 Φ+bootstrap（Pardo）。数学：折扣 shaping 全程伸缩为 γ^N·Φ_N−Φ_0（路径无关），闭合循环的未折扣伪影 (γ^N−1)Φ_0 上界 |Φ_0|≈8.3，远小于 15/85 分档。实现后单测通过，再交 **3 个无上下文 subagent 独立检查**，全部通过才投入训练；实现修正后须重审。
- 训练和评测都只在官方 `run_reward` 成功（`reward >= 1 - 1e-3`）或达到步数上限时结束回合；公开得分 0/15/100 本身不结束回合。逐回合数据中 `layout_id` 是布局编号，`seed` 是训练 RNG 配置种子（课程阶段为 0），两者独立。

## 单轮运行与评测

- 当前课程阶段目标是布局 14–23 全部达到 15 分以上。有效性判据（用户明确指定）：本轮训练后评测的 `_result.json` 顶层 `score` 均分高于上一轮评测均分即有效，不考虑运气或方差解释；若均分与上一轮相同，固定同一候选再评测一次，第二次仍相同（或更高）也算有效，更低则无效。round14 从 epoch4000 全新训练，起点均分按既有 epoch4000 课程评测取 1.5；round10 曾被记为第一个有效 checkpoint，现已按用户指示废弃。
- 每轮训练软预算**自 round15 起为八小时**（`max_train_seconds: 28800`；round14 是最后一个四小时轮，仍按 14400 自然停止，不改在途轮次）。启动、checkpoint 加载、最后的更新与保存及独立评测另计。更新数上限依据实测采样波次耗时安排，避免上限使训练明显提前结束（波次耗时随显存爬升，八小时约需 24–28 次更新上限，launch 前按本轮实测确认），不靠增加 PPO 重复遍数填满预算。到期仅在完整采样波次边界停止，非空批次必须完成 PPO 更新并保存。八小时连续训练大概率触发 Isaac 渲染器显存蠕变 OOM（约 3.5–4 小时一次，round11b/12 已实测）；波次边界崩溃不丢回合数据，由启动器 supervisor 重启或按崩溃恢复处理（属崩溃恢复，非预算强杀），重启耗时计入预算。
- 严禁因预算到点发送 SIGTERM/SIGKILL 或使用外部超时强杀。确认 `policy_last.ckpt` 原子保存、可加载、参数有限后，才进入评测。
- 每轮独立输出目录，日志不拼接；checkpoint 保存冻结基座、残差、优化器、CPU/CUDA/NumPy RNG、累计进度及布局轮转游标。独立 actor/critic 使用 v3 格式，不迁移旧共享网络 v1/v2 残差 checkpoint；原 ACT 离线 checkpoint 仍可作基座。
- 评测只使用 `bash scripts/eval_policy.sh` 和项目的 `configs/eval.yaml`，不编写其他评测器。课程阶段固定 `seed: 0`、`eval_num: 10`、`num_envs: 4`、`layout_ids: [14,15,16,17,18,19,20,21,22,23]`，残差执行均值动作。起点和候选必须用同一布局集合评测。
- 以脚本生成的 `_result.json` 确认 `eval_time=10` 和逐回合分数，不能凭退出码、视频文件名或 Fail 数推断得分。结果明细中的 `score` 为 0/0.15/1，顶层均分为 0–100。客户端会把布局子集重编号为 0..N-1；`eval_policy.sh` 在验证后按 `layout_map.json` 把 `_result.json` 的 `layout_id` 改写回真实布局编号（课程阶段即 14–23），并把评测视频文件名中的回合编号替换为真实布局编号（`episode_<真实布局号>_cam_*`）。汇报和记录一律使用真实编号。
- 每轮评测前必须把 `configs/eval.yaml` 的 `ckpt` 改为本轮训练结束产生的 `policy_last.ckpt`，严禁误用基座 epoch4000 或其他轮次权重；评测后核对 run 目录记录的 ckpt 路径。
- 候选均分高于上一轮评测均分即接受；相同则同一候选再评测一次，第二次相同或更高也接受（用户指定）。接受后在 `outputs/act_residual/<任务目录>/useful_checkpoints/` 建立 `<训练目录时间戳>.ckpt` 软链接，不复制 checkpoint。无效则回退本轮起点并诊断。
- **链式推进（用户最新指定，覆盖旧回退规则）**：下一轮训练一律从上一轮训练产出的 `policy_last.ckpt` 恢复（残差、优化器、RNG、累计进度与布局游标），**即使本轮评测无效也继续用该 ckpt，不回退**。区别在于：评测有效（均分判据通过）则保持训练逻辑与超参数直接续训；评测无效则**下一轮训练前先优化 Residual PPO**（依据研究报告/lessons/实测做有根据的单变量调整，如奖励、lr、std、KL、归一化、value-only warmup 等），优化后再从该 ckpt 续训。整轮零任务得分仍须立即检查场景、观测、动作、评分和残差幅度，不能仅凭零分断定实现故障。
- **每轮研究-优化时序（用户最新指定，round15 起，取代固定 8 小时周期研究 cron）**：每轮八小时训练进行到**第五小时（距结束约 3 小时，elapsed≈18000s）**时，助手在 20 分钟监督中检测到该时刻后**手动启动一次 dynamic Workflow 多代理研究**（ultracode 已开启，不开发自动调度器）。研究主题与切入点由助手依据**本轮前 5 小时的实测训练数据**（逐回合得分/峰值、均值修正与探索残差、势能分项 reach/lift/pair/release、advantage_std、价值拟合 ev/value_loss、显存与波次耗时爬升、OOM 崩溃恢复等）自行选定，目的是为**本轮八小时训练结束后如何优化 Residual PPO** 给出有依据的单变量方案（指向源码/论文/本项目实测）。研究产出报告保存到 `/home/cc/.cache/act-rounds-cEr5Ot/research-reports/`，不得干扰正在运行的训练/评测进程。训练自然停止、验证 ckpt 并完成官方评测后，**必须等待该研究完成**再决定优化：评测无效则据研究报告+lessons+实测做单变量优化再从本轮 ckpt 续训；评测有效则保持超参直接续训（报告仍留存供后续学习）。原固定 8 小时研究 cron（00261e8c）已废弃，研究改为按轮触发。
- 稳定性确认可固定同一候选 checkpoint 连续评测，不在确认期间改变被测策略。训练分数、loss、价值拟合改善不能代替独立评测。

## 监督与资源

- 用户已授权助手自行分析评测结果、选择或回退 checkpoint、做必要调整和测试，并持续逐轮训练与独立评测。**所有决策一律由助手自行做出并执行，不向用户发送选择题或请求决断，也不提交方案文档等待批准**；合适且已被验证（源码/论文/本项目实测）的改动直接采纳执行，适合 stack_blocks 任务的直接用，保持最简实现、不复杂化。每次阶段切换由助手手动执行，不开发或启动自动调度器。
- 每小时称呼用户“帅哥”简洁汇报真实进度，异常立即处理并汇报；保留日志与显存监控。
- 用户提出新要求时，将合理、长期适用且服务于 stack_blocks 在线强化学习目标的约定及时整合进本文件，合并或替换过时条款；临时运行状态与实验明细保存在项目外状态记录，不堆入本文件。
- 活动运行信息保存在项目外 `/home/cc/.cache/act-rounds-cEr5Ot/manual-state.json`。恢复会话先读该记录，再核对实际进程、容器、连接和日志，避免重复启动；历史轮次细节留在状态记录中。
- 切换训练/评测时检查 5555 端口及实际连接。RemoteEnv 关闭后 Kit 可能仍存活；确认旧客户端结束，再回收本轮自有容器，由现有启动器或评测脚本启动新服务。
- 两场景的相机 render product 数应保持为 3。必须先恢复布局、`apply_saved_poses()` 和相机预热，再初始化机器人与评分；不能只选择布局文件而未恢复桌子和积木。
- 本体位置是腕部 link6，不是手指抓取点；手指沿局部 +x 延伸约 0.09–0.16 米。不能以腕部高度直接断言没有下探。
- 整轮零任务得分时检查场景、观测、动作、评分和实际残差幅度。当前基座布局 0 的诊断抓取横向偏差约 35 毫米，可用来核对修正范围。
- 磁盘空间有限，不重复保存相同权重，不自行删除既有产物；空间不足明确报告。复用已有资产软链接与只读挂载，不复制资产、不修改上游子模块。
- 分轮运行和调参脚本放项目外；项目内仅修改必要算法、训练器、接入、配置、文档和测试。**每训练完一轮（无论本轮做了优化还是无需优化继续训练），下一轮开始前提交一次 git**（用户明确授权，覆盖默认不提交规则）；提交在当前 `act-residual-ppo` 分支，只 commit 不 push/pull，不提交 outputs/权重（已 gitignore）。其余未经明确要求的 commit/pull/push 仍禁止。

## 实现来源

- ResiP：*From Imitation to Refinement -- Residual RL for Precise Assembly*，ICRA 2025，https://arxiv.org/abs/2407.16677 。官方实现：https://github.com/ankile/robust-rearrangement ，重点核对 `src/train/residual_ppo.py`、`src/models/residual.py`、`src/gym/env_rl_wrapper.py`。
- ACT：https://github.com/tonyzhaozh/act ，arXiv:2304.13705；图像预处理、时间聚合和动作契约复用项目固定的 XPolicyLab 接入。
- PPO/GAE：Schulman 等，arXiv:1707.06347、1506.02438；成熟实现参考 https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py 。
- 势能塑形：Ng、Harada、Russell，*Policy Invariance Under Reward Transformations*，ICML 1999。后续堆叠奖励若调整，应核对 robosuite `environments/manipulation/stack.py` 或 IsaacLab stack/lift 实现，并按本任务几何适配，不能把未验证的自定奖励声称为论文原实现。
- 势能塑形终端条件（Grzes, AAMAS 2017, https://aamas.csc.liv.ac.uk/Proceedings/aamas2017/pdfs/p565.pdf ）：有限时域须在真终止状态令 Φ(s_N)=0，唯一能改变策略的项是 γ^N·Φ(s_N)。本项目 `act_online.py` 已对 terminated 取 0、对时间上限截断保留 φ(next) 并 bootstrap（截断按非终止处理的依据是 Pardo et al., ICML 2018，arXiv:1712.00370），两者自洽，终端条件已核对、不是缺陷。
- ResiP 固定 commit `cf80d12d` 的残差 PPO 默认值（`src/config/base_residual_rl.yaml`，已逐项核对）：1024 环境、1e9 步、`num_minibatches: 1`（全批）、`update_epochs: 50`、`target_kl: 0.1`、`max_grad_norm: 1.0`、actor lr 3e-4、critic lr 5e-3、`normalize_reward: true`（RunningMeanStdClip，clip 5.0）、`residual_l1/l2: 0.0`、`n_iterations_train_only_value: 0`。这是固定基线而非调优最优（作者自己 sweep 过 target_kl 0.01–0.3 等）；本项目 target_kl 0.02、无奖励归一化等偏离属本项目选择，不能据此断言是零分根因。`residual_regularization` 是废弃键，代码不读取。
- ResiP 残差 actor 初始化（`src/models/residual.py` + `src/config/actor/*.yaml`，已核对）：末层无 bias、`action_head_std: 0.0` 零起点。MLP 残差（headline 结果）用 `init_logstd: -1.5`、`learn_std: true`、`action_scale: 0.1`；diffusion 残差用 `init_logstd: -1.0`、`learn_std: false`。本项目零初始化均值头一致；固定 std 0.1 只有 diffusion 变体的出处，对 MLP 残差无官方支持，是否合适需受控实验。
- ResiP value-only warmup（`src/train/residual_ppo.py`，已核对）：`n_iterations_train_only_value` 前 N 次迭代把整个 policy loss（PPO+L1/L2）置 0 只训 critic；默认 0，官方 sweep 搜 2–5，mug_rack 脚本用 20。该数值按 1024 环境×700 步调出，本项目 2 环境每轮约 24–28 次更新不能照搬；只可作为有出处的小范围受控实验，不是已证实的零分修复。
- ResiP 能力边界与样本量（arXiv:2407.16677v2 正文+附录，已核对原文）：残差是局部修正（"designed to act locally, constrained by the base"），收益案例基座起点 one_leg 54%→98%、peg-in-hole 5%→99%，低起点不封顶；但初始场景随机性更高时"the training performance drops to zero, causing the learning to collapse"（即使基座起点成功率相近），ACT 基座只在 low randomness 下验证过。单步残差约需 75M 环境步达 ~85%（分块残差约 250M，1024 环境）；本项目八小时一轮约 25–30 万步，加预算只是有意义 on-policy 对比的前提，不是修复。"基座须先产生非零 15 分"是本项目推论，不是论文结论。
- 在相关代码注明具体来源与适配范围；不能仅加论文名就声称代码来自该项目，也不能照搬不同机器人、并行规模的超参数。社区/Stack Overflow 观点未经本项目实测不得当作已证实根因或指定修复。

### 可借鉴要点（来源已核验，采纳需本项目实测）

- **优化必须有依据**：每次调整奖励、超参数或训练逻辑前，先参考研究报告 `/home/cc/.cache/act-rounds-cEr5Ot/research-report.md`（Residual PPO 来源核验）、`reward-audit-report.md`（内部奖励审计）与项目根目录 `lessons.md`（失败教训记录），以及 CLAUDE.md 实现来源；改动须能指向源码/论文/本项目实测数据中的具体依据，不能靠感觉。单变量对照，逐轮验证。每轮被拒、实验失败或发现缺陷后，把已验证的教训追加进 `lessons.md`，防止再犯。

- **KL 阈值别卡死学习量**：ResiP 默认 `target_kl 0.1`；我们曾用 0.02 导致单次更新最少只跑 6 个 optimizer step 就早停。round11 起改 0.1 后跑满 700 步、KL 仅 ~0.011，学习量约 8 倍且未失稳。这是已采纳并有实测支撑的借鉴。
- **value-only warmup 可小范围试**：shaping 回报下 critic 不准会让 actor 先学偏。ResiP 有现成开关（前 N 次迭代只训 critic），我们每轮约 24–28 次更新，若试只能取 N=1–2 并单独一轮对照，不与其他改动混做。
- **奖励归一化是候选实验**：ResiP 用 RunningMeanStdClip(clip 5.0) 归一化回报；我们的势能项量级（±5–7）比 15 分档位还大且未归一化，critic 拟合难度更高。若价值 loss 长期不降，可对照试归一化，属未验证候选。
- **固定探索 std 无 MLP 出处**：ResiP MLP 残差是 `learn_std: true`（init_logstd -1.5 ≈ std 0.22）；我们固定 0.1 只有 diffusion 变体支持。若均值修正长期远小于探索噪声，可对照试 learn_std 或更大初始 std。
- **残差是局部修正，别指望它救宏观失败**：论文附录明确高初始随机性会让训练崩到 0（基座不差也一样）。整批采样几乎无接近成功的回合时，先查基座/场景/评分，而不是加大残差或延长训练。
- **样本量级差两个数量级**：ResiP 单步残差 ~75M 步（1024 环境）；我们八小时 ~30 万步。逐轮小步验证 + 链式接受有效 checkpoint 是当前规模下唯一可行路径，单轮结果波动大是预期内的。
- 完整研究记录（55 代理、94 发现、24 核验、7 存活，含被否决说法与未决问题）：`/home/cc/.cache/act-rounds-cEr5Ot/research-final.json` 与 `research-synthesis.json`。
