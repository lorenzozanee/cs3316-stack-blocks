# 项目起点：方舟 X5 双臂堆积木

命令在 `startcode/` 下执行。这里提供了仿真环境、示范数据读取，以及 [Diffusion](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/DP/README.md) 和 [ACT](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/ACT/README.md) 两条链路的启动方式。Flow Matching 已提供离线训练和冻结基座的残差 PPO 在线训练；评测服务仍需实现。完成一种路线即可。

不启动仿真时，可以先检查环境和数据接口：

```bash
PYTHONPATH=src python3 -m unittest tests.test_env_data
```

`NotImplementedError` 来自 `src/starter/policies/`。这是留给你的入口，不是环境故障。

## 和选题的对应关系

这版起点只覆盖期中的仿真阶段。真机部署、4 块积木和域随机化不在代码里，期末由课程组另行提供机器人与部署接口。

选题要求三块纹理不同的积木从随机位置堆稳，并在规定步数内把双臂收回。策略输入是头部相机和本体状态，输出是末端位姿，因此动作类型用 `ee`。不要把积木位姿送进策略。外部奖励不能改。

建议的路线是离线预训练再在线强化学习，基线可参考 XPolicyLab。本目录给出 Diffusion Policy 与 ACT 两条这样的链路，完成其中一种即可。选题也允许用 PPO、SAC 或其他算法，配上内在奖励、探索或课程学习，直接在线训练。那条路线没有单独的启动脚本；若采用，仍使用这里的环境和下面的评测命令。不带这些设计的随机高斯策略，看不到成功样本，不能当作完成。

使用公开库时，报告里写明来源、改了什么，以及关键训练参数。保留配置、随机种子、学习曲线和检查点，不必保存全部交互数据。

## 目录里没有的大文件

起始代码不包含镜像、资产、示范数据和上游仓库。磁盘请预留数百 GB。训练得到的 checkpoint 也不在里面，由你自己保存。


| 组件                                  | 是否必须      | 获得方式                                                                               |
| ----------------------------------- | --------- | ---------------------------------------------------------------------------------- |
| 仿真镜像 `stack-blocks-sim:isaacsim5.1` | 在线训练和评测必须 | 从下面的数据集下载 `stack-blocks-sim_isaacsim5.1.tar`，再 `docker load`                       |
| 策略镜像 `stack-blocks-policy:dp-act`   | 不必须，只供参考  | 同一数据集里的 `stack-blocks-policy_dp-act.tar`，再 `docker load`                           |
| RoboDojo、XPolicyLab 源码              | 必须        | `bash scripts/bootstrap_third_party.sh`                                            |
| Isaac 子模块                           | 不必须       | 只有自己重新构建仿真镜像时才执行 `git -C third_party/RoboDojo submodule update --init --recursive` |
| 场景资产 `Assets/`                      | 必须，不在镜像里  | `bash third_party/RoboDojo/scripts/init_assets.sh`                                 |
| 离线示范                                | 必须        | `bash scripts/download_stack_blocks.sh`                                            |


锁定版本写在 `third_party/pins.lock`：


| 仓库         | commit                                     |
| ---------- | ------------------------------------------ |
| RoboDojo   | `f57e1633ffed59761bc8e568cfe7530c4b33381c` |
| XPolicyLab | `aa46382aa12dc46b360e2fdc63d0d3d39215b3e9` |


```bash
bash scripts/bootstrap_third_party.sh
bash third_party/RoboDojo/scripts/init_assets.sh
bash scripts/download_stack_blocks.sh
```

`init_assets.sh` 用 `git` 和 `git lfs` 做稀疏检出，从 Hugging Face 数据集 `RoboDojo-Benchmark/RoboDojo` 只取本任务用到的机器人、房间、方块、杂物、材质和 `arx_x5` 布局，然后链到 `third_party/RoboDojo/Assets`。它不使用 `huggingface-cli`。

示范脚本只下载 `data/RoboDojo/stack_blocks/arx_x5`，并链到 `data/RoboDojo/stack_blocks/arx_x5`。命令是新版 `hf download`，没有 `hf` 时用旧版 `huggingface-cli download`。选题也允许从 ModelScope 取同一份离线数据；若从那边下载，仍放到这个目录结构里。在线训练和评测都用这个任务，不要改成 `stack_blocks_random`。

已发布的仿真镜像里已经有 Isaac Sim。使用这个镜像时不必初始化 Isaac 子模块；那一步只在自己重新构建仿真镜像时才需要。

上游仓库里有别人的策略实现。要交的是你自己在对应入口里写的模型，而不是把上游训练脚本的输出直接当作结果。

## 两个镜像

两个 tar 包都在 [NantianHe/stack-blocks-images](https://huggingface.co/datasets/NantianHe/stack-blocks-images)。页面上的 Dataset Viewer 会报错，因为这两个文件是容器镜像包，不是可预览的数据集。直接下载文件即可。


| 文件                                 | 导入后的标签                         |
| ---------------------------------- | ------------------------------ |
| `stack-blocks-sim_isaacsim5.1.tar` | `stack-blocks-sim:isaacsim5.1` |
| `stack-blocks-policy_dp-act.tar`   | `stack-blocks-policy:dp-act`   |


直接下载地址写在 `configs/sim_image.env`。仿真包大约 37 GB，策略包大约 800 MB。有 `hf` 时：

```bash
hf download NantianHe/stack-blocks-images \
  stack-blocks-sim_isaacsim5.1.tar \
  stack-blocks-policy_dp-act.tar \
  --repo-type dataset \
  --local-dir .
```

没有 `hf` 时，把命令里的 `hf` 换成 `huggingface-cli`。下载完成后导入。这是 tar 包，不要对这两个链接执行 `docker pull`。

```bash
docker load -i stack-blocks-sim_isaacsim5.1.tar
docker image inspect stack-blocks-sim:isaacsim5.1 >/dev/null && echo "sim ok"

docker load -i stack-blocks-policy_dp-act.tar
docker image inspect stack-blocks-policy:dp-act >/dev/null && echo "policy ok"
```

本机的 `docker` 命令若实际是 Podman，同样使用这些命令。Podman 常常忽略 `--gpus`，仿真脚本因此改为只挂载 `env_gpu` 那一张卡。

仿真镜像里是 Isaac Sim。它不包含你的策略代码，也不包含 `Assets/` 和示范数据，这两样在运行时挂载。在线训练和评测脚本会自己 `docker run` 这个镜像。

策略镜像是参考环境，不是必须项。里面是 Ubuntu 22.04、Miniconda，以及两个空的 Python 3.10 环境 `dp` 和 `act`。没有 Isaac，没有训练好的权重，也还没有 PyTorch 或模型库。训练脚本不会启动它。可以不使用这个镜像，在宿主机用 conda 或 venv 自己装依赖，再激活后运行下面的脚本。

如果用它作参考，挂上本目录后进入容器，在 `dp` 或 `act` 里安装你的实现所需的包。只想对照上游 DP / ACT 的依赖时，可以在容器里对克隆下来的 XPolicyLab 执行 `policy/DP/install.sh` 和 `policy/ACT/install.sh`。离线训练可以在这个容器里跑。在线训练要能再启动仿真容器：把宿主机的容器套接字挂进策略容器，或者更直接地在宿主机环境里执行 `train_dppo.sh` / `train_act_residual.sh`，让这两个脚本去启动仿真镜像。

```bash
docker run --rm -it \
  --network host \
  -v "$PWD:/stack-blocks" \
  -w /stack-blocks \
  stack-blocks-policy:dp-act
```

容器内：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate dp    # 或 act
```



## 镜像在什么时候被调用

在线训练时，策略进程留在当前环境，Isaac 由镜像启动。

- `bash scripts/train_dppo.sh` 读取 `configs/env.yaml`，用 `scripts/run_online_env.sh` 执行 `docker run`，容器内运行 `scripts/isaac_env_server.py`。然后在当前环境调用 `starter.pipelines.diffusion` 的 online 阶段，训练冻结 Flow Matching 基座的残差 PPO。
- `bash scripts/train_act_residual.sh` 同样启动镜像，再调用 `starter.pipelines.act` 的 online 阶段。

离线阶段不启动镜像：

```bash
bash scripts/train_dp.sh
bash scripts/train_act.sh
```

运行 Flow Matching 离线训练前，在策略环境安装 `python3 -m pip install -r requirements-fm.txt`，并按前文准备示范和 XPolicyLab。使用全部示范，不做轨迹留出。默认预测 16 步动作、取最近两帧观测，训练 100 个 epoch；有 CUDA 时使用 `policy_gpu`，否则使用 CPU。可复制 `configs/env.yaml`，在副本中添加以下配置，再用 `bash scripts/train_dp.sh <配置文件>` 启动：

```yaml
diffusion:
  epochs: 100
  horizon: 16
  action_steps: 4
  n_obs_steps: 2
  image_size: 224
  batch_size: 16
  num_workers: 4
  lr: 0.0001
  vision_lr: 0.0001
  inference_steps: 20
```

头部 RGB 按需从 HDF5 解码，不把全部 224×224 图像放入内存。动作窗口延伸到轨迹最后一帧，末尾不足一块时重复最后一个示范动作；状态和动作的归一化范围从全部示范计算。训练使用轻量平移、颜色增强和 dropout 抑制过拟合。

模型从零初始化，不使用预训练权重：约 1.02 亿参数中，约 2166 万参数的 ViT 编码头部图像，其余约 8042 万参数的 DiT 预测动作速度场。训练使用 Flow Matching 回归、AdamW、5% 预热与余弦退火、混合精度、梯度裁剪和带预热的 EMA。视觉与动作参数默认学习率相同；`dropout`、`vision_lr` 等可覆盖。

`metrics.csv` 每轮记录 `train_loss`、`train_probe_action_mae`、`lr`、`grad_norm`、`samples` 和 `elapsed_s`。`train_probe_action_mae` 是从全部示范中抽取的固定训练窗口上的动作误差，不是独立验证集指标，也不能替代成功率。`steps.csv` 记录每次更新的损失、梯度范数和学习率。训练结束自动在 `outputs/dp/<run>/<stamp>/plots/` 生成损失曲线、动作误差曲线和学习率曲线，`config.json` 保存本次配置。

checkpoint 保存在 `outputs/dp/<run>/<stamp>/checkpoints/`：首轮、每 10 轮和末轮写出阶段 EMA 权重 `epoch_XXXX.ckpt`，`last.ckpt` 额外保存优化器、调度器、原始权重和随机数状态。离线阶段没有环境回报，报告中的回报曲线需由在线训练或仿真评测提供。

用 `starter.policies.diffusion_model.load_policy(path)` 加载离线 checkpoint，再调用 `predict_action_chunk(rgb, state)`，输入头部 RGB 和 16 维末端状态，得到 `[horizon, 16]` 动作；每个 episode 开头调用 `reset()`。复现实验时保留配置、种子、`metrics.csv`、曲线和 checkpoint；checkpoint 也记录了实际使用的示范轨迹。

### Flow Matching 残差 PPO 在线训练

入口文件名沿用历史 `train_dppo.sh`，实际训练的是冻结 Flow Matching 基座的残差 PPO，不是 DPPO。

先准备离线 checkpoint。在 `configs/env.yaml` 中加入必填项：

```yaml
residual_ppo:
  offline_checkpoint: outputs/dp/<run>/<stamp>/checkpoints/last.ckpt
```

运行：

```bash
bash scripts/train_dppo.sh
```

脚本固定读取 `configs/env.yaml`，要求 `policy_gpu` 和 `env_gpu` 不同，并通过 `scripts/run_online_env.sh` 以 `once=0` 启动常驻仿真服务；结束时停掉 `stack-blocks-dppo-env` 容器和后台服务。

在线超参默认在 Python 中：`total_timesteps=100000`、`rollout_steps=128`、`ppo_epochs=10`、`minibatch_size=256`、`lr=0.0003`、`gamma=0.99`、`gae_lambda=0.95`、`clip_coef=0.2`、`value_coef=0.5`、`entropy_coef=0.0`、`target_kl=0.02`、`max_grad_norm=0.5`、`hidden_dim=256`、`initial_std=0.2`、`position_scale=0.02`、`quaternion_scale=0.05`、`gripper_scale=0.1`、`checkpoint_every=10`、`stats_window=100`。GPU、端口、任务、场景数等环境设置仍从配置文件读取。

冻结的 Flow Matching 基座在每个重规划点用固定潜变量生成动作块，执行离线 checkpoint 中记录的 `action_steps` 步后重新规划。每一步的残差控制器读取当前头部 RGB 和 16 维本体，输出该步残差；残差经 tanh 和尺度限制，四元数归一化，夹爪裁到 `[0, 1]`。PPO 只对残差潜变量的高斯对数概率计算，基座不更新。环境得分保持 0/15/100，成功时 `terminated`；550 步截断时用价值函数 bootstrap。

产物写到 `outputs/dp/<run>/<stamp>/`：`config.json`、`metrics.csv`、`episodes.csv`、`checkpoints/update_XXXXXX.ckpt` 和 `checkpoints/last.ckpt`。`starter.policies.diffusion_model.load_policy(path)` 同时加载离线和在线 checkpoint；在线 checkpoint 返回残差控制器，其 `predict_action_chunk(rgb, state)` 只返回一个 16 维动作，因此每一步都要提供新观测。

在线训练每次更新在终端显示本轮已完成回合的平均回报、最近 `stats_window` 个已完成回合的平均回报和成功率，以及 actor/critic 损失。尚无完整回合时，回报和成功率留空，不记成零；滚动统计跨 PPO 更新保留。`episodes.csv` 逐回合保存累计环境交互步数、种子、长度、未经折扣的回报、最高/最终外部得分、成功与终止/截断标志；`metrics.csv` 保存逐次更新的回报、滚动统计和 PPO 指标。回报是每步外部得分之和，可能超过 100；成功率单独统计，15 分不算成功。

首轮更新、每隔 `checkpoint_every` 次更新和训练结束时自动刷新 `plots/`：

| 文件 | 内容 |
| --- | --- |
| `return_curve.png` | 完整回合原始回报与最近至多 `stats_window` 个回合的平均回报 |
| `success_rate.png` | 同一回合窗口内的训练成功率 |
| `episode_length.png` | 完整回合步数与滚动均值 |
| `loss_curve.png` | PPO policy/value loss，分别使用两个子图 |

横轴统一为所有并行场景累计的环境交互步数，回合曲线使用各回合实际完成的位置。还没有完整回合时显示无数据提示，不把 rollout 片段当作完整回合。Ctrl+C 后也会从已经写入的 CSV 刷新曲线。可单独重画已有日志：

```bash
PYTHONPATH=src python3 -m starter.policies.online_metrics "outputs/dp/<run>/<stamp>" --window 100
```

### 对照 PDF 的期中仿真交付要求

选题 PDF 第 3 页要求保留配置、种子、学习曲线和检查点；第 4 页要求报告中的阶段性回报/损失结果，并注明开源来源。代码来源、直接依赖和本地修改说明见 [`src/starter/policies/SOURCES.md`](src/starter/policies/SOURCES.md)。

| PDF 要求 | 当前覆盖情况 |
| --- | --- |
| 训练过程回报、损失等阶段性结果（第 4 页） | 在线 CSV、终端统计及上述四张曲线；离线损失曲线在离线训练目录中 |
| 配置、种子、检查点、可加载权重（第 3–4 页） | 每次训练保存；离线/在线权重均可通过 `load_policy()` 加载 |
| 技术报告中的算法、训练奖励、关键参数、改进与失败分析（第 3–4 页） | README、来源说明和运行日志提供依据，报告及对照/消融仍需根据实际实验撰写 |
| 离线与在线模型分别统一评测（第 2、4 页） | 尚未完成：`policies/diffusion/setup_eval_policy_server.sh` 仍为占位入口 |
| `_result.json` 和 `episode_*.mp4`（第 4、6 页） | 需实现评测服务并运行 RoboDojo 官方评测后产生，训练日志和训练成功率不能替代 |
| 声称改进时至少 3 个随机种子、完整曲线与评测指标对照（第 4 页） | 当前每次运行记录一个训练种子；需分别运行实验并评测，单次训练不满足此项 |

因此，`train_online()` 提供训练过程证据，尚不能单独产出全部期中提交材料。PDF 第 2 页的 `stack_blocks_random`、1/0 奖励表与当前起点代码的 `stack_blocks`、0/15/100 外部得分存在差异；本实现遵循当前仓库的任务约定，不修改外部奖励，提交时需确认课程实际使用的任务与评测口径。

评测时，策略服务在你的环境里监听端口，Isaac 客户端在镜像里运行。

```bash
bash scripts/eval_policy.sh configs/eval.yaml
```

`eval_policy.sh` 是在这台机器上拆开跑选题规定的评测：策略服务留在你的环境里，Isaac 客户端进仿真镜像。它先执行 `policy_dir` 指向的 `setup_eval_policy_server.sh`，再在容器里执行 RoboDojo 的 `scripts/robodojo.sh client`。容器内仿真卡写成 `cuda:0`，因为只挂载了 `env_gpu` 那一张卡。结果写到 RoboDojo 的 `eval_result/`。期中提交至少要有其中的 `_result.json` 和 `episode_*.mp4`。

选题原文的入口是：

```bash
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/<POLICY_NAME> \
  --task stack_blocks \
  --ckpt <CKPT_NAME> \
  --policy-env <POLICY_ENV> \
  --env-cfg arx_x5 \
  --action-type ee \
  --seed 0 \
  --eval-num native
```

策略要包成 XPolicyLab 能加载的目录，里面有 `setup_eval_policy_server.sh`。客户端按目录名到 `third_party/RoboDojo/XPolicyLab/policy/<名字>/deploy.py` 找适配。两阶段训练必须分别评离线检查点和在线检查点，在线相对离线的变化是评分的一项。若报告里写了相对基线的改进，对照至少用 3 个随机种子，并保留训练曲线和评测指标。

换 ACT 时，把 `policy_dir` 改成 `policies/act`，并换成该链路的 checkpoint 名。

## 任务约定

仿真任务是 `stack_blocks`，环境配置 `arx_x5`，动作类型 `ee`。离线示范、在线训练和评测都是这个任务：同一张背景、同一间房间、同一张桌子，没有杂物。各条评测布局里变化的是积木型号和桌面位置。

动作 16 维，世界基座坐标系：

```text
[left_ee_pose(7), left_gripper(1), right_ee_pose(7), right_gripper(1)]
```

位姿顺序是 `[x, y, z, qw, qx, qy, qz]`。夹爪下标是 7 和 15，送进环境前裁到 `[0, 1]`。关节角是 14 维，不要把它当成 `ee` 的宽度。官方数据脚本对 `arx_x5` 报的动作维度是关节的 14，这里要使用 16。

策略观测只有头部相机 `cam_head` 和这 16 维本体。`src/starter/envs/stack_blocks.py` 已经把腕部图像和积木位姿从策略观测里拿掉。积木位姿在 `get_privileged_state()`，只能用于训练时的内在奖励。`src/starter/rewards/intrinsic.py` 现在加的是 0。奖励可以改，但不要写回环境的 `reward_manager`，也不要改外部得分。

HDF5 在 `data/RoboDojo/stack_blocks/arx_x5/data/episode_*.hdf5`。`starter.data.demos.read_episode` 给出 16 维状态、16 维动作，以及 `vision/cam_head/colors` 里原样存放的图像缓冲。解码请使用克隆下来的 `XPolicyLab.utils.process_data.decode_image_bit`。这批数据里有旧的 JPEG 标记，用 PIL 直接解会把通道读反。文件里还有腕部相机，不要送进策略。图像如何缩放、是否堆历史帧，由你决定。

选题表里的奖励写成成功为 1、其余为 0。环境实际给出的外部得分不能改，数值是：


| 得分  | 含义             |
| --- | -------------- |
| 0   | 还没有两块积木堆在一起    |
| 15  | 任意两块堆在一起，且夹爪张开 |
| 100 | 三块堆稳，且双臂回到原点   |


100 就是选题里的成功，不要把它改写成 1，也不要把 15 改写成 100。回合最长 550 步。成功时 `terminated` 为真；走到 550 步仍未成功时 `truncated` 为真。环境异常或无效结束不是成功。不要在尚未成功时把回合立刻截断。15 分可以当作回报的一部分。

`env.reset` 返回 `(observation, info)`。`env.step` 返回 `(observation, score, terminated, truncated, info)`。在线采样用 `RemoteEnv`，一次连接里有 `configs/env.yaml` 的 `num_envs` 个场景。

## 需要实现的部分

两条链路都是先离线行为克隆，再在同一个 `stack_blocks` 任务里在线训练。环境设置写在你自己的配置里，在线超参默认在 Python 中。

1. `src/starter/policies/diffusion.py`
  `train_offline` 从示范训练 Flow Matching 基座；`train_online` 已实现冻结该基座的残差 PPO，用 `RemoteEnv` 与已经启动的仿真通信。`bash scripts/train_dp.sh` 和 `bash scripts/train_dppo.sh` 分别进入这两个函数。
2. `src/starter/policies/act.py`
  同样分离线和在线。入口是 `bash scripts/train_act.sh` 和 `bash scripts/train_act_residual.sh`。
3. `policies/diffusion/setup_eval_policy_server.sh` 和 `policies/act/setup_eval_policy_server.sh`
  参数顺序是：bench、task、ckpt、env_cfg、action_type、seed、policy_gpu、conda 环境名、端口、主机。在你的环境里加载对应 checkpoint，按 RoboDojo 评测客户端期待的协议应答。目录名要和 `deploy.py` 的查找对上，网络结构和 checkpoint 格式可以自己定。离线检查点和在线检查点按上一节分开评。



## 两条 pipeline 的原理

两条都是先在 `stack_blocks` 的示范上做行为克隆，再在同一个任务里用环境得分做在线强化学习。观测是头部相机和 16 维末端状态。一次预测可以是未来一小段动作，执行其中的若干步再重新规划。网络用卷积、Transformer 还是别的结构，历史帧取几帧，损失和学习率取多少，都由你定。

### Flow Matching，然后做冻结基座的残差 PPO

离线阶段把行为克隆写成条件 Flow Matching：给一个观测和噪声水平，网络预测从噪声指向示范动作的速度场；推理时从固定潜变量出发积分，得到动作块。在线阶段冻结基座，只训练一个小的残差控制器。

基座在每个重规划点用固定潜变量生成动作块，执行离线 checkpoint 中记录的 `action_steps` 步后重新规划。每一步残差控制器读取当前头部 RGB 和本体状态，输出一个受限残差并加到基座动作上。PPO 只对残差潜变量的高斯对数概率计算，不对基座的动作链或积分过程计算。残差应明显小于动作本身，四元数归一化，夹爪裁到 `[0, 1]`。回报直接使用环境得分 0/15/100，截断时用价值函数 bootstrap。PPO 实现可参考 CleanRL、Stable-Baselines3 等公开库。

### [ACT](https://github.com/tonyzhaozh/act)，然后做残差 PPO

离线阶段训练一个确定性策略，由当前观测直接预测一段未来动作，用示范做行为克隆，然后冻结。原始 ACT 用条件变分结构把动作块和图像对齐，测试时潜变量取零；你也可以用别的确定性序列模型，只要能冻结、并能在同样的观测上重复给出基座动作。

在线阶段不改基座。另学一个小的高斯残差，执行的动作是基座动作加残差。PPO 的对数概率只对残差计算。残差幅度应明显小于动作本身，否则一开始就会盖住已经克隆到的行为。基座权重在更新之后必须和冻结时相同。官方 ACT 常用三台相机；本作业只使用 `cam_head`。

不利用示范、也不设计内在奖励或课程的随机高斯策略，很难在 550 步内碰到 100 分。选题允许的直接在线学习，指的是 PPO、SAC 这一类，并配上合理的内在奖励、探索或课程，而且仍然不改外部得分。离线策略如果一次都堆不成功，在线阶段也看不到 100 分。15 分是中途信号，可以进入回报，不能改成成功。
