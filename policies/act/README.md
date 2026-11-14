# ACT 评测接入

本目录负责评测适配；模型、离线训练和冻结基座的残差 PPO 在线训练使用 `src/starter/policies/act*.py`。支持 `RoboDojo / stack_blocks / arx_x5 / ee`，输入只使用 `cam_head` RGB 和 16 维末端状态。

## 环境

在项目根目录、已有 `act` conda 环境中执行：

```bash
conda activate act
bash policies/act/install.sh
```

依赖复用离线训练环境，仅增加 WebSocket / msgpack / pydantic 通信依赖。仿真使用本机已有 `localhost/stack-blocks-sim:isaacsim5.1` 镜像。`third_party/RoboDojo/Assets` 链接已下载的共享资产，资产以只读方式挂载。

运行时还需要这些链接：

- `third_party/RoboDojo/XPolicyLab -> ../XPolicyLab`
- `third_party/XPolicyLab/policy/act -> ../../../policies/act`
- `third_party/env_cfg -> RoboDojo/env_cfg`

这些链接让 RoboDojo 找到本目录下的适配器，让 XPolicyLab 使用本项目的机器人配置；不复制或修改上游模型。

## 权重

`checkpoints` 是指向本项目训练输出目录的软链接，不复制权重。目录需要包含：

```text
checkpoints/
  config.json
  dataset_stats.pkl
  policy_last.ckpt
  policy_epoch_<epoch>_seed_<seed>.ckpt
```

`configs/eval.yaml` 的 `ckpt` 指向具体 `.ckpt` 文件，默认 `policies/act/checkpoints/policy_last.ckpt`。最后权重在训练结束时写出；也可指定同目录下的中间检查点。权重通过保存的配置和统计量加载，不重新下载 ResNet 权重。开始新的训练运行后，需显式更新 `checkpoints` 链接或评测配置，避免误评其他模型。

残差 PPO 在线产物位于 `outputs/act_residual/`。在线 `.ckpt` 包含冻结 ACT、残差 actor/critic 和归一化统计量；同目录保留标记 `policy_type: act_residual_ppo` 的 `config.json` 即可，不再需要额外 `dataset_stats.pkl`。将 `configs/eval.yaml` 的 `ckpt` 指向它，仍使用同一适配器。在线评测与训练一致，每步时间聚合基座动作，再加确定性残差均值；必须保持 `temporal_agg: true`。离线 checkpoint 继续支持整块动作执行和时间聚合两种评测方式。

## 检查和评测

```bash
# 只检查路径、镜像和挂载配置，不启动服务或 Isaac。
bash scripts/eval_policy.sh --dry-run

# 正式评测，由用户需要时启动。
bash scripts/eval_policy.sh configs/eval.yaml
```

默认策略和仿真共用 GPU 0，仿真并行数为 3。`num_envs` 只写入每次运行生成的仿真配置，不修改共享 RoboDojo 配置。正式评测应在训练结束后单独运行。

结果、视频和恢复记录保存在 `outputs/eval/results/`；每次运行的配置、策略日志和仿真日志保存在 `outputs/eval/<run_id>/`。结果不会写回共享资产所在项目。

`deploy.yml` 默认启用与上游 ACT 相同的时间聚合：每个控制步预测 50 步动作块，按照 `exp(-0.01 * age_index)` 对当前时刻的重叠预测加权，顺序与上游一致（较早预测权重较高）。各仿真环境独立保留历史，回合重置时清空。关闭 `temporal_agg` 后每次返回完整动作块。

已配置环境不代表仿真任务评测通过；正式运行后再检查 `_result.json` 和回放视频。
