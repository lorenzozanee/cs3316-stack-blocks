# Policy Sources and Provenance

The implemented modules (`diffusion.py`, `diffusion_model.py`, `residual_model.py`,
`residual_ppo.py`, `online_metrics.py`) were written for this project. No upstream source files were copied or
vendored. `act.py` is the provided, untouched, unimplemented stub and is not part of this
implementation.

## Directly imported third-party code

| Source | License | Use |
| --- | --- | --- |
| [pytorch/vision](https://github.com/pytorch/vision) `VisionTransformer` | BSD-3-Clause | Image encoder, randomly initialized; classifier head replaced with identity |
| [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) `decode_image_bit` | see pinned upstream repository | HDF5 image decoding; version in `third_party/pins.lock` |
| PyTorch, torchvision, NumPy, h5py, matplotlib | upstream licenses | Numerical, data, and plotting libraries |

Offline training initializes the model from scratch. Online training loads the local offline checkpoint.

## Algorithm and architecture references (no code copied)

- **Flow matching / CFM** - Lipman et al., *Flow Matching for Generative Modeling*; Meta
  [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching)
  (CC BY-NC 4.0). Reference for the linear interpolant, velocity-regression target, and
  midpoint RK2 integration. Implemented locally in `diffusion_model.py`; no upstream module
  is imported.
- **DiT** - Peebles and Xie, *Scalable Diffusion Models with Transformers*;
  [facebookresearch/DiT](https://github.com/facebookresearch/DiT) (CC BY-NC 4.0). Reference
  for adaptive LayerNorm modulation and residual gating. Adapted here to cross-attend
  observation memory and predict action velocities rather than image tokens.
- **PPO** - Schulman et al., *Proximal Policy Optimization Algorithms*. CleanRL
  ([vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl), MIT) and Stable-Baselines3
  ([DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3), MIT) are
  conceptual prior art for the clipped surrogate, advantage normalization, KL early stop,
  and GAE. Implemented locally; neither package is imported.
- **Residual policy** - frozen-base plus bounded-residual composition is implemented here.
  No residual-RL package is imported.
- **Training plots** - `online_metrics.py` directly uses Matplotlib's
  [FigureCanvasAgg example](https://github.com/matplotlib/matplotlib/blob/main/galleries/examples/user_interfaces/canvasagg.py)
  as an API reference for rendering PNGs without a GUI. Rolling statistics and CSV handling are local.

## Local adaptations

- Inputs are the head camera and 16-D end-effector state only; no privileged block poses.
- A ViT encodes a short observation history; an action transformer cross-attends to that
  memory and predicts the flow velocity for an action horizon.
- During online training and inference, the frozen base generates action chunks with fixed
  latent noise; PPO never updates it. Offline flow sampling remains stochastic.
- PPO scores a Gaussian latent, which is mapped to a per-step bounded residual
  (`tanh(latent) * scale` for position, quaternion, and gripper), then normalized/clipped.
- GAE bootstraps values at time-limit truncation but does not carry across any
  terminated-or-truncated episode boundary.
- The residual network observes fresh images/states every step while the base chunk only
  replans every `action_steps` steps.

## Hyperparameters and run provenance

Defaults live in code and are overridable, so the values actually used are recorded with
each run rather than asserted here.

- Offline defaults (`train_offline` in `diffusion.py`, documented in the repository README):
  100 epochs, horizon 16, action steps 4, 2 observation frames, 224 px images, batch 16,
  learning rates 1e-4, 20 inference steps; ViT width/depth/heads 384/12/6; DiT
  512/12/8; dropout 0.1.
- Online defaults (`PPOOptions` in `residual_ppo.py`): 100,000 timesteps, rollout 128,
  10 epochs, minibatch 256, learning rate 3e-4, gamma 0.99, GAE lambda 0.95, clip 0.2,
  value coefficient 0.5, entropy coefficient 0.0, target KL 0.02, gradient norm 0.5,
  hidden width 256, initial residual std 0.2, scales 0.02/0.05/0.1, checkpoint every 10
  updates; trailing training statistics use up to 100 completed episodes.
- `configs/env.yaml` holds environment/data settings; optional `diffusion` and
  `residual_ppo` sections override the code defaults.
- Per-run `config.json`, `metrics.csv`, `episodes.csv`, `steps.csv`, plots, and checkpoints
  under `outputs/dp/<run>/<stamp>/` are the authoritative provenance for seeds, actual
  hyperparameters, and learned weights.
