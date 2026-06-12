"""RL configuration for PKL-based motion tracking."""

from __future__ import annotations

import dataclasses

from mjlab.rl import RslRlPpoAlgorithmCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

from wbc_mjlab.dagger_ppo_config import WbcDaggerAlgorithmCfg, WbcDaggerRunnerCfg
from wbc_mjlab.dagger_ppo_config import WbcMoEDaggerAlgorithmCfg, WbcMoEModelCfg
from wbc_mjlab.dagger_ppo_config import WbcTeacherSlotCfg, WbcTwistModelCfg
from wbc_mjlab.g1_constants_custom import _WBC_SAVE_INTERVAL
from wbc_mjlab import observations as wbc_obs


def _flat_model_cfg(base_model) -> WbcTwistModelCfg:
  return WbcTwistModelCfg(
    hidden_dims=base_model.hidden_dims,
    activation=base_model.activation,
    obs_normalization=base_model.obs_normalization,
    distribution_cfg=base_model.distribution_cfg,
  )


def _wbc_algorithm(
  base_algo: RslRlPpoAlgorithmCfg,
  *,
  dagger_coef: float = 0.0,
  dagger_coef_min: float = 0.0,
  dagger_coef_anneal_steps: int = 0,
  residual_reg_coef: float = 0.0,
  loco_num_actions: int = 15,
  loco_blend_dims: int = 15,
  loco_teacher_action_std_rescale: float = 1.0,
  arm_kl_coef: float | None = None,
  arm_kl_coef_min: float | None = None,
) -> WbcDaggerAlgorithmCfg:
  """Create a WbcDaggerAlgorithmCfg by copying PPO fields from *base_algo*."""
  parent_fields = {
    f.name: getattr(base_algo, f.name)
    for f in dataclasses.fields(RslRlPpoAlgorithmCfg)
    if f.name != "class_name"
  }
  return WbcDaggerAlgorithmCfg(
    **parent_fields,
    dagger_coef=dagger_coef,
    dagger_coef_min=dagger_coef_min,
    dagger_coef_anneal_steps=dagger_coef_anneal_steps,
    residual_reg_coef=residual_reg_coef,
    loco_num_actions=loco_num_actions,
    loco_blend_dims=loco_blend_dims,
    loco_teacher_action_std_rescale=loco_teacher_action_std_rescale,
    arm_kl_coef=arm_kl_coef,
    arm_kl_coef_min=arm_kl_coef_min,
  )


def _wbc_moe_algorithm(
  base_algo: RslRlPpoAlgorithmCfg,
  *,
  load_balance_coef: float = 0.01,
  dagger_coef: float = 0.0,
  dagger_coef_min: float = 0.0,
  dagger_coef_anneal_steps: int = 0,
  residual_reg_coef: float = 0.0,
  loco_num_actions: int = 15,
  loco_blend_dims: int = 15,
  loco_teacher_action_std_rescale: float = 1.0,
  arm_kl_coef: float | None = None,
  arm_kl_coef_min: float | None = None,
) -> WbcMoEDaggerAlgorithmCfg:
  """Create a WbcMoEDaggerAlgorithmCfg by copying PPO fields from *base_algo*."""
  parent_fields = {
    f.name: getattr(base_algo, f.name)
    for f in dataclasses.fields(RslRlPpoAlgorithmCfg)
    if f.name != "class_name"
  }
  return WbcMoEDaggerAlgorithmCfg(
    **parent_fields,
    load_balance_coef=load_balance_coef,
    dagger_coef=dagger_coef,
    dagger_coef_min=dagger_coef_min,
    dagger_coef_anneal_steps=dagger_coef_anneal_steps,
    residual_reg_coef=residual_reg_coef,
    loco_num_actions=loco_num_actions,
    loco_blend_dims=loco_blend_dims,
    loco_teacher_action_std_rescale=loco_teacher_action_std_rescale,
    arm_kl_coef=arm_kl_coef,
    arm_kl_coef_min=arm_kl_coef_min,
  )


def unitree_g1_pkl_tracking_custom_ppo_runner_cfg() -> WbcDaggerRunnerCfg:
  """Custom-PPO config for PKL tracking using HANDOFF-style structured observations."""

  base = unitree_g1_tracking_ppo_runner_cfg()
  actor = WbcTwistModelCfg(
    hidden_dims=base.actor.hidden_dims,
    activation=base.actor.activation,
    obs_normalization=base.actor.obs_normalization,
    distribution_cfg=base.actor.distribution_cfg,
    current_mimic_dim=wbc_obs.ACTOR_MIMIC_DIM,
    current_proprio_dim=wbc_obs.ACTOR_PROPRIO_DIM,
    history_feature_dim=wbc_obs.ACTOR_HISTORY_FEATURE_DIM,
    history_length=wbc_obs.ACTOR_HISTORY_LENGTH,
    history_latent_dim=64,
    motion_latent_dim=64,
    num_motion_observations=wbc_obs.ACTOR_MIMIC_DIM,
    num_motion_steps=1,
    num_priop_observations=wbc_obs.ACTOR_PROPRIO_DIM,
    num_history_steps=wbc_obs.ACTOR_HISTORY_LENGTH,
    num_future_observations=0,
    num_future_steps=0,
  )
  critic = WbcTwistModelCfg(
    hidden_dims=base.critic.hidden_dims,
    activation=base.critic.activation,
    obs_normalization=base.critic.obs_normalization,
    distribution_cfg=base.critic.distribution_cfg,
    privileged_future_step_dim=wbc_obs.CRITIC_PRIV_STEP_DIM,
    privileged_future_steps=wbc_obs.CRITIC_PRIV_STEPS,
    critic_current_dim=wbc_obs.ACTOR_PROPRIO_DIM,
    critic_extras_dim=wbc_obs.critic_extras_dim(),
    motion_latent_dim=64,
    history_latent_dim=64,
    num_motion_observations=wbc_obs.critic_priv_step_dim() * wbc_obs.CRITIC_PRIV_STEPS,
    num_motion_steps=wbc_obs.CRITIC_PRIV_STEPS,
    num_priop_observations=wbc_obs.ACTOR_PROPRIO_DIM,
    num_history_steps=0,
    num_future_observations=0,
    num_future_steps=0,
  )
  algorithm = _wbc_algorithm(base.algorithm)
  return WbcDaggerRunnerCfg(
    seed=base.seed,
    num_steps_per_env=base.num_steps_per_env,
    max_iterations=base.max_iterations,
    obs_groups={
      "actor": ("actor_current", "actor_history"),
      "critic": (
        "critic_priv_future_sequence",
        "critic_current",
        "critic_extras",
      ),
    },
    save_interval=_WBC_SAVE_INTERVAL,
    experiment_name="g1_pkl_tracking_custom_ppo",
    run_name=base.run_name,
    logger=base.logger,
    wandb_project="wbc_mjlab",
    wandb_tags=base.wandb_tags,
    resume=base.resume,
    load_run=base.load_run,
    load_checkpoint=base.load_checkpoint,
    clip_actions=base.clip_actions,
    upload_model=base.upload_model,
    actor=actor,
    critic=critic,
    algorithm=algorithm,
    teacher_experiment_name=None,
    teacher_proj_name=None,
    teacher_checkpoint=None,
    loco_teacher_experiment_name=None,
    loco_teacher_proj_name=None,
    loco_teacher_checkpoint=None,
    wbc_teacher_actor=None,
    wbc_teacher_obs_groups=None,
    wbc_teacher_obs_set=None,
    loco_teacher_actor=None,
    loco_teacher_obs_groups=None,
    loco_teacher_obs_set=None,
    blend_obs_group=None,
    eval_student=False,
  )


def unitree_g1_loco_teacher_flat_nobv_runner_cfg() -> WbcDaggerRunnerCfg:
  """Runner config for the non-privileged (no base_lin_vel) loco teacher."""
  cfg = unitree_g1_loco_teacher_flat_runner_cfg()
  cfg.experiment_name = "g1_loco_teacher_flat_nobv"
  cfg.run_name = "g1_loco_teacher_flat_nobv"
  return cfg






def unitree_g1_loco_teacher_flat_runner_cfg() -> WbcDaggerRunnerCfg:
  base = unitree_g1_tracking_ppo_runner_cfg()
  actor = _flat_model_cfg(base.actor)
  critic = _flat_model_cfg(base.critic)
  algorithm = _wbc_algorithm(base.algorithm, loco_num_actions=wbc_obs.NUM_LOCO_ACTIONS)
  return WbcDaggerRunnerCfg(
    seed=base.seed,
    num_steps_per_env=base.num_steps_per_env,
    max_iterations=20_001,
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
    },
    save_interval=_WBC_SAVE_INTERVAL,
    experiment_name="g1_loco_teacher_flat",
    run_name="g1_loco_teacher_flat",
    logger=base.logger,
    wandb_project="wbc_mjlab",
    wandb_tags=base.wandb_tags,
    resume=base.resume,
    load_run=base.load_run,
    load_checkpoint=base.load_checkpoint,
    clip_actions=base.clip_actions,
    upload_model=base.upload_model,
    actor=actor,
    critic=critic,
    algorithm=algorithm,
    teacher_experiment_name=None,
    teacher_proj_name=None,
    teacher_checkpoint=None,
    loco_teacher_experiment_name=None,
    loco_teacher_proj_name=None,
    loco_teacher_checkpoint=None,
    wbc_teacher_actor=None,
    wbc_teacher_obs_groups=None,
    wbc_teacher_obs_set=None,
    loco_teacher_actor=None,
    loco_teacher_obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
    },
    loco_teacher_obs_set="actor",
    blend_obs_group=None,
    eval_student=False,
  )




def unitree_g1_hand_moe_flat_runner_cfg() -> WbcDaggerRunnerCfg:
  """Runner config for the MoE dual-teacher hand student variant.

  Identical to the dual-teacher config except the actor uses
  ActorCriticFutureMoE and the algorithm uses MoEDaggerPPO.
  """
  wbc_teacher_cfg = unitree_g1_pkl_tracking_custom_ppo_runner_cfg()
  loco_teacher_cfg = unitree_g1_loco_teacher_flat_runner_cfg()

  _student_hidden_dims = (512, 512, 256, 128)
  _student_activation = "swish"

  actor = WbcMoEModelCfg(
    hidden_dims=_student_hidden_dims,
    activation=_student_activation,
    obs_normalization=wbc_teacher_cfg.actor.obs_normalization,
    distribution_cfg=wbc_teacher_cfg.actor.distribution_cfg,
    layer_norm=True,
    current_mimic_dim=wbc_obs.HAND_STUDENT_MIMIC_DIM,
    current_proprio_dim=wbc_obs.ACTOR_PROPRIO_DIM,
    history_feature_dim=wbc_obs.HAND_STUDENT_HISTORY_FEATURE_DIM,
    history_length=wbc_obs.ACTOR_HISTORY_LENGTH,
    history_latent_dim=64,
    motion_latent_dim=128,
    num_motion_observations=wbc_obs.HAND_STUDENT_MIMIC_DIM,
    num_motion_steps=1,
    num_priop_observations=wbc_obs.ACTOR_PROPRIO_DIM,
    num_history_steps=wbc_obs.ACTOR_HISTORY_LENGTH,
    num_future_observations=0,
    num_future_steps=0,
    num_experts=2,
    expert_hidden_dims=(256, 128),
    gate_hidden_dims=(128, 64),
  )
  critic = WbcTwistModelCfg(
    hidden_dims=_student_hidden_dims,
    activation=_student_activation,
    obs_normalization=wbc_teacher_cfg.critic.obs_normalization,
    distribution_cfg=wbc_teacher_cfg.critic.distribution_cfg,
    layer_norm=True,
    privileged_future_step_dim=wbc_obs.CRITIC_PRIV_STEP_DIM,
    privileged_future_steps=wbc_obs.CRITIC_PRIV_STEPS,
    critic_current_dim=wbc_obs.ACTOR_PROPRIO_DIM,
    critic_extras_dim=wbc_obs.critic_extras_dim(),
    motion_latent_dim=64,
    history_latent_dim=64,
    num_motion_observations=wbc_obs.critic_priv_step_dim() * wbc_obs.CRITIC_PRIV_STEPS,
    num_motion_steps=wbc_obs.CRITIC_PRIV_STEPS,
    num_priop_observations=wbc_obs.ACTOR_PROPRIO_DIM,
    num_history_steps=0,
    num_future_observations=0,
    num_future_steps=0,
  )
  algorithm = _wbc_moe_algorithm(
    wbc_teacher_cfg.algorithm,
    load_balance_coef=0.01,
    dagger_coef=0.2,
    dagger_coef_min=0.1,
    dagger_coef_anneal_steps=60_000,
    loco_num_actions=wbc_obs.NUM_LOCO_ACTIONS,
    loco_blend_dims=15,
    loco_teacher_action_std_rescale=4.0,
  )
  return WbcDaggerRunnerCfg(
    seed=wbc_teacher_cfg.seed,
    num_steps_per_env=wbc_teacher_cfg.num_steps_per_env,
    max_iterations=wbc_teacher_cfg.max_iterations,
    obs_groups={
      "actor": ("actor_current", "actor_history"),
      "critic": (
        "critic_priv_future_sequence",
        "critic_current",
        "critic_extras",
      ),
    },
    save_interval=_WBC_SAVE_INTERVAL,
    experiment_name="g1_hand_moe_flat",
    run_name="g1_hand_moe_flat",
    logger=wbc_teacher_cfg.logger,
    wandb_project="wbc_mjlab",
    wandb_tags=wbc_teacher_cfg.wandb_tags,
    resume=wbc_teacher_cfg.resume,
    load_run=wbc_teacher_cfg.load_run,
    load_checkpoint=wbc_teacher_cfg.load_checkpoint,
    clip_actions=wbc_teacher_cfg.clip_actions,
    upload_model=wbc_teacher_cfg.upload_model,
    actor=actor,
    critic=critic,
    algorithm=algorithm,
    teacher_experiment_name="2026-04-07_09-10-33_g1_wbc_teacher_flat",
    teacher_proj_name="logs/rsl_rl/g1_wbc_teacher_flat",
    teacher_checkpoint=-1,
    loco_teacher_experiment_name="2026-04-06_17-56-59_g1_loco_teacher_flat",
    loco_teacher_proj_name="logs/rsl_rl/g1_loco_teacher_flat",
    loco_teacher_checkpoint=-1,
    wbc_teacher_actor=wbc_teacher_cfg.actor,
    wbc_teacher_obs_groups={
      "actor": ("wbc_teacher_actor_current", "wbc_teacher_actor_history"),
    },
    wbc_teacher_obs_set="actor",
    loco_teacher_actor=loco_teacher_cfg.actor,
    loco_teacher_obs_groups={"actor": ("loco_teacher_actor",)},
    loco_teacher_obs_set="actor",
    blend_obs_group="loco_blend",
    eval_student=False,
  )


def unitree_g1_hand_moe_flat_unicmd_nobv_runner_cfg() -> WbcDaggerRunnerCfg:
  """Runner config for MoE unicmd student paired with the nobv loco teacher."""
  cfg = unitree_g1_hand_moe_flat_runner_cfg()
  cfg.experiment_name = "g1_hand_moe_flat_unicmd_nobv"
  cfg.run_name = "g1_hand_moe_flat_unicmd_nobv"
  # Pair with the NoBV loco teacher (86-dim obs); inheriting from the MoE
  # base cfg would otherwise leave us pointed at the 89-dim base-vel teacher
  # and fail at checkpoint load time with an obs-dim mismatch.
  cfg.loco_teacher_experiment_name = "2026-04-14_23-25-21_g1_loco_teacher_flat_nobv"
  cfg.loco_teacher_proj_name = "logs/rsl_rl/g1_loco_teacher_flat_nobv"
  return cfg


def unitree_g1_hand_moe_flat_unicmd_nobv_amp_runner_cfg() -> WbcDaggerRunnerCfg:
  """3-teacher MoE student: WBC + NoBV loco + AMP recovery teacher.

  Diff vs ``unitree_g1_hand_moe_flat_unicmd_nobv_runner_cfg``:
    1. Actor grows to 3 experts (one per teacher).
    2. Subset-aware load balance: ``load_balance_coef=0.01`` now applies
       only to the WBC + loco experts on non-recovery samples (uniform
       target over the 2-expert subset, recovery expert excluded). On
       recovery samples a separate ``recovery_routing_coef`` directly
       pushes gate mass onto expert index 2. This preserves the WBC/loco
       blend on locomotion samples (free arm motion during walking)
       without contaminating the standing/walking distribution with the
       recovery expert.
    3. Wires the AMP teacher slot: experiment, proj, checkpoint, actor cfg
       (plain MLPModel), obs_groups (single ``amp_teacher_actor`` group).
    4. Sets ``amp_kl_coef=1.0 -> 0.5`` (annealed), deliberately stronger
       than ``dagger_coef`` since recovery envs zero out the task PPO
       reward and the AMP teacher KL is the dominant gradient source.
    5. Sets ``recovery_obs_group="recovery_active"`` so the runner pulls
       the per-env recovery mask out of the env's obs each step.
  """
  from wbc_mjlab.amp_config import unitree_g1_amp_teacher_flat_runner_cfg

  cfg = unitree_g1_hand_moe_flat_unicmd_nobv_runner_cfg()
  cfg.experiment_name = "g1_hand_moe_flat_unicmd_nobv_amp"
  cfg.run_name = "g1_hand_moe_flat_unicmd_nobv_amp"

  assert isinstance(cfg.actor, WbcMoEModelCfg), (
    "Expected MoE actor cfg from unitree_g1_hand_moe_flat_unicmd_nobv_runner_cfg."
  )
  cfg.actor.num_experts = 3

  # Recovery KL: stronger than body KL (dagger_coef 0.4 -> 0.2 in the seed
  # train script). Recovery envs zero velocity/hand task rewards, so the
  # AMP-teacher KL is essentially the only gradient pulling them upright.
  cfg.algorithm.amp_kl_coef = 1.0
  cfg.algorithm.amp_kl_coef_min = 0.5
  cfg.algorithm.amp_num_actions = 29
  if isinstance(cfg.algorithm, WbcMoEDaggerAlgorithmCfg):
    # Subset-aware load balance: apply only to (WBC, loco) experts on
    # non-recovery samples. Recovery expert is excluded from the uniform
    # target, so we can keep the original 0.01 coef without contaminating
    # the standing/walking distribution.
    cfg.algorithm.load_balance_coef = 0.01
    # Direct routing supervision on recovery samples: push gate weight onto
    # expert index 2 via (1 - gate[:, 2]).mean(). Strong enough to ensure
    # the gate cleanly hands off to the recovery expert, but not so large
    # that it dwarfs the AMP-KL gradient.
    cfg.algorithm.recovery_routing_coef = 0.5
    cfg.algorithm.recovery_expert_idx = 2

  # AMP teacher slot. Default points at the latest g1_amp_teacher_flat run
  # produced by ``train_amp_teacher_flat.sh`` (with payloads attached). Pin
  # via CLI ``--agent.amp-teacher-experiment-name <YYYY-MM-DD_HH-MM-SS_g1_amp_teacher_flat>``
  # for reproducibility.
  amp_teacher_cfg = unitree_g1_amp_teacher_flat_runner_cfg()
  cfg.amp_teacher_experiment_name = "2026-04-30_22-36-53_g1_amp_teacher_flat"
  cfg.amp_teacher_proj_name = "logs/rsl_rl/g1_amp_teacher_flat"
  cfg.amp_teacher_checkpoint = -1
  cfg.amp_teacher_actor = amp_teacher_cfg.actor
  cfg.amp_teacher_obs_groups = {"actor": ("amp_teacher_actor",)}
  cfg.amp_teacher_obs_set = "actor"
  cfg.recovery_obs_group = "recovery_active"
  return cfg


























