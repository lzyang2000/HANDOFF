"""Config helpers for the custom HANDOFF-style RL stack."""


from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class WbcTwistModelCfg(RslRlModelCfg):
  class_name: str = "wbc_mjlab.rl.models:ActorCriticFuture"
  current_mimic_dim: int = 0
  current_proprio_dim: int = 0
  history_feature_dim: int = 0
  history_length: int = 0
  privileged_future_step_dim: int = 0
  privileged_future_steps: int = 0
  critic_current_dim: int = 0
  critic_extras_dim: int = 0
  num_motion_observations: int = 0
  num_motion_steps: int = 1
  num_priop_observations: int = 0
  num_history_steps: int = 0
  num_future_observations: int = 0
  num_future_steps: int = 0
  motion_latent_dim: int = 64
  history_latent_dim: int = 64
  future_latent_dim: int = 64
  future_encoder_dims: tuple[int, ...] = (256, 128)
  future_dropout: float = 0.1
  layer_norm: bool = False
  num_depth_observations: int = 0
  depth_shape: tuple[int, ...] = (1, 18, 32)
  depth_latent_dim: int = 64
  depth_encoder_channels: tuple[int, int, int] = (8, 16, 32)


@dataclass
class WbcDaggerAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "wbc_mjlab.rl.algorithms:DaggerPPO"
  dagger_coef: float = 0.0
  dagger_coef_min: float = 0.0
  dagger_coef_anneal_steps: int = 0
  residual_reg_coef: float = 0.0
  loco_num_actions: int = 15
  loco_blend_dims: int = 15
  loco_teacher_action_std_rescale: float = 1.0
  arm_kl_coef: float | None = None
  arm_kl_coef_min: float | None = None
  amp_kl_coef: float | None = None
  amp_kl_coef_min: float | None = None
  amp_num_actions: int = 29


@dataclass
class WbcTeacherSlotCfg:
  """Config-side description of one ``TeacherSlot``.

  The runner consumes a ``list[WbcTeacherSlotCfg]`` and builds runtime
  ``TeacherSlot`` instances by loading actor checkpoints and resolving obs
  groups against the live env obs. See ``wbc_mjlab.rl.teacher_slot`` for the
  runtime counterpart.

  ``dim_slice`` is stored as a ``(start, stop)`` tuple here (rather than a
  ``slice``) so the cfg stays YAML-serializable. The runner converts it to a
  ``slice`` when constructing the ``TeacherSlot``.

  ``body_blend_role`` opts the slot into the body-blend KL block (anchor /
  flat_specialist / stair_specialist) — see ``TeacherSlot`` docstring.
  """

  name: str
  expert_idx: int | None = None
  dim_slice: tuple[int | None, int | None] | None = None
  routing_signal_key: str = ""
  blend_obs_group: str | None = None
  kl_coef_init: float = 0.0
  kl_coef_min: float = 0.0
  pin_routing_coef: float = 0.0
  anti_route_coef: float = 0.0
  output_dim: int = 0
  is_arm_kl: bool = False
  body_blend_role: str | None = None
  experiment_name: str | None = None
  proj_name: str | None = None
  checkpoint: str | int | None = None
  actor_cfg: dict[str, Any] | None = None
  obs_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
  obs_set: str = "actor"


@dataclass
class WbcDaggerRunnerCfg(RslRlOnPolicyRunnerCfg):
  class_name: str = "wbc_mjlab.rl.runner:DaggerOnPolicyRunner"
  actor: WbcTwistModelCfg = field(
    default_factory=lambda: WbcTwistModelCfg(
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      }
    )
  )
  critic: WbcTwistModelCfg = field(default_factory=WbcTwistModelCfg)
  algorithm: WbcDaggerAlgorithmCfg = field(default_factory=WbcDaggerAlgorithmCfg)
  teacher_experiment_name: str | None = None
  teacher_proj_name: str | None = None
  teacher_checkpoint: str | int | None = None
  loco_teacher_experiment_name: str | None = None
  loco_teacher_proj_name: str | None = None
  loco_teacher_checkpoint: str | int | None = None
  wbc_teacher_actor: dict[str, Any] | None = None
  wbc_teacher_obs_groups: dict[str, tuple[str, ...]] | None = None
  wbc_teacher_obs_set: str | None = None
  loco_teacher_actor: dict[str, Any] | None = None
  loco_teacher_obs_groups: dict[str, tuple[str, ...]] | None = None
  loco_teacher_obs_set: str | None = None
  amp_teacher_experiment_name: str | None = None
  amp_teacher_proj_name: str | None = None
  amp_teacher_checkpoint: str | int | None = None
  amp_teacher_actor: dict[str, Any] | None = None
  amp_teacher_obs_groups: dict[str, tuple[str, ...]] | None = None
  amp_teacher_obs_set: str | None = None
  # Stair-depth-CNN teacher slot (4-teacher MoE student). Top-level
  # fields mirror the wbc_/loco_/amp_ pattern so they can be pinned from
  # the call site via ``--agent.stair-teacher-experiment-name …`` etc.
  # The runner's ``_build_teacher_slot`` rebinds an explicit ``stair``
  # slot's exp/proj/ckpt from these post-CLI fields.
  stair_teacher_experiment_name: str | None = None
  stair_teacher_proj_name: str | None = None
  stair_teacher_checkpoint: str | int | None = None
  blend_obs_group: str | None = None
  recovery_obs_group: str | None = None
  # Generic teacher registry. Empty for legacy 3-teacher cfgs; the runner
  # calls ``resolve_slots()`` before consuming this list, which auto-
  # synthesizes entries from the legacy per-teacher fields if ``slots`` is
  # still empty at runner-init time. The new 4-teacher cfg populates
  # ``slots`` explicitly (auto-synthesis is then a no-op).
  slots: list[WbcTeacherSlotCfg] = field(default_factory=list)
  eval_student: bool = False

  def resolve_slots(self) -> list[WbcTeacherSlotCfg]:
    """Return ``slots`` if non-empty, else synthesize from legacy fields.

    Called by the runner just before constructing the runtime ``TeacherSlot``
    list, so it sees the final post-CLI-override cfg state. ``__post_init__``
    can't do this because rl_cfg.py functions chain field assignments AFTER
    construction (e.g. ``unitree_g1_hand_moe_flat_unicmd_nobv_amp_runner_cfg``
    sets ``amp_teacher_*`` on a cfg that was already constructed by the
    parent loco-pair cfg).

    Once invoked, ``self.slots`` is mutated to the synthesized list so that
    subsequent reads (e.g. by the algorithm cfg dump) see a consistent value.
    """
    if self.slots:
      return self.slots

    alg = self.algorithm
    synthesized: list[WbcTeacherSlotCfg] = []

    # WBC body slot — body-blend anchor, slice [0:15], dagger_coef.
    if self.teacher_experiment_name is not None:
      synthesized.append(
        WbcTeacherSlotCfg(
          name="wbc_body",
          dim_slice=(0, 15),
          kl_coef_init=alg.dagger_coef,
          kl_coef_min=alg.dagger_coef_min,
          body_blend_role="anchor",
          experiment_name=self.teacher_experiment_name,
          proj_name=self.teacher_proj_name,
          checkpoint=self.teacher_checkpoint,
          actor_cfg=self.wbc_teacher_actor,
          obs_groups=self.wbc_teacher_obs_groups or {},
          obs_set=self.wbc_teacher_obs_set or "actor",
          output_dim=29,
        )
      )
      # WBC arm slot — separate arm-KL term, slice [15:29], arm_kl_coef.
      # Reuses the same WBC checkpoint (runner dedupes by (exp, proj, ckpt)).
      if alg.arm_kl_coef is not None:
        synthesized.append(
          WbcTeacherSlotCfg(
            name="wbc_arm",
            dim_slice=(15, 29),
            kl_coef_init=float(alg.arm_kl_coef),
            kl_coef_min=float(alg.arm_kl_coef_min)
            if alg.arm_kl_coef_min is not None
            else float(alg.arm_kl_coef),
            is_arm_kl=True,
            experiment_name=self.teacher_experiment_name,
            proj_name=self.teacher_proj_name,
            checkpoint=self.teacher_checkpoint,
            actor_cfg=self.wbc_teacher_actor,
            obs_groups=self.wbc_teacher_obs_groups or {},
            obs_set=self.wbc_teacher_obs_set or "actor",
            output_dim=29,
          )
        )

    # Loco slot — body-blend flat-specialist, slice [0:15], dagger_coef,
    # blended via ``blend_obs_group`` (typically ``"loco_blend"``).
    if self.loco_teacher_experiment_name is not None:
      synthesized.append(
        WbcTeacherSlotCfg(
          name="loco",
          dim_slice=(0, alg.loco_num_actions),
          kl_coef_init=alg.dagger_coef,
          kl_coef_min=alg.dagger_coef_min,
          body_blend_role="flat_specialist",
          blend_obs_group=self.blend_obs_group,
          experiment_name=self.loco_teacher_experiment_name,
          proj_name=self.loco_teacher_proj_name,
          checkpoint=self.loco_teacher_checkpoint,
          actor_cfg=self.loco_teacher_actor,
          obs_groups=self.loco_teacher_obs_groups or {},
          obs_set=self.loco_teacher_obs_set or "actor",
          output_dim=alg.loco_num_actions,
        )
      )

    # AMP slot — pinned to a recovery expert, slice [0:29], amp_kl_coef.
    if self.amp_teacher_experiment_name is not None and alg.amp_kl_coef is not None:
      # ``recovery_expert_idx`` lives on WbcMoEDaggerAlgorithmCfg. Read
      # defensively so legacy non-MoE algorithm cfgs don't break.
      rec_idx = getattr(alg, "recovery_expert_idx", -1)
      pin_coef = float(getattr(alg, "recovery_routing_coef", 0.0))
      synthesized.append(
        WbcTeacherSlotCfg(
          name="amp",
          expert_idx=int(rec_idx) if rec_idx is not None else None,
          dim_slice=(0, alg.amp_num_actions),
          routing_signal_key=self.recovery_obs_group or "recovery_active",
          kl_coef_init=float(alg.amp_kl_coef),
          kl_coef_min=float(alg.amp_kl_coef_min)
          if alg.amp_kl_coef_min is not None
          else float(alg.amp_kl_coef),
          pin_routing_coef=pin_coef,
          experiment_name=self.amp_teacher_experiment_name,
          proj_name=self.amp_teacher_proj_name,
          checkpoint=self.amp_teacher_checkpoint,
          actor_cfg=self.amp_teacher_actor,
          obs_groups=self.amp_teacher_obs_groups or {},
          obs_set=self.amp_teacher_obs_set or "actor",
          output_dim=alg.amp_num_actions,
        )
      )

    self.slots = synthesized
    return self.slots


@dataclass
class WbcMoEModelCfg(WbcTwistModelCfg):
  class_name: str = "wbc_mjlab.rl.models:ActorCriticFutureMoE"
  num_experts: int = 2
  expert_hidden_dims: tuple[int, ...] = (256, 128)
  gate_hidden_dims: tuple[int, ...] = (128, 64)


@dataclass
class WbcMoEDaggerAlgorithmCfg(WbcDaggerAlgorithmCfg):
  class_name: str = "wbc_mjlab.rl.algorithms:MoEDaggerPPO"
  load_balance_coef: float = 0.01
  recovery_routing_coef: float = 0.0
  recovery_expert_idx: int = -1
