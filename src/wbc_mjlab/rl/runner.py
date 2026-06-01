"""Custom runners for PKL motion tracking and HANDOFF-style DAgger PPO."""

from __future__ import annotations

import copy
import os
import time
from typing import cast

import torch
import wandb
from rsl_rl.env.vec_env import VecEnv
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict
from torch import nn

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner

from wbc_mjlab.commands import PklMotionCommand
from wbc_mjlab.rl.algorithms import DaggerPPO
from wbc_mjlab.rl.teacher_slot import TeacherSlot


def _select_obs_groups(
  obs: TensorDict,
  obs_groups: dict[str, list[str]] | dict[str, tuple[str, ...]],
  obs_set: str,
) -> TensorDict:
  groups = tuple(obs_groups[obs_set])
  return TensorDict(
    {group: obs[group] for group in groups},
    batch_size=obs.batch_size,
    device=obs.device,
  )


def _select_single_obs_group(
  obs: TensorDict,
  obs_groups: dict[str, list[str]] | dict[str, tuple[str, ...]],
  obs_set: str,
) -> torch.Tensor:
  selected = _select_obs_groups(obs, obs_groups, obs_set)
  groups = tuple(selected.keys())
  if len(groups) != 1:
    raise ValueError(
      f"Expected obs_set '{obs_set}' to resolve to exactly one group, got {groups}."
    )
  return selected[groups[0]]


class _OnnxMotionModel(nn.Module):
  """ONNX-exportable model that wraps the policy and bundles motion data."""

  def __init__(self, actor: nn.Module, command: PklMotionCommand) -> None:
    super().__init__()
    self.policy = actor.as_onnx(verbose=False)

    lib = command.motion_lib
    num_frames = lib._motion_num_frames[0].item()
    start = lib._motion_start_idx[0].item()
    end = start + num_frames

    self.register_buffer("joint_pos", lib._all_joint_pos[start:end].cpu())
    self.register_buffer("joint_vel", lib._all_joint_vel[start:end].cpu())
    self.register_buffer("body_pos_w", lib._all_body_pos_w[start:end].cpu())
    self.register_buffer("body_quat_w", lib._all_body_quat_w[start:end].cpu())
    self.register_buffer(
      "body_lin_vel_w", lib._all_body_lin_vel_w[start:end].cpu()
    )
    self.register_buffer(
      "body_ang_vel_w", lib._all_body_ang_vel_w[start:end].cpu()
    )
    self.time_step_total: int = num_frames

  def forward(
    self, x: torch.Tensor, time_step: torch.Tensor
  ) -> tuple[torch.Tensor, ...]:
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=self.time_step_total - 1
    )
    return (
      self.policy(x),
      self.joint_pos[time_step_clamped],
      self.joint_vel[time_step_clamped],
      self.body_pos_w[time_step_clamped],
      self.body_quat_w[time_step_clamped],
      self.body_lin_vel_w[time_step_clamped],
      self.body_ang_vel_w[time_step_clamped],
    )


class PklMotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  """Runner that bundles motion library data into the ONNX export.

  .. deprecated::
      Only used by the base ``Wbc-Tracking-Flat`` task. All other tasks
      (WBC Teacher, Loco Teacher, Dual-Student, Dual-Student-Force) now
      use ``DaggerOnPolicyRunner`` with plain actor ONNX export.
      Pending removal after testing confirms DaggerOnPolicyRunner works
      as a full replacement.
  """

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    os.makedirs(path, exist_ok=True)
    cmd = cast(
      PklMotionCommand, self.env.unwrapped.command_manager.get_term("motion")
    )
    model = _OnnxMotionModel(self.alg.get_policy(), cmd)
    model.to("cpu")
    model.eval()
    obs = torch.zeros(1, model.policy.input_size)
    time_step = torch.zeros(1, 1)
    torch.onnx.export(
      model,
      (obs, time_step),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=["obs", "time_step"],
      output_names=[
        "actions",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      ],
      dynamic_axes={},
      dynamo=False,
    )

  def save(self, path: str, infos=None) -> None:
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name
        if self.logger.logger_type == "wandb" and wandb.run
        else "local"
      )
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      motion_term = cast(
        PklMotionCommand, self.env.unwrapped.command_manager.get_term("motion")
      )
      metadata.update(
        {
          "anchor_body_name": motion_term.cfg.anchor_body_name,
          "body_names": list(motion_term.cfg.body_names),
        }
      )
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
        if self.registry_name is not None:
          wandb.run.use_artifact(self.registry_name)
          self.registry_name = None
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")


class _TeacherVisPolicy:
  """Wraps the student policy to also run teacher models and store their
  joint-position targets on the PklMotionCommand for ghost visualization."""

  def __init__(
    self,
    student_policy: nn.Module,
    wbc_teacher: nn.Module | None,
    loco_teacher: nn.Module | None,
    env: RslRlVecEnvWrapper,
    wbc_obs_groups: dict | None,
    wbc_obs_set: str | None,
    loco_obs_groups: dict | None,
    loco_obs_set: str | None,
    action_scale: torch.Tensor,
    action_offset: torch.Tensor,
    loco_num_actions: int,
    device: str,
  ) -> None:
    self.student_policy = student_policy
    self.wbc_teacher = wbc_teacher
    self.loco_teacher = loco_teacher
    self.env = env
    self.wbc_obs_groups = wbc_obs_groups
    self.wbc_obs_set = wbc_obs_set
    self.loco_obs_groups = loco_obs_groups
    self.loco_obs_set = loco_obs_set
    self.action_scale = action_scale
    self.action_offset = action_offset
    self.loco_num_actions = loco_num_actions
    self.device = device

    # Resolve the motion command term once
    try:
      self._motion_cmd: PklMotionCommand | None = cast(
        PklMotionCommand, env.unwrapped.command_manager.get_term("motion")
      )
    except Exception:
      self._motion_cmd = None

  def reset(self) -> None:
    reset_fn = getattr(self.student_policy, "reset", None)
    if reset_fn is not None:
      reset_fn()

  def __call__(self, obs: TensorDict) -> torch.Tensor:
    actions = self.student_policy(obs)
    if self._motion_cmd is None:
      return actions

    with torch.inference_mode():
      # WBC teacher ghost
      if (
        self.wbc_teacher is not None
        and self.wbc_obs_groups is not None
        and self.wbc_obs_set is not None
      ):
        wbc_obs = _select_obs_groups(obs, self.wbc_obs_groups, self.wbc_obs_set)
        wbc_mean = self.wbc_teacher(wbc_obs, stochastic_output=False)
        wbc_targets = self.action_offset + wbc_mean * self.action_scale
        self._motion_cmd._wbc_teacher_joint_targets = wbc_targets

      # Loco teacher ghost
      if (
        self.loco_teacher is not None
        and self.loco_obs_groups is not None
        and self.loco_obs_set is not None
      ):
        loco_obs = _select_single_obs_group(obs, self.loco_obs_groups, self.loco_obs_set)
        loco_mean = self.loco_teacher(loco_obs, stochastic_output=False)
        loco_targets = self.action_offset.clone()
        n = self.loco_num_actions
        loco_targets[:, :n] = self.action_offset[:, :n] + loco_mean * self.action_scale[:, :n]
        self._motion_cmd._loco_teacher_joint_targets = loco_targets

    return actions


class DaggerOnPolicyRunner(MjlabOnPolicyRunner):
  """Runner that keeps base OnPolicy behavior but optionally loads frozen teachers."""

  alg: DaggerPPO

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name
    self._clip_actions = train_cfg.get("clip_actions", None)
    full_obs = self.env.get_observations().to(self.device)
    self._critic_obs_groups = train_cfg["obs_groups"]
    self._wbc_teacher_obs_groups = (
      train_cfg.get("wbc_teacher_obs_groups")
      or train_cfg.get("teacher_obs_groups")
    )
    self._wbc_teacher_obs_set = (
      train_cfg.get("wbc_teacher_obs_set")
      or train_cfg.get("teacher_obs_set")
    )
    self._loco_teacher_obs_groups = (
      train_cfg.get("loco_teacher_obs_groups")
      or train_cfg.get("loco_obs_groups")
    )
    self._loco_teacher_obs_set = (
      train_cfg.get("loco_teacher_obs_set")
      or train_cfg.get("loco_obs_set")
    )
    self._blend_obs_group = train_cfg.get("blend_obs_group")
    self._recovery_obs_group = train_cfg.get("recovery_obs_group")
    self._amp_teacher_obs_groups = train_cfg.get("amp_teacher_obs_groups")
    self._amp_teacher_obs_set = train_cfg.get("amp_teacher_obs_set")
    self._teacher_actor = self._load_optional_teacher(
      teacher_label="WBC",
      experiment_name=train_cfg.get("teacher_experiment_name"),
      proj_name=train_cfg.get("teacher_proj_name"),
      checkpoint=train_cfg.get("teacher_checkpoint"),
      actor_cfg=(
        train_cfg.get("wbc_teacher_actor")
        or train_cfg.get("teacher_actor")
        or train_cfg["actor"]
      ),
      obs_groups=self._wbc_teacher_obs_groups or train_cfg["obs_groups"],
      obs_set=self._wbc_teacher_obs_set or "actor",
      output_dim=env.num_actions,
      obs_example=(
        _select_obs_groups(full_obs, self._wbc_teacher_obs_groups, self._wbc_teacher_obs_set)
        if self._wbc_teacher_obs_groups is not None and self._wbc_teacher_obs_set is not None
        else full_obs
      ),
    )
    self._loco_teacher_actor = self._load_optional_teacher(
      teacher_label="Loco",
      experiment_name=train_cfg.get("loco_teacher_experiment_name"),
      proj_name=train_cfg.get("loco_teacher_proj_name"),
      checkpoint=train_cfg.get("loco_teacher_checkpoint"),
      actor_cfg=(
        train_cfg.get("loco_teacher_actor")
        or train_cfg.get("loco_actor")
        or train_cfg["actor"]
      ),
      obs_groups=self._loco_teacher_obs_groups or train_cfg["obs_groups"],
      obs_set=self._loco_teacher_obs_set or "actor",
      output_dim=train_cfg["algorithm"].get("loco_num_actions", env.num_actions),
      obs_example=(
        _select_obs_groups(full_obs, self._loco_teacher_obs_groups, self._loco_teacher_obs_set)
        if self._loco_teacher_obs_groups is not None and self._loco_teacher_obs_set is not None
        else None
      ),
    )
    self._amp_teacher_actor = self._load_optional_teacher(
      teacher_label="AMP",
      experiment_name=train_cfg.get("amp_teacher_experiment_name"),
      proj_name=train_cfg.get("amp_teacher_proj_name"),
      checkpoint=train_cfg.get("amp_teacher_checkpoint"),
      actor_cfg=(
        train_cfg.get("amp_teacher_actor")
        or train_cfg["actor"]
      ),
      obs_groups=self._amp_teacher_obs_groups or train_cfg["obs_groups"],
      obs_set=self._amp_teacher_obs_set or "actor",
      output_dim=train_cfg["algorithm"].get("amp_num_actions", env.num_actions),
      obs_example=(
        _select_obs_groups(full_obs, self._amp_teacher_obs_groups, self._amp_teacher_obs_set)
        if self._amp_teacher_obs_groups is not None and self._amp_teacher_obs_set is not None
        else None
      ),
    )
    # ---- Slot-driven teacher registry. ----
    # Build a generic ``list[TeacherSlot]`` from either the explicit
    # ``train_cfg["slots"]`` (the new 4-teacher cfg path) or, for back-compat
    # with rl_cfg.py functions that only set the legacy ``*_experiment_name``
    # fields, synthesize the equivalent slot list from those fields. The
    # legacy actors loaded above are reused (no double-load) when their
    # (exp, proj, ckpt) triple matches a slot's; new slots (e.g. the stair
    # specialist) get loaded fresh.
    slot_cfgs = train_cfg.get("slots") or self._synthesize_legacy_slot_cfgs(train_cfg)
    legacy_actor_index: dict[tuple[str, str, str | int], nn.Module] = {}

    def _legacy_key(exp: str | None, proj: str | None, ckpt: str | int | None):
      if exp is None or proj is None:
        return None
      return (str(exp), str(proj), str(ckpt))

    for legacy_actor, legacy_keys in (
      (self._teacher_actor, _legacy_key(
        train_cfg.get("teacher_experiment_name"),
        train_cfg.get("teacher_proj_name"),
        train_cfg.get("teacher_checkpoint"),
      )),
      (self._loco_teacher_actor, _legacy_key(
        train_cfg.get("loco_teacher_experiment_name"),
        train_cfg.get("loco_teacher_proj_name"),
        train_cfg.get("loco_teacher_checkpoint"),
      )),
      (self._amp_teacher_actor, _legacy_key(
        train_cfg.get("amp_teacher_experiment_name"),
        train_cfg.get("amp_teacher_proj_name"),
        train_cfg.get("amp_teacher_checkpoint"),
      )),
    ):
      if legacy_actor is not None and legacy_keys is not None:
        legacy_actor_index[legacy_keys] = legacy_actor

    self._slots: list[TeacherSlot] = []
    for s_cfg in slot_cfgs:
      slot = self._build_teacher_slot(s_cfg, train_cfg, full_obs, legacy_actor_index)
      if slot is not None:
        self._slots.append(slot)

    if self._slots:
      # Allocate per-slot rollout buffers in the storage so add_transition /
      # mini_batch_generator can flow slot obs and routing weights through.
      for slot in self._slots:
        obs_td = _select_obs_groups(full_obs, slot.obs_groups, slot.obs_set)
        slot_obs_example: torch.Tensor | TensorDict
        if slot.is_structured:
          slot_obs_example = obs_td
        else:
          # Single-group teacher → flat tensor.
          slot_obs_example = next(iter(obs_td.values()))
        self.alg.storage.init_slot_obs_buffer(slot.name, slot_obs_example)
        self.alg.storage.init_slot_routing_buffer(slot.name)
      # Switch the algorithm to slot-aware mode.
      self.alg.set_teacher_models(slots=self._slots)
    else:
      # Empty slot list (no legacy fields set, no explicit slots) →
      # fall back to legacy three-arg form. Storage stays slot-empty.
      self.alg.set_teacher_models(
        self._teacher_actor,
        self._loco_teacher_actor,
        amp_teacher_actor=self._amp_teacher_actor,
      )

  def _synthesize_legacy_slot_cfgs(self, train_cfg: dict) -> list[dict]:
    """Build ``slot_cfgs`` (list of dicts) from legacy per-teacher fields.

    Mirrors ``WbcDaggerRunnerCfg.resolve_slots`` but operates on a dict
    (the form ``train_cfg`` arrives in after dataclass ``asdict``). Used
    when ``train_cfg["slots"]`` is missing/empty so existing rl_cfg.py
    functions that set the legacy fields get auto-migrated at runner init
    time.
    """
    alg = train_cfg.get("algorithm", {})
    slot_cfgs: list[dict] = []

    if train_cfg.get("teacher_experiment_name") is not None:
      slot_cfgs.append({
        "name": "wbc_body",
        "expert_idx": None,
        "dim_slice": (0, 15),
        "kl_coef_init": float(alg.get("dagger_coef", 0.0)),
        "kl_coef_min": float(alg.get("dagger_coef_min", 0.0)),
        "body_blend_role": "anchor",
        "experiment_name": train_cfg.get("teacher_experiment_name"),
        "proj_name": train_cfg.get("teacher_proj_name"),
        "checkpoint": train_cfg.get("teacher_checkpoint"),
        "actor_cfg": train_cfg.get("wbc_teacher_actor"),
        "obs_groups": train_cfg.get("wbc_teacher_obs_groups") or {},
        "obs_set": train_cfg.get("wbc_teacher_obs_set") or "actor",
        "output_dim": 29,
      })
      if alg.get("arm_kl_coef") is not None:
        slot_cfgs.append({
          "name": "wbc_arm",
          "expert_idx": None,
          "dim_slice": (15, 29),
          "kl_coef_init": float(alg["arm_kl_coef"]),
          "kl_coef_min": float(alg.get("arm_kl_coef_min") or alg["arm_kl_coef"]),
          "is_arm_kl": True,
          "experiment_name": train_cfg.get("teacher_experiment_name"),
          "proj_name": train_cfg.get("teacher_proj_name"),
          "checkpoint": train_cfg.get("teacher_checkpoint"),
          "actor_cfg": train_cfg.get("wbc_teacher_actor"),
          "obs_groups": train_cfg.get("wbc_teacher_obs_groups") or {},
          "obs_set": train_cfg.get("wbc_teacher_obs_set") or "actor",
          "output_dim": 29,
        })

    if train_cfg.get("loco_teacher_experiment_name") is not None:
      slot_cfgs.append({
        "name": "loco",
        "expert_idx": None,
        "dim_slice": (0, alg.get("loco_num_actions", 15)),
        "kl_coef_init": float(alg.get("dagger_coef", 0.0)),
        "kl_coef_min": float(alg.get("dagger_coef_min", 0.0)),
        "body_blend_role": "flat_specialist",
        "blend_obs_group": train_cfg.get("blend_obs_group"),
        "experiment_name": train_cfg.get("loco_teacher_experiment_name"),
        "proj_name": train_cfg.get("loco_teacher_proj_name"),
        "checkpoint": train_cfg.get("loco_teacher_checkpoint"),
        "actor_cfg": train_cfg.get("loco_teacher_actor"),
        "obs_groups": train_cfg.get("loco_teacher_obs_groups") or {},
        "obs_set": train_cfg.get("loco_teacher_obs_set") or "actor",
        "output_dim": alg.get("loco_num_actions", 15),
      })

    if (
      train_cfg.get("amp_teacher_experiment_name") is not None
      and alg.get("amp_kl_coef") is not None
    ):
      rec_idx = alg.get("recovery_expert_idx")
      pin_coef = float(alg.get("recovery_routing_coef", 0.0))
      slot_cfgs.append({
        "name": "amp",
        "expert_idx": int(rec_idx) if rec_idx is not None else None,
        "dim_slice": (0, alg.get("amp_num_actions", 29)),
        "routing_signal_key": train_cfg.get("recovery_obs_group") or "recovery_active",
        "kl_coef_init": float(alg["amp_kl_coef"]),
        "kl_coef_min": float(alg.get("amp_kl_coef_min") or alg["amp_kl_coef"]),
        "pin_routing_coef": pin_coef,
        "experiment_name": train_cfg.get("amp_teacher_experiment_name"),
        "proj_name": train_cfg.get("amp_teacher_proj_name"),
        "checkpoint": train_cfg.get("amp_teacher_checkpoint"),
        "actor_cfg": train_cfg.get("amp_teacher_actor"),
        "obs_groups": train_cfg.get("amp_teacher_obs_groups") or {},
        "obs_set": train_cfg.get("amp_teacher_obs_set") or "actor",
        "output_dim": alg.get("amp_num_actions", 29),
      })

    return slot_cfgs

  def _build_teacher_slot(
    self,
    s_cfg: dict,
    train_cfg: dict,
    full_obs: TensorDict,
    legacy_actor_index: dict[tuple[str, str, str | int], nn.Module],
  ) -> TeacherSlot | None:
    """Build a runtime ``TeacherSlot`` from a slot cfg dict.

    Resolves the slot's obs groups against the live env obs, dedups the
    actor against already-loaded legacy teachers, and returns the slot.
    Returns ``None`` if the slot has no checkpoint (e.g. an explicit
    slot cfg with ``experiment_name=None``).
    """
    name = s_cfg["name"]
    obs_groups = s_cfg.get("obs_groups") or {}
    obs_set = s_cfg.get("obs_set") or "actor"

    if not obs_groups:
      # Fall back to the actor obs_groups (used by legacy WBC teacher slots).
      obs_groups = train_cfg["obs_groups"]

    obs_td_example = _select_obs_groups(full_obs, obs_groups, obs_set)
    is_structured = len(obs_td_example.keys()) > 1

    # Re-bind exp/proj/ckpt from the post-CLI ``train_cfg`` legacy fields
    # for slots whose source is a known legacy teacher (wbc_body / wbc_arm
    # → teacher_*; loco → loco_teacher_*; amp → amp_teacher_*). Explicit
    # cfgs built at module-import time capture rl_cfg defaults; this pulls
    # in CLI overrides like ``--agent.teacher-experiment-name``.
    exp = s_cfg.get("experiment_name")
    proj = s_cfg.get("proj_name")
    ckpt = s_cfg.get("checkpoint")
    if name in ("wbc_body", "wbc_arm"):
      exp = train_cfg.get("teacher_experiment_name") or exp
      proj = train_cfg.get("teacher_proj_name") or proj
      ckpt = train_cfg.get("teacher_checkpoint") if train_cfg.get("teacher_checkpoint") is not None else ckpt
    elif name == "loco":
      exp = train_cfg.get("loco_teacher_experiment_name") or exp
      proj = train_cfg.get("loco_teacher_proj_name") or proj
      ckpt = train_cfg.get("loco_teacher_checkpoint") if train_cfg.get("loco_teacher_checkpoint") is not None else ckpt
    elif name == "amp":
      exp = train_cfg.get("amp_teacher_experiment_name") or exp
      proj = train_cfg.get("amp_teacher_proj_name") or proj
      ckpt = train_cfg.get("amp_teacher_checkpoint") if train_cfg.get("amp_teacher_checkpoint") is not None else ckpt
    elif name == "stair":
      exp = train_cfg.get("stair_teacher_experiment_name") or exp
      proj = train_cfg.get("stair_teacher_proj_name") or proj
      ckpt = train_cfg.get("stair_teacher_checkpoint") if train_cfg.get("stair_teacher_checkpoint") is not None else ckpt
    legacy_key = (
      (str(exp), str(proj), str(ckpt))
      if (exp is not None and proj is not None)
      else None
    )

    actor: nn.Module | None
    if legacy_key is not None and legacy_key in legacy_actor_index:
      actor = legacy_actor_index[legacy_key]
    else:
      actor = self._load_optional_teacher(
        teacher_label=f"Slot[{name}]",
        experiment_name=exp,
        proj_name=proj,
        checkpoint=ckpt,
        actor_cfg=s_cfg.get("actor_cfg") or train_cfg["actor"],
        obs_groups=obs_groups,
        obs_set=obs_set,
        output_dim=s_cfg.get("output_dim") or train_cfg["actor"].get("num_actions", 29),
        obs_example=obs_td_example,
      )
      if actor is not None and legacy_key is not None:
        legacy_actor_index[legacy_key] = actor

    if actor is None:
      return None

    # Convert dim_slice tuple -> slice. None entries default to "open" slice.
    dim_slice_t = s_cfg.get("dim_slice")
    if dim_slice_t is None:
      dim_slice = slice(None, None)
    else:
      dim_slice = slice(dim_slice_t[0], dim_slice_t[1])

    # Re-bind kl_coef_init / kl_coef_min from the post-CLI ``train_cfg``
    # algorithm dict for slots whose KL strength is owned by an algorithm
    # cfg field (body_blend / arm / amp). Explicit cfgs built at module
    # import time capture the rl_cfg defaults (e.g. dagger_coef=0.2);
    # this re-binds to the actual CLI-applied values
    # (e.g. dagger_coef=0.4 from the seed train script).
    alg_cfg = train_cfg.get("algorithm", {})
    kl_coef_init = float(s_cfg.get("kl_coef_init", 0.0))
    kl_coef_min = float(s_cfg.get("kl_coef_min", 0.0))
    body_blend_role = s_cfg.get("body_blend_role")
    is_arm_kl = bool(s_cfg.get("is_arm_kl", False))
    expert_idx = s_cfg.get("expert_idx")
    if body_blend_role in ("anchor", "flat_specialist", "stair_specialist"):
      # Body-blend slots all share dagger_coef (body block multiplies the
      # blended linear combo by self.dagger_coef once).
      kl_coef_init = float(alg_cfg.get("dagger_coef", kl_coef_init))
      kl_coef_min = float(alg_cfg.get("dagger_coef_min", kl_coef_min))
    elif is_arm_kl:
      kl_coef_init = float(
        alg_cfg.get("arm_kl_coef") if alg_cfg.get("arm_kl_coef") is not None else kl_coef_init
      )
      kl_coef_min = float(
        alg_cfg.get("arm_kl_coef_min")
        if alg_cfg.get("arm_kl_coef_min") is not None
        else kl_coef_init
      )
    elif (
      expert_idx is not None
      and alg_cfg.get("recovery_expert_idx") is not None
      and expert_idx == alg_cfg["recovery_expert_idx"]
    ):
      kl_coef_init = float(
        alg_cfg.get("amp_kl_coef") if alg_cfg.get("amp_kl_coef") is not None else kl_coef_init
      )
      kl_coef_min = float(
        alg_cfg.get("amp_kl_coef_min")
        if alg_cfg.get("amp_kl_coef_min") is not None
        else kl_coef_init
      )

    return TeacherSlot(
      name=name,
      actor=actor,
      actor_cfg=s_cfg.get("actor_cfg") or {},
      obs_groups=obs_groups,
      obs_set=obs_set,
      experiment_name=exp,
      proj_name=proj,
      checkpoint=ckpt,
      expert_idx=expert_idx,
      dim_slice=dim_slice,
      kl_coef_init=kl_coef_init,
      kl_coef=kl_coef_init,
      kl_coef_min=kl_coef_min,
      routing_signal_key=s_cfg.get("routing_signal_key") or "",
      pin_routing_coef=float(s_cfg.get("pin_routing_coef", 0.0)),
      anti_route_coef=float(s_cfg.get("anti_route_coef", 0.0)),
      output_dim=int(s_cfg.get("output_dim", 0)),
      is_structured=is_structured,
      blend_obs_group=s_cfg.get("blend_obs_group"),
      body_blend_role=body_blend_role,
      is_arm_kl=is_arm_kl,
    )

  def save(self, path: str, infos=None) -> None:
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name
        if self.logger.logger_type == "wandb" and wandb.run
        else "local"
      )
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")

  def get_inference_policy(self, device: str | None = None) -> nn.Module:
    """Return inference policy wrapped with teacher ghost visualization."""
    student_policy = super().get_inference_policy(device)
    if self._teacher_actor is None and self._loco_teacher_actor is None:
      return student_policy

    # Read action scale/offset from the env's action term
    action_term = self.env.unwrapped.action_manager.get_term("joint_pos")
    scale = action_term.scale
    offset = action_term.offset
    if not isinstance(scale, torch.Tensor):
      scale = torch.full(
        (self.env.num_envs, self.env.num_actions),
        float(scale),
        device=self.device,
      )
    if not isinstance(offset, torch.Tensor):
      offset = torch.full(
        (self.env.num_envs, self.env.num_actions),
        float(offset),
        device=self.device,
      )
    target_device = device or self.device
    return _TeacherVisPolicy(
      student_policy=student_policy,
      wbc_teacher=self._teacher_actor,
      loco_teacher=self._loco_teacher_actor,
      env=self.env,
      wbc_obs_groups=self._wbc_teacher_obs_groups,
      wbc_obs_set=self._wbc_teacher_obs_set,
      loco_obs_groups=self._loco_teacher_obs_groups,
      loco_obs_set=self._loco_teacher_obs_set,
      action_scale=scale.to(target_device),
      action_offset=offset.to(target_device),
      loco_num_actions=self.alg.loco_num_actions,
      device=target_device,
    )

  def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    if self.is_distributed:
      self.alg.broadcast_parameters()
    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      rollout_start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          critic_obs = _select_obs_groups(obs, self._critic_obs_groups, "critic")
          teacher_obs = (
            _select_obs_groups(
              obs, self._wbc_teacher_obs_groups, self._wbc_teacher_obs_set
            )
            if self._teacher_actor is not None
            and self._wbc_teacher_obs_groups is not None
            and self._wbc_teacher_obs_set is not None
            else None
          )
          # Legacy single-group plumbing for the loco teacher's flat
          # ``loco_observations`` storage buffer. When the loco obs is
          # multi-group (depth-aware stair-plainvel teacher with
          # ``("loco_teacher_actor_flat", "actor_depth")``), this is a
          # no-op — the slot mechanism's ``slot_obs["loco"]`` carries
          # the TensorDict-shaped obs instead.
          loco_obs = None
          if (
            self._loco_teacher_actor is not None
            and self._loco_teacher_obs_groups is not None
            and self._loco_teacher_obs_set is not None
          ):
            _loco_td = _select_obs_groups(
              obs, self._loco_teacher_obs_groups, self._loco_teacher_obs_set
            )
            if len(_loco_td.keys()) == 1:
              loco_obs = next(iter(_loco_td.values()))
          blend_weights = (
            obs[self._blend_obs_group]
            if self._blend_obs_group is not None and self._blend_obs_group in obs.keys()
            else None
          )
          amp_obs = (
            _select_single_obs_group(
              obs, self._amp_teacher_obs_groups, self._amp_teacher_obs_set
            )
            if self._amp_teacher_actor is not None
            and self._amp_teacher_obs_groups is not None
            and self._amp_teacher_obs_set is not None
            else None
          )
          recovery_active = (
            obs[self._recovery_obs_group]
            if self._recovery_obs_group is not None
            and self._recovery_obs_group in obs.keys()
            else None
          )

          # Slot-driven obs / routing dicts (used by the slot-aware KL block
          # in DaggerPPO.update). For legacy runs without slots this loop is
          # a no-op and the empty dicts pass through to ``alg.act``.
          slot_obs: dict[str, "torch.Tensor | TensorDict"] = {}
          slot_routing: dict[str, torch.Tensor] = {}
          for slot in self._slots:
            s_obs_td = _select_obs_groups(obs, slot.obs_groups, slot.obs_set)
            if slot.is_structured:
              slot_obs[slot.name] = s_obs_td
            else:
              slot_obs[slot.name] = next(iter(s_obs_td.values()))
            # Routing: prefer the slot's own routing key, else its blend
            # group (loco's blend factor), else zeros (unpinned + no blend).
            r_key = slot.routing_signal_key or slot.blend_obs_group
            if r_key and r_key in obs.keys():
              slot_routing[slot.name] = obs[r_key]
            else:
              slot_routing[slot.name] = torch.zeros(
                self.env.num_envs, 1, device=self.device
              )

          actions = self.alg.act(
            obs,
            critic_obs=critic_obs,
            teacher_obs=teacher_obs,
            loco_obs=loco_obs,
            blend_weights=blend_weights,
            amp_obs=amp_obs,
            recovery_active=recovery_active,
            slot_obs=slot_obs,
            slot_routing=slot_routing,
          )
          if self._clip_actions is not None:
            actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
          next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          # Defensive: mjlab's reward_manager runs BEFORE _reset_idx, so a
          # ``nan_state`` termination resets the env's qpos/qvel cleanly
          # (next_obs is finite post-reset) but the per-step reward was
          # already computed from the NaN qpos and may contain NaN entries.
          # Without this mask, ``check_nan(next_obs, rewards, dones)`` would
          # trip on the rewards tensor and kill training. The terminated env's
          # reward is discarded by PPO anyway (dones=True clears the bootstrap
          # term), so zeroing it is semantically correct.
          rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
          if self.cfg.get("check_for_nan", True):
            from rsl_rl.utils import check_nan

            try:
              check_nan(next_obs, rewards, dones)
            except ValueError:
              # Per-group, per-term NaN dump for the failing envs. The mjData
              # state dump (mjlab NanGuard) captures qpos/qvel only; this
              # surfaces which OBSERVATION term went NaN, which is what
              # actually triggers the rsl_rl ValueError.
              import sys as _sys

              om = self.env.unwrapped.observation_manager
              for _gname in om.active_terms.keys():
                _gval = next_obs.get(_gname)
                if _gval is None:
                  continue
                if isinstance(_gval, dict):
                  # TensorDict path — terms aren't concatenated.
                  for _tname, _tv in _gval.items():
                    _bad = (
                      torch.isnan(_tv).any(dim=tuple(range(1, _tv.dim())))
                      | torch.isinf(_tv).any(dim=tuple(range(1, _tv.dim())))
                    )
                    if _bad.any():
                      _envs = torch.where(_bad)[0].tolist()[:8]
                      print(
                        f"[NanGuard][per-term] {_gname}.{_tname} envs={_envs} "
                        f"shape={tuple(_tv.shape[1:])}",
                        file=_sys.stderr,
                      )
                  continue
                # Flat concatenated path — slice per term.
                _term_names = om.active_terms[_gname]
                _term_dims = om.group_obs_term_dim[_gname]
                _offset = 0
                for _tname, _tdim in zip(_term_names, _term_dims):
                  _size = 1
                  for _d in _tdim:
                    _size *= _d
                  _slice = _gval[..., _offset : _offset + _size]
                  _bad = (
                    torch.isnan(_slice).any(dim=tuple(range(1, _slice.dim())))
                    | torch.isinf(_slice).any(dim=tuple(range(1, _slice.dim())))
                  )
                  if _bad.any():
                    _envs = torch.where(_bad)[0].tolist()[:8]
                    print(
                      f"[NanGuard][per-term] {_gname}.{_tname} envs={_envs} "
                      f"shape={tuple(_tdim)} offset={_offset}",
                      file=_sys.stderr,
                    )
                  _offset += _size

              # Also surface the previous-step action for the failing envs —
              # NaN actions at step N produce NaN actor_current.actions at N+1.
              _act_bad = (
                torch.isnan(actions).any(dim=tuple(range(1, actions.dim())))
                | torch.isinf(actions).any(dim=tuple(range(1, actions.dim())))
              )
              if _act_bad.any():
                _envs = torch.where(_act_bad)[0].tolist()[:8]
                print(
                  f"[NanGuard][per-term] applied_action[failing_envs={_envs}] "
                  f"is NaN/Inf — policy output went bad at the PREVIOUS step",
                  file=_sys.stderr,
                )
              else:
                print(
                  f"[NanGuard][per-term] applied_action is finite — NaN "
                  f"originates inside env step (not from policy output)",
                  file=_sys.stderr,
                )
              raise
          next_obs = next_obs.to(self.device)
          rewards = rewards.to(self.device)
          dones = dones.to(self.device)
          next_critic_obs = _select_obs_groups(next_obs, self._critic_obs_groups, "critic")
          self.alg.process_env_step(
            next_obs,
            rewards,
            dones,
            extras,
            critic_obs=next_critic_obs,
          )
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards=None)
          obs = next_obs

        collect_time = time.time() - rollout_start
        learn_start = time.time()
        final_critic_obs = _select_obs_groups(obs, self._critic_obs_groups, "critic")
        self.alg.compute_returns(final_critic_obs)

      loss_dict = self.alg.update()
      learn_time = time.time() - learn_start
      self.current_learning_iteration = it
      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=None,
      )

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

    if self.logger.writer is not None:
      self.save(
        os.path.join(
          self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"
        )
      )  # type: ignore[arg-type]
      self.logger.stop_logging_writer()

  def _load_optional_teacher(
    self,
    teacher_label: str,
    experiment_name: str | None,
    proj_name: str | None,
    checkpoint: str | int | None,
    actor_cfg: dict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    obs_example: TensorDict | torch.Tensor | None = None,
  ) -> nn.Module | None:
    if experiment_name in (None, "", "None", "dummy"):
      print(f"[DaggerOnPolicyRunner] {teacher_label} teacher disabled.")
      return None

    checkpoint_path = self._resolve_checkpoint_path(proj_name, experiment_name, checkpoint)
    if checkpoint_path is None:
      raise FileNotFoundError(
        f"[DaggerOnPolicyRunner] {teacher_label} teacher checkpoint not found "
        f"(experiment={experiment_name}, proj={proj_name}, checkpoint={checkpoint}). "
        "If you intended to run without this teacher, pass "
        "``--agent.<teacher>-experiment-name None`` explicitly. "
        "Common cause: running from a git worktree whose ``logs/`` dir is "
        "missing the teacher run; create a symlink or use an absolute path."
      )

    actor_cfg_local = copy.deepcopy(actor_cfg)
    class_name = actor_cfg_local.pop("class_name", None)
    actor_class = (
      cast(type[nn.Module], self.alg.get_policy().__class__)
      if class_name is None
      else cast(type[nn.Module], resolve_callable(class_name))
    )
    # Filter actor_cfg to kwargs the chosen class actually accepts. This is
    # required when loading a teacher trained with a richer model cfg (e.g.
    # RslRlModelCfg with cnn_cfg) into a simpler model class (MLPModel) — the
    # extra fields would raise TypeError.
    import inspect

    sig = inspect.signature(actor_class.__init__)
    if all(
      p.kind is not inspect.Parameter.VAR_KEYWORD
      for p in sig.parameters.values()
    ):
      accepted = {p.name for p in sig.parameters.values()}
      actor_cfg_local = {k: v for k, v in actor_cfg_local.items() if k in accepted}
    model = actor_class(
      (
        obs_example.to(self.device)
        if obs_example is not None
        else self.env.get_observations().to(self.device)
      ),
      obs_groups,
      obs_set,
      output_dim,
      **actor_cfg_local,
    ).to(self.device)
    loaded = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
    actor_state = loaded.get("actor_state_dict", loaded.get("model_state_dict"))
    if actor_state is None:
      raise KeyError(f"No actor weights found in checkpoint: {checkpoint_path}")
    model.load_state_dict(actor_state, strict=False)
    model.eval()
    print(
      "[DaggerOnPolicyRunner] "
      f"Loaded {teacher_label} teacher checkpoint: {checkpoint_path}"
    )
    return model

  def _resolve_checkpoint_path(
    self,
    proj_name: str | None,
    experiment_name: str,
    checkpoint: str | int | None,
  ) -> str | None:
    if isinstance(checkpoint, str) and checkpoint and os.path.isfile(checkpoint):
      return checkpoint
    if os.path.isfile(experiment_name):
      return experiment_name

    run_dir = None
    if proj_name and os.path.isdir(os.path.join(proj_name, experiment_name)):
      run_dir = os.path.join(proj_name, experiment_name)
    elif proj_name and os.path.isdir(proj_name):
      run_dir = proj_name
    elif os.path.isdir(experiment_name):
      run_dir = experiment_name

    # Fallback: search logs/rsl_rl/<experiment_name>/ for the latest run
    # (handles play mode where only the experiment name is in the default config)
    if run_dir is None:
      log_exp_dir = os.path.join("logs", "rsl_rl", experiment_name)
      if os.path.isdir(log_exp_dir):
        runs = sorted(
          d for d in os.listdir(log_exp_dir)
          if os.path.isdir(os.path.join(log_exp_dir, d))
        )
        if runs:
          run_dir = os.path.join(log_exp_dir, runs[-1])

    if run_dir is None:
      return None
    if isinstance(checkpoint, int) and checkpoint >= 0:
      path = os.path.join(run_dir, f"model_{checkpoint}.pt")
      return path if os.path.isfile(path) else None

    models = [
      os.path.join(run_dir, name)
      for name in os.listdir(run_dir)
      if name.startswith("model_") and name.endswith(".pt")
    ]
    if not models:
      return None
    models.sort(key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    return models[-1]
