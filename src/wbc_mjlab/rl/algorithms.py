"""Custom PPO + DAgger algorithm for HANDOFF-style policies."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from wbc_mjlab.rl.moe_network import (
  load_balancing_loss,
  recovery_aware_load_balancing_loss,
  subset_aware_load_balancing_loss,
)
from wbc_mjlab.rl.storage import DaggerRolloutStorage
from wbc_mjlab.rl.teacher_slot import TeacherSlot


def _gaussian_kl_per_dim(
  student_params: tuple[torch.Tensor, ...],
  teacher_params: tuple[torch.Tensor, ...],
) -> torch.Tensor:
  if len(student_params) != 2 or len(teacher_params) != 2:
    raise ValueError("Dual-teacher KL currently assumes Gaussian-style (mean, std) params.")
  mu_s, sigma_s = student_params
  mu_t, sigma_t = teacher_params
  return torch.log(sigma_t / sigma_s) + (
    sigma_s.pow(2) + (mu_s - mu_t).pow(2)
  ) / (2 * sigma_t.pow(2)) - 0.5


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


class DaggerPPO(PPO):
  """PPO with optional single- or dual-teacher KL imitation loss."""

  def __init__(
    self,
    actor: nn.Module,
    critic: nn.Module,
    storage: DaggerRolloutStorage,
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    clip_param: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    value_loss_coef: float = 1.0,
    entropy_coef: float = 0.01,
    learning_rate: float = 0.001,
    max_grad_norm: float = 1.0,
    optimizer: str = "adam",
    use_clipped_value_loss: bool = True,
    schedule: str = "adaptive",
    desired_kl: float = 0.01,
    normalize_advantage_per_mini_batch: bool = False,
    device: str = "cpu",
    dagger_coef: float = 0.0,
    dagger_coef_anneal_steps: int = 0,
    dagger_coef_min: float = 0.0,
    residual_reg_coef: float = 0.0,
    loco_num_actions: int = 15,
    loco_blend_dims: int = 15,
    loco_teacher_action_std_rescale: float = 1.0,
    arm_kl_coef: float | None = None,
    arm_kl_coef_min: float | None = None,
    amp_kl_coef: float | None = None,
    amp_kl_coef_min: float | None = None,
    amp_num_actions: int = 29,
    multi_gpu_cfg: dict | None = None,
    **kwargs: object,
  ) -> None:
    super().__init__(
      actor=actor,  # type: ignore[arg-type]
      critic=critic,  # type: ignore[arg-type]
      storage=storage,  # type: ignore[arg-type]
      num_learning_epochs=num_learning_epochs,
      num_mini_batches=num_mini_batches,
      clip_param=clip_param,
      gamma=gamma,
      lam=lam,
      value_loss_coef=value_loss_coef,
      entropy_coef=entropy_coef,
      learning_rate=learning_rate,
      max_grad_norm=max_grad_norm,
      optimizer=optimizer,
      use_clipped_value_loss=use_clipped_value_loss,
      schedule=schedule,
      desired_kl=desired_kl,
      normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
      device=device,
      rnd_cfg=None,
      symmetry_cfg=None,
      multi_gpu_cfg=multi_gpu_cfg,
    )
    del kwargs
    self.storage = storage
    self.dagger_coef_init = float(dagger_coef)
    self.dagger_coef = float(dagger_coef)
    self.dagger_coef_anneal_steps = int(dagger_coef_anneal_steps)
    self.dagger_coef_min = float(dagger_coef_min)
    self.residual_reg_coef = float(residual_reg_coef)
    self.loco_num_actions = int(loco_num_actions)
    self.loco_blend_dims = int(loco_blend_dims)
    self.loco_teacher_action_std_rescale = float(loco_teacher_action_std_rescale)
    self.arm_kl_coef_init = float(arm_kl_coef) if arm_kl_coef is not None else None
    self.arm_kl_coef = self.arm_kl_coef_init
    self.arm_kl_coef_min = float(arm_kl_coef_min) if arm_kl_coef_min is not None else self.arm_kl_coef_init
    self.amp_kl_coef_init = float(amp_kl_coef) if amp_kl_coef is not None else None
    self.amp_kl_coef = self.amp_kl_coef_init
    self.amp_kl_coef_min = (
      float(amp_kl_coef_min)
      if amp_kl_coef_min is not None
      else self.amp_kl_coef_init
    )
    self.amp_num_actions = int(amp_num_actions)
    self.teacher_actor: nn.Module | None = None
    self.loco_teacher_actor: nn.Module | None = None
    self.amp_teacher_actor: nn.Module | None = None
    # Generic teacher registry. Populated by ``set_teacher_models`` when the
    # runner passes a slot list (Step 4 of the migration). When non-empty,
    # ``update()`` runs the slot-aware KL block instead of the legacy
    # WBC/loco/AMP hard-coded blocks. ``self.slots_dict`` keys by name for
    # O(1) lookup; ``self.slots_list`` preserves cfg ordering for iteration.
    self.slots_list: list[TeacherSlot] = []
    self.slots_dict: dict[str, TeacherSlot] = {}
    self.update_counter = 0

  def set_teacher_models(
    self,
    teacher_actor: nn.Module | None = None,
    loco_teacher_actor: nn.Module | None = None,
    amp_teacher_actor: nn.Module | None = None,
    slots: list[TeacherSlot] | None = None,
  ) -> None:
    """Register frozen teacher actors.

    Two calling conventions during the migration to a slot-driven runtime:

    1. Legacy three-arg form ``set_teacher_models(teacher, loco, amp)`` —
       used by the unmigrated runner. Populates the per-teacher attributes;
       ``self.slots_*`` stays empty so ``update()`` runs the legacy KL block.
    2. Slot form ``set_teacher_models(slots=[TeacherSlot, ...])`` — used by
       the migrated runner. Populates ``self.slots_list`` /
       ``self.slots_dict``; ``update()`` runs the slot-aware KL block.

    The two forms are mutually exclusive: passing both raises ``ValueError``.
    """
    has_legacy = any(
      a is not None for a in (teacher_actor, loco_teacher_actor, amp_teacher_actor)
    )
    if slots is not None and has_legacy:
      raise ValueError(
        "set_teacher_models received both legacy positional teachers and a "
        "slots list. Use exactly one form."
      )

    if slots is not None:
      self.slots_list = []
      self.slots_dict = {}
      for s in slots:
        if s.actor is not None:
          s.actor = s.actor.to(self.device)
          s.actor.eval()
        s.kl_coef = float(s.kl_coef_init)
        self.slots_list.append(s)
        self.slots_dict[s.name] = s
      # Mirror the legacy attributes so any helper that still reads them
      # (e.g. ``_TeacherVisPolicy``) keeps working during the migration.
      self.teacher_actor = self._slot_actor("wbc_body") or self._slot_actor("wbc_arm")
      self.loco_teacher_actor = self._slot_actor("loco")
      self.amp_teacher_actor = self._slot_actor("amp")
      return

    self.teacher_actor = teacher_actor.to(self.device) if teacher_actor is not None else None
    self.loco_teacher_actor = (
      loco_teacher_actor.to(self.device) if loco_teacher_actor is not None else None
    )
    self.amp_teacher_actor = (
      amp_teacher_actor.to(self.device) if amp_teacher_actor is not None else None
    )
    if self.teacher_actor is not None:
      self.teacher_actor.eval()
    if self.loco_teacher_actor is not None:
      self.loco_teacher_actor.eval()
    if self.amp_teacher_actor is not None:
      self.amp_teacher_actor.eval()

  def _slot_actor(self, name: str) -> nn.Module | None:
    s = self.slots_dict.get(name)
    return s.actor if s is not None else None

  def _wrap_slot_obs(
    self,
    slot: TeacherSlot,
    obs: "torch.Tensor | TensorDict",
  ) -> "torch.Tensor | TensorDict":
    """Wrap a flat slot obs tensor into a TensorDict for the teacher's forward.

    Mirrors the legacy AMP wrapping at algorithms.py:471-479: when the teacher
    model is single-group (e.g. ``MLPModel`` / ``ActorCriticFuture`` with a
    single obs term) and the stored slot obs is a flat tensor, wrap under the
    teacher's expected key. Multi-group teachers (e.g. stair-depth) get
    a TensorDict-shaped buffer at storage init, so they pass through.
    """
    if not isinstance(obs, torch.Tensor):
      return obs
    teacher_obs_groups = getattr(slot.actor, "obs_groups", None)
    if teacher_obs_groups is None or len(teacher_obs_groups) != 1:
      return obs
    return TensorDict(
      {teacher_obs_groups[0]: obs},
      batch_size=obs.shape[:1],
      device=obs.device,
    )

  def _compute_slot_kl(
    self,
    batch: DaggerRolloutStorage.Batch,
    distribution_params: tuple[torch.Tensor, ...],
    recov_per_env: torch.Tensor | None,
  ) -> tuple[torch.Tensor, dict[str, float]]:
    """Slot-aware replacement for the legacy WBC + loco + AMP KL block.

    Returns ``(slot_kl_total, logs)`` where:
      - ``slot_kl_total`` is a single scalar tensor that aggregates ALL
        slot KL contributions (body-blend + arm + every pinned slot).
        ``update()`` adds this to ``loss`` once, so future slots can't
        be silently dropped from the gradient.
      - ``logs`` is a dict of scalar floats with per-slot tensorboard
        keys (``body_kl``, ``arm_kl``, ``<slot.name>_kl`` for each
        pinned slot). ``update()`` synthesizes the legacy
        ``Loss/teacher_kl`` (= body + arm) and ``Loss/amp_kl`` (= the
        AMP slot's contribution) from this dict for backwards-compat.

    Body-blend (slot ``body_blend_role`` in {anchor, flat_specialist,
    stair_specialist}) is one ``_gated_mean(linear_combo) * dagger_coef``
    over the linear combination
    ``(1-blend)*kl_wbc + blend*((1-stair)*kl_loco + stair*kl_stair)``. Arm
    KL is one ``_gated_mean(kl_wbc_arm) * arm_kl_coef`` (terrain-independent).
    Pinned non-blend slots (e.g. AMP) emit per-slot KL gated by the slot's
    routing signal.

    With the auto-migrated 4 legacy slots and no stair slot, this is
    algebraically equivalent to the legacy ``loco_blend_dims=15`` path at
    [algorithms.py:416-426] (single ``_gated_mean`` over the body blend).
    """
    B = distribution_params[0].shape[0]
    device = distribution_params[0].device

    walk_base = (1.0 - recov_per_env) if recov_per_env is not None else torch.ones(B, device=device)
    walk_2d = walk_base.unsqueeze(-1)

    # Per-env loco_blend factor (sigmoid of reference root velocity).
    blend = (
      batch.blend_weights.squeeze(-1)
      if batch.blend_weights is not None
      else torch.zeros(B, device=device)
    )
    # Stair-active mask. Looked up by slot name; absent slot -> all zeros so
    # the stair term collapses out (flat-only behavior).
    stair_routing = batch.slot_routing.get("stair")
    stair_active_t = (
      stair_routing.squeeze(-1).clamp(0.0, 1.0)
      if stair_routing is not None
      else torch.zeros(B, device=device)
    )

    # ---- Body-blend block. ----
    # Body slice is determined by ``loco_blend_dims`` for back-compat with
    # 12-dim configs; the slot abstraction defaults to the body slice from
    # the wbc_body slot's dim_slice.
    wbc_body_slot = self.slots_dict.get("wbc_body")
    loco_slot = self.slots_dict.get("loco")
    stair_slot = self.slots_dict.get("stair")
    # Body slice end (15 by default, 12 if loco_blend_dims=12).
    body_end = self.loco_blend_dims if self.loco_blend_dims in (12, 15) else 15
    body_term_total = torch.zeros(B, body_end, device=device)

    if (
      wbc_body_slot is not None
      and wbc_body_slot.actor is not None
      and self.dagger_coef > 0
    ):
      wbc_obs = batch.slot_obs.get(wbc_body_slot.name)
      if wbc_obs is not None:
        wbc_obs = self._wrap_slot_obs(wbc_body_slot, wbc_obs)
        with torch.no_grad():
          wbc_body_slot.actor(wbc_obs, stochastic_output=False)
          wbc_params = tuple(
            p.detach() for p in wbc_body_slot.actor.output_distribution_params
          )
        kl_wbc_body = _gaussian_kl_per_dim(
          (distribution_params[0][:, :body_end], distribution_params[1][:, :body_end]),
          (wbc_params[0][:, :body_end], wbc_params[1][:, :body_end]),
        )
        body_term_total = body_term_total + (1.0 - blend).unsqueeze(-1) * kl_wbc_body

    if loco_slot is not None and loco_slot.actor is not None and self.dagger_coef > 0:
      loco_obs = batch.slot_obs.get(loco_slot.name)
      if loco_obs is not None:
        loco_obs_wrapped = self._wrap_slot_obs(loco_slot, loco_obs)
        with torch.no_grad():
          loco_slot.actor(loco_obs_wrapped, stochastic_output=False)
          loco_params = tuple(
            p.detach() for p in loco_slot.actor.output_distribution_params
          )
          # Optional std-rescale on dim 14 (legacy: loco_teacher_action_std_rescale).
          if (
            self.loco_teacher_action_std_rescale != 1.0
            and loco_params[1].shape[-1] > 14
          ):
            scaled_std = loco_params[1].clone()
            scaled_std[:, 14] = scaled_std[:, 14] * self.loco_teacher_action_std_rescale
            loco_params = (loco_params[0], scaled_std)
        n = min(body_end, loco_params[0].shape[-1])
        kl_loco = _gaussian_kl_per_dim(
          (distribution_params[0][:, :n], distribution_params[1][:, :n]),
          (loco_params[0][:, :n], loco_params[1][:, :n]),
        )
        # Pad to body_end with zeros if loco outputs fewer dims.
        if n < body_end:
          pad = torch.zeros(B, body_end - n, device=device)
          kl_loco = torch.cat([kl_loco, pad], dim=-1)
        body_term_total = (
          body_term_total + (blend * (1.0 - stair_active_t)).unsqueeze(-1) * kl_loco
        )

    if (
      stair_slot is not None
      and stair_slot.actor is not None
      and self.dagger_coef > 0
    ):
      stair_obs = batch.slot_obs.get(stair_slot.name)
      if stair_obs is not None:
        stair_obs_wrapped = self._wrap_slot_obs(stair_slot, stair_obs)
        with torch.no_grad():
          stair_slot.actor(stair_obs_wrapped, stochastic_output=False)
          stair_params = tuple(
            p.detach() for p in stair_slot.actor.output_distribution_params
          )
        n = min(body_end, stair_params[0].shape[-1])
        kl_stair = _gaussian_kl_per_dim(
          (distribution_params[0][:, :n], distribution_params[1][:, :n]),
          (stair_params[0][:, :n], stair_params[1][:, :n]),
        )
        if n < body_end:
          pad = torch.zeros(B, body_end - n, device=device)
          kl_stair = torch.cat([kl_stair, pad], dim=-1)
        body_term_total = (
          body_term_total + (blend * stair_active_t).unsqueeze(-1) * kl_stair
        )

    # Single _gated_mean over the linear combination, single dagger_coef.
    body_kl_loss = torch.zeros((), device=device)
    if self.dagger_coef > 0 and (
      wbc_body_slot is not None or loco_slot is not None or stair_slot is not None
    ):
      denom = (walk_2d.sum() * body_end).clamp_min(1.0)
      body_kl_loss = (body_term_total * walk_2d).sum() / denom * self.dagger_coef

    # ---- Arm KL term (terrain-independent, gated by walk_base only). ----
    arm_kl_loss = torch.zeros((), device=device)
    wbc_arm_slot = self.slots_dict.get("wbc_arm")
    if (
      wbc_arm_slot is not None
      and wbc_arm_slot.actor is not None
      and wbc_arm_slot.kl_coef > 0
    ):
      arm_obs = batch.slot_obs.get(wbc_arm_slot.name)
      if arm_obs is not None:
        arm_obs_wrapped = self._wrap_slot_obs(wbc_arm_slot, arm_obs)
        with torch.no_grad():
          wbc_arm_slot.actor(arm_obs_wrapped, stochastic_output=False)
          wbc_arm_params = tuple(
            p.detach() for p in wbc_arm_slot.actor.output_distribution_params
          )
        sl = wbc_arm_slot.dim_slice
        kl_arm = _gaussian_kl_per_dim(
          (distribution_params[0][:, sl], distribution_params[1][:, sl]),
          (wbc_arm_params[0][:, sl], wbc_arm_params[1][:, sl]),
        )
        denom = (walk_2d.sum() * kl_arm.shape[-1]).clamp_min(1.0)
        arm_kl_loss = (kl_arm * walk_2d).sum() / denom * wbc_arm_slot.kl_coef

    # Single accumulator for the gradient — every slot contribution lands
    # here, so adding a future "siphoned-off" slot can never be silently
    # dropped from ``loss``.
    slot_kl_total = body_kl_loss + arm_kl_loss
    logs: dict[str, float] = {
      "body_kl": body_kl_loss.item(),
      "arm_kl": arm_kl_loss.item(),
    }

    # ---- Pinned non-blend slots (e.g. AMP). ----
    for slot in self.slots_list:
      if slot.expert_idx is None:
        continue  # body-blend slots and arm slot are unpinned
      if slot.body_blend_role is not None:
        continue  # body-blend specialists already handled above
      if slot.is_arm_kl:
        continue  # arm slot already handled above
      if slot.kl_coef <= 0 or slot.actor is None:
        continue
      routing = batch.slot_routing.get(slot.name)
      if routing is None:
        continue
      routing_per_env = routing.squeeze(-1).clamp(0.0, 1.0)
      if routing_per_env.sum() <= 0:
        continue
      s_obs = batch.slot_obs.get(slot.name)
      if s_obs is None:
        continue
      s_obs_wrapped = self._wrap_slot_obs(slot, s_obs)
      with torch.no_grad():
        slot.actor(s_obs_wrapped, stochastic_output=True)
        t_params = tuple(p.detach() for p in slot.actor.output_distribution_params)
      sl = slot.dim_slice
      n = min(
        slot.output_dim if slot.output_dim > 0 else distribution_params[0][:, sl].shape[-1],
        distribution_params[0][:, sl].shape[-1],
        t_params[0].shape[-1],
      )
      s_p = (
        distribution_params[0][:, sl][:, :n],
        distribution_params[1][:, sl][:, :n],
      )
      t_p = (t_params[0][:, :n], t_params[1][:, :n])
      kl_pd = _gaussian_kl_per_dim(s_p, t_p)
      masked_sum = (kl_pd * routing_per_env.unsqueeze(-1)).sum()
      denom = (routing_per_env.sum() * kl_pd.shape[-1]).clamp_min(1.0)
      slot_loss = (masked_sum / denom) * slot.kl_coef
      slot_kl_total = slot_kl_total + slot_loss
      logs[f"{slot.name}_kl"] = slot_loss.item()

    return slot_kl_total, logs

  def _compute_aux_loss(
    self,
    batch: DaggerRolloutStorage.Batch,
    loss: torch.Tensor,
  ) -> tuple[torch.Tensor, dict[str, float]]:
    """Hook for subclasses to inject extra losses before backward().

    Returns the (possibly augmented) loss and a dict of scalar metrics
    to include in the update log.
    """
    return loss, {}

  def act(
    self,
    obs: TensorDict,
    critic_obs: TensorDict | None = None,
    teacher_obs: TensorDict | None = None,
    loco_obs: torch.Tensor | None = None,
    blend_weights: torch.Tensor | None = None,
    amp_obs: torch.Tensor | None = None,
    recovery_active: torch.Tensor | None = None,
    slot_obs: dict[str, "torch.Tensor | TensorDict"] | None = None,
    slot_routing: dict[str, torch.Tensor] | None = None,
  ) -> torch.Tensor:
    self.transition.hidden_states = (
      self.actor.get_hidden_state(),
      self.critic.get_hidden_state(),
    )
    self.transition.actions = self.actor(obs, stochastic_output=True).detach()
    critic_input = critic_obs if critic_obs is not None else obs
    self.transition.values = self.critic(critic_input).detach()
    self.transition.actions_log_prob = self.actor.get_output_log_prob(
      self.transition.actions
    ).detach()
    self.transition.distribution_params = tuple(
      param.detach() for param in self.actor.output_distribution_params
    )
    self.transition.observations = obs
    self.transition.critic_observations = critic_input
    self.transition.teacher_observations = teacher_obs
    self.transition.loco_observations = loco_obs
    self.transition.blend_weights = blend_weights
    self.transition.amp_observations = amp_obs
    self.transition.recovery_active = recovery_active
    self.transition.slot_obs = slot_obs or {}
    self.transition.slot_routing = slot_routing or {}
    return self.transition.actions

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
    critic_obs: TensorDict | None = None,
    teacher_obs: TensorDict | None = None,
    loco_obs: torch.Tensor | None = None,
    blend_weights: torch.Tensor | None = None,
  ) -> None:
    self.actor.update_normalization(obs)
    self.critic.update_normalization(critic_obs if critic_obs is not None else obs)

    self.transition.rewards = rewards.clone()
    self.transition.dones = dones

    if "time_outs" in extras:
      self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
        1,
      )

    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.actor.reset(dones)
    self.critic.reset(dones)

  def compute_returns(self, critic_obs: TensorDict | None = None) -> None:
    critic_input = (
      critic_obs
      if critic_obs is not None
      else self.storage.critic_observations[self.storage.step - 1]  # type: ignore[index]
      if self.storage.critic_observations is not None and self.storage.step > 0
      else self.storage.observations[self.storage.step - 1]
    )
    last_values = self.critic(critic_input).detach()
    advantage = 0
    for step in reversed(range(self.storage.num_transitions_per_env)):
      next_values = (
        last_values if step == self.storage.num_transitions_per_env - 1 else self.storage.values[step + 1]
      )
      next_is_not_terminal = 1.0 - self.storage.dones[step].float()
      delta = (
        self.storage.rewards[step]
        + next_is_not_terminal * self.gamma * next_values
        - self.storage.values[step]
      )
      advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
      self.storage.returns[step] = advantage + self.storage.values[step]
    self.storage.advantages = self.storage.returns - self.storage.values
    if not self.normalize_advantage_per_mini_batch:
      self.storage.advantages = (
        self.storage.advantages - self.storage.advantages.mean()
      ) / (self.storage.advantages.std() + 1e-8)

  def update(self) -> dict[str, float]:
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_teacher_kl = 0.0
    mean_amp_kl = 0.0
    mean_residual_reg = 0.0
    _aux_accum: dict[str, float] = {}

    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    for batch in generator:
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (
            batch.advantages - batch.advantages.mean()
          ) / (batch.advantages.std() + 1e-8)

      self.actor(batch.observations, stochastic_output=True)
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(
        batch.critic_observations
        if batch.critic_observations is not None
        else batch.observations
      )
      distribution_params = self.actor.output_distribution_params
      entropy = self.actor.output_entropy

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params, distribution_params
          )
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

      teacher_kl_loss = torch.zeros((), device=self.device)
      amp_kl_loss = torch.zeros((), device=self.device)
      slot_kl_logs: dict[str, float] = {}

      # Per-env recovery gate: 1.0 on envs in down-state, 0.0 elsewhere.
      # Used to multiply (1 - recov) into the existing WBC/loco KL terms,
      # and recov into the AMP-teacher full-body KL.
      if (
        batch.recovery_active is not None
        and self.amp_teacher_actor is not None
        and self.amp_kl_coef is not None
      ):
        recov_per_env = batch.recovery_active.squeeze(-1).clamp(0.0, 1.0)
      else:
        recov_per_env = None

      if self.slots_list:
        # Slot-aware path. ``_compute_slot_kl`` returns a single
        # ``slot_kl_total`` plus a dict of per-slot scalar logs. Adding
        # ``slot_kl_total`` to ``loss`` once means future slots can't be
        # silently dropped from the gradient (cf. the AMP-loss-not-added
        # regression at commit aaf7318). The legacy
        # ``Loss/teacher_kl`` (= body + arm) and ``Loss/amp_kl`` keys are
        # synthesized from ``slot_kl_logs`` for backwards-compat with
        # existing wandb dashboards.
        slot_kl_total, slot_kl_logs = self._compute_slot_kl(
          batch, distribution_params, recov_per_env
        )
        loss = loss + slot_kl_total
        # Legacy aliases — same semantics as the pre-refactor split.
        teacher_kl_loss = torch.tensor(
          slot_kl_logs.get("body_kl", 0.0) + slot_kl_logs.get("arm_kl", 0.0),
          device=self.device,
        )
        amp_kl_loss = torch.tensor(
          slot_kl_logs.get("amp_kl", 0.0), device=self.device
        )
      elif self.teacher_actor is not None and self.dagger_coef > 0:
        teacher_obs = (
          batch.teacher_observations
          if batch.teacher_observations is not None
          else batch.critic_observations
          if batch.critic_observations is not None
          else batch.observations
        )
        with torch.no_grad():
          self.teacher_actor(teacher_obs, stochastic_output=False)
          teacher_params = tuple(
            param.detach() for param in self.teacher_actor.output_distribution_params
          )
        student_params = tuple(param for param in distribution_params)
        kl_manip = _gaussian_kl_per_dim(student_params, teacher_params)

        not_recov_per_env = (
          (1.0 - recov_per_env).unsqueeze(-1) if recov_per_env is not None else None
        )

        def _gated_mean(tensor: torch.Tensor) -> torch.Tensor:
          """Active-subset mean: divides by the number of *active* (non-recov)
          (env, dim) cells, NOT the full batch size. Without this the
          documented KL coefficient is silently diluted by the recovery
          fraction. With ``delay_reset_env_ratio=0.20``, full-batch mean
          attenuates the body/arm KL by ~0.8x; subset mean restores the
          documented strength so ``dagger_coef`` and ``arm_kl_coef``
          mean what their docstrings say.
          """
          if not_recov_per_env is None:
            return tensor.mean()
          masked_sum = (tensor * not_recov_per_env).sum()
          # Effective denom = (active envs) * (dims of `tensor` along last axis).
          # tensor may be shape [B] or [B, D]; not_recov_per_env is [B, 1].
          if tensor.dim() == 1:
            denom = not_recov_per_env.squeeze(-1).sum().clamp_min(1.0)
          else:
            denom = (not_recov_per_env.sum() * tensor.shape[-1]).clamp_min(1.0)
          return masked_sum / denom

        if (
          self.loco_teacher_actor is not None
          and batch.loco_observations is not None
          and batch.blend_weights is not None
        ):
          with torch.no_grad():
            self.loco_teacher_actor(batch.loco_observations, stochastic_output=False)
            loco_params = tuple(
              param.detach() for param in self.loco_teacher_actor.output_distribution_params
            )
            if self.loco_teacher_action_std_rescale != 1.0:
              scaled_std = loco_params[1].clone()
              scaled_std[:, 14] = scaled_std[:, 14] * self.loco_teacher_action_std_rescale
              loco_params = (loco_params[0], scaled_std)
          blend = batch.blend_weights.squeeze(-1)
          loco_blend_dims = self.loco_blend_dims
          if loco_blend_dims == 12:
            # Blend only legs (0-11); waist (12-14) + arms (15+) always follow WBC teacher.
            # Loco teacher may output 12 or 15 dims — only first 12 are used.
            kl_loco = _gaussian_kl_per_dim(
              (student_params[0][:, :12], student_params[1][:, :12]),
              (loco_params[0][:, :12], loco_params[1][:, :12]),
            ).mean(dim=-1)
            kl_manip_legs = kl_manip[:, :12].mean(dim=-1)
            kl_manip_rest = kl_manip[:, 12:].mean(dim=-1)
            body_per_env = (1 - blend) * kl_manip_legs + blend * kl_loco
            if self.arm_kl_coef is not None:
              body_kl_loss = (
                _gated_mean(body_per_env.unsqueeze(-1)).squeeze() * self.dagger_coef
              )
              arm_kl_loss = (
                _gated_mean(kl_manip_rest.unsqueeze(-1)).squeeze() * self.arm_kl_coef
              )
              teacher_kl_loss = body_kl_loss + arm_kl_loss
            else:
              combined = body_per_env + kl_manip_rest
              teacher_kl_loss = (
                _gated_mean(combined.unsqueeze(-1)).squeeze() * self.dagger_coef
              )
          elif loco_blend_dims == 15:
            blend_body = blend.unsqueeze(-1)
            kl_loco = _gaussian_kl_per_dim(
              (student_params[0][:, :15], student_params[1][:, :15]),
              (loco_params[0][:, :15], loco_params[1][:, :15]),
            )
            blended = (1 - blend_body) * kl_manip[:, :15] + blend_body * kl_loco
            if self.arm_kl_coef is not None:
              body_kl_loss = _gated_mean(blended) * self.dagger_coef
              arm_kl_loss = _gated_mean(kl_manip[:, 15:]) * self.arm_kl_coef
              teacher_kl_loss = body_kl_loss + arm_kl_loss
            else:
              teacher_kl_loss = (
                _gated_mean(torch.cat([blended, kl_manip[:, 15:]], dim=-1))
                * self.dagger_coef
              )
          else:
            raise ValueError(
              f"Unsupported loco_blend_dims={loco_blend_dims}; expected 12 or 15."
            )
        else:
          teacher_kl_loss = _gated_mean(kl_manip) * self.dagger_coef
      elif (
        self.loco_teacher_actor is not None
        and batch.loco_observations is not None
        and self.dagger_coef > 0
      ):
        # Loco-teacher-only KL path (e.g. 15-DoF student with no WBC teacher).
        with torch.no_grad():
          self.loco_teacher_actor(batch.loco_observations, stochastic_output=False)
          loco_params = tuple(
            param.detach() for param in self.loco_teacher_actor.output_distribution_params
          )
        student_params = tuple(param for param in distribution_params)
        n = min(student_params[0].shape[-1], loco_params[0].shape[-1])
        kl_loco = _gaussian_kl_per_dim(
          (student_params[0][:, :n], student_params[1][:, :n]),
          (loco_params[0][:, :n], loco_params[1][:, :n]),
        )
        teacher_kl_loss = kl_loco.mean() * self.dagger_coef

      # Slot path adds ``slot_kl_total`` to ``loss`` directly inside the
      # ``if self.slots_list:`` branch above; the legacy ``teacher_kl_loss``
      # and ``amp_kl_loss`` tensors there are scalar copies for the
      # ``mean_teacher_kl`` / ``mean_amp_kl`` accumulators only. Skip the
      # legacy adds in that case to avoid double-counting.
      if not self.slots_list:
        loss = loss + teacher_kl_loss

      # Legacy AMP recovery teacher: full-body KL gated to recovery-active
      # envs. Only fires when slots aren't populated; the slot-aware path
      # accounts for the AMP slot's KL via ``slot_kl_total`` above.
      if (
        not self.slots_list
        and self.amp_teacher_actor is not None
        and self.amp_kl_coef is not None
        and self.amp_kl_coef > 0
        and batch.amp_observations is not None
        and batch.recovery_active is not None
      ):
        recov_mask = batch.recovery_active.squeeze(-1).clamp(0.0, 1.0)
        if recov_mask.sum() > 0:
          # MLPModel-style teachers expect a TensorDict keyed on their own
          # obs_groups; wrap the stored flat tensor accordingly.
          amp_obs_groups = getattr(self.amp_teacher_actor, "obs_groups", None)
          if amp_obs_groups is not None and len(amp_obs_groups) == 1:
            amp_obs_input = TensorDict(
              {amp_obs_groups[0]: batch.amp_observations},
              batch_size=batch.amp_observations.shape[:1],
              device=batch.amp_observations.device,
            )
          else:
            amp_obs_input = batch.amp_observations
          with torch.no_grad():
            # MLPModel only updates self.distribution.params when
            # stochastic_output=True; we discard the sampled action and read
            # the distribution params directly. Other teacher classes
            # (ActorCriticFuture) ignore the flag for params caching.
            self.amp_teacher_actor(amp_obs_input, stochastic_output=True)
            amp_params = tuple(
              param.detach()
              for param in self.amp_teacher_actor.output_distribution_params
            )
          n = min(self.amp_num_actions, distribution_params[0].shape[-1], amp_params[0].shape[-1])
          student_amp_params = (
            distribution_params[0][:, :n],
            distribution_params[1][:, :n],
          )
          amp_params_sliced = (amp_params[0][:, :n], amp_params[1][:, :n])
          kl_amp_per_dim = _gaussian_kl_per_dim(student_amp_params, amp_params_sliced)
          # Active-subset mean: divide by recov-active (env, dim) cells,
          # not the full minibatch. Without this the documented
          # ``amp_kl_coef`` is silently diluted by the recovery fraction
          # (~0.20), so 0.4 in the train script becomes an effective
          # 0.08 per-recovery-sample weight — much weaker than body KL
          # at 0.32. Subset normalization restores the documented value
          # to its actual per-sample magnitude on recovery samples.
          masked_sum = (kl_amp_per_dim * recov_mask.unsqueeze(-1)).sum()
          denom = (recov_mask.sum() * kl_amp_per_dim.shape[-1]).clamp_min(1.0)
          amp_kl_loss = (masked_sum / denom) * self.amp_kl_coef
          # Legacy path adds the freshly-computed AMP KL to ``loss`` here.
          # Slot path adds it implicitly via ``slot_kl_total`` above.
          loss = loss + amp_kl_loss

      # Per-slot KL granular tensorboard keys (slot path only). Pushed
      # into ``_aux_accum`` so they ride alongside ``Aux/lb_loss`` etc.
      for k, v in slot_kl_logs.items():
        _aux_accum[k] = _aux_accum.get(k, 0.0) + v

      residual_reg_loss = torch.zeros((), device=self.device)
      if self.residual_reg_coef > 0 and hasattr(self.actor, "get_residual_regularization"):
        residual_reg_loss = self.actor.get_residual_regularization(batch.observations)
        loss += self.residual_reg_coef * residual_reg_loss

      loss, _aux = self._compute_aux_loss(batch, loss)
      for k, v in _aux.items():
        _aux_accum[k] = _aux_accum.get(k, 0.0) + v

      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_teacher_kl += teacher_kl_loss.item()
      mean_amp_kl += amp_kl_loss.item()
      mean_residual_reg += residual_reg_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_teacher_kl /= num_updates
    mean_amp_kl /= num_updates
    mean_residual_reg /= num_updates
    aux_means = {k: v / num_updates for k, v in _aux_accum.items()}

    self.storage.clear()
    self.update_counter += 1
    self._update_dagger_coef()

    return {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "teacher_kl": mean_teacher_kl,
      "amp_kl": mean_amp_kl,
      "residual_reg": mean_residual_reg,
      "dagger_coef": self.dagger_coef,
      "amp_kl_coef": self.amp_kl_coef if self.amp_kl_coef is not None else 0.0,
      **aux_means,
    }

  def _update_dagger_coef(self) -> None:
    if self.dagger_coef_anneal_steps <= 0:
      return
    if self.update_counter < self.dagger_coef_anneal_steps:
      progress = self.update_counter / self.dagger_coef_anneal_steps
      cosine = 0.5 * (1 + math.cos(math.pi * progress))
      self.dagger_coef = self.dagger_coef_min + (
        self.dagger_coef_init - self.dagger_coef_min
      ) * cosine
      if self.arm_kl_coef_init is not None and self.arm_kl_coef_min is not None:
        self.arm_kl_coef = self.arm_kl_coef_min + (
          self.arm_kl_coef_init - self.arm_kl_coef_min
        ) * cosine
      if self.amp_kl_coef_init is not None and self.amp_kl_coef_min is not None:
        self.amp_kl_coef = self.amp_kl_coef_min + (
          self.amp_kl_coef_init - self.amp_kl_coef_min
        ) * cosine
    else:
      self.dagger_coef = self.dagger_coef_min
      if self.arm_kl_coef_min is not None:
        self.arm_kl_coef = self.arm_kl_coef_min
      if self.amp_kl_coef_min is not None:
        self.amp_kl_coef = self.amp_kl_coef_min

    # Anneal each slot's ``kl_coef`` between its ``kl_coef_init`` and
    # ``kl_coef_min`` with the same cosine schedule. Body-blend slots'
    # ``kl_coef`` is unused (the body-blend block reads ``self.dagger_coef``
    # directly); annealing them anyway keeps the slot dataclass consistent
    # with what it would be if read by an external caller / log dump.
    if self.update_counter < self.dagger_coef_anneal_steps:
      progress = self.update_counter / self.dagger_coef_anneal_steps
      cosine = 0.5 * (1 + math.cos(math.pi * progress))
    else:
      cosine = 0.0
    for slot in self.slots_list:
      if slot.kl_coef_init <= 0 and slot.kl_coef_min <= 0:
        continue
      slot.kl_coef = slot.kl_coef_min + (slot.kl_coef_init - slot.kl_coef_min) * cosine

  @staticmethod
  def construct_algorithm(
    obs: TensorDict,
    env: VecEnv,
    cfg: dict,
    device: str,
  ) -> "DaggerPPO":
    alg_class: type[DaggerPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore[assignment]
    actor_class = resolve_callable(cfg["actor"].pop("class_name"))
    critic_class = resolve_callable(cfg["critic"].pop("class_name"))

    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
    wbc_teacher_obs_groups = cfg.get("wbc_teacher_obs_groups")
    wbc_teacher_obs_set = cfg.get("wbc_teacher_obs_set")
    if wbc_teacher_obs_groups is not None and wbc_teacher_obs_set is not None:
      wbc_teacher_obs = _select_obs_groups(obs, wbc_teacher_obs_groups, wbc_teacher_obs_set)
    else:
      wbc_teacher_obs = obs
    loco_teacher_obs_groups = cfg.get("loco_teacher_obs_groups")
    loco_teacher_obs_set = cfg.get("loco_teacher_obs_set")
    loco_teacher_obs = None
    if loco_teacher_obs_groups is not None and loco_teacher_obs_set is not None:
      loco_teacher_obs_td = _select_obs_groups(
        obs,
        loco_teacher_obs_groups,
        loco_teacher_obs_set,
      )
      # Multi-group loco obs (e.g. depth-aware stair-plainvel teacher with
      # ``("loco_teacher_actor_flat", "actor_depth")``) is plumbed through
      # the slot mechanism in the runner — it gets a TensorDict-shaped
      # ``slot_obs["loco"]`` buffer instead of the legacy flat
      # ``loco_observations`` buffer. Skip the legacy single-group check
      # in that case; ``init_loco_buffers`` is also skipped below.
      if len(loco_teacher_obs_td.keys()) == 1:
        loco_teacher_obs = next(iter(loco_teacher_obs_td.values()))
    amp_teacher_obs_groups = cfg.get("amp_teacher_obs_groups")
    amp_teacher_obs_set = cfg.get("amp_teacher_obs_set")
    amp_teacher_obs = None
    if amp_teacher_obs_groups is not None and amp_teacher_obs_set is not None:
      amp_teacher_obs_td = _select_obs_groups(
        obs,
        amp_teacher_obs_groups,
        amp_teacher_obs_set,
      )
      if len(amp_teacher_obs_td.keys()) != 1:
        raise ValueError(
          "amp_teacher_obs_set must resolve to exactly one observation group."
        )
      amp_teacher_obs = next(iter(amp_teacher_obs_td.values()))
    cfg["algorithm"].setdefault("rnd_cfg", None)
    cfg["algorithm"].setdefault("symmetry_cfg", None)

    actor = actor_class(
      obs,
      cfg["obs_groups"],
      "actor",
      env.num_actions,
      **copy.deepcopy(cfg["actor"]),
    ).to(device)
    critic = critic_class(
      obs,
      cfg["obs_groups"],
      "critic",
      1,
      **copy.deepcopy(cfg["critic"]),
    ).to(device)

    storage = DaggerRolloutStorage(
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      [env.num_actions],
      device=device,
      critic_obs=obs,
      teacher_obs=wbc_teacher_obs,
    )
    if loco_teacher_obs is not None:
      storage.init_loco_buffers(loco_teacher_obs.shape[-1])
    if amp_teacher_obs is not None:
      storage.init_amp_buffers(amp_teacher_obs.shape[-1])
    return alg_class(
      actor=actor,
      critic=critic,
      storage=storage,
      device=device,
      **cfg["algorithm"],
      multi_gpu_cfg=cfg.get("multi_gpu"),
    )


class MoEDaggerPPO(DaggerPPO):
  """DaggerPPO with an auxiliary load-balancing loss for MoE actors.

  Requires the actor to be ActorCriticFutureMoE (or any model whose
  self.mlp is a MixtureOfExperts with a .gate attribute).
  """

  def __init__(
    self,
    actor: nn.Module,
    critic: nn.Module,
    storage: DaggerRolloutStorage,
    load_balance_coef: float = 0.01,
    recovery_routing_coef: float = 0.0,
    recovery_expert_idx: int = -1,
    **kwargs: object,
  ) -> None:
    super().__init__(actor, critic, storage, **kwargs)
    self.load_balance_coef = float(load_balance_coef)
    self.recovery_routing_coef = float(recovery_routing_coef)
    self.recovery_expert_idx = int(recovery_expert_idx)

  def _compute_aux_loss(
    self,
    batch: DaggerRolloutStorage.Batch,
    loss: torch.Tensor,
  ) -> tuple[torch.Tensor, dict[str, float]]:
    latent = self.actor.get_latent(batch.observations)
    gate_weights = self.actor.mlp.gate(latent)
    diagnostics: dict[str, float] = {}
    num_experts = gate_weights.shape[-1]

    # Build the (pinned_idxs, pinned_routing, pinned_coefs) trio. Slot-aware
    # path takes precedence; legacy ``self.recovery_expert_idx`` /
    # ``self.recovery_routing_coef`` is the fallback for runs that haven't
    # been migrated to slots yet.
    pinned_idxs: list[int] = []
    pinned_routing: dict[int, torch.Tensor] = {}
    pinned_coefs: dict[int, float] = {}

    if self.slots_list:
      for slot in self.slots_list:
        if slot.expert_idx is None:
          continue
        if slot.pin_routing_coef <= 0.0:
          continue
        idx = slot.expert_idx % num_experts
        r = batch.slot_routing.get(slot.name)
        if r is None:
          continue
        pinned_idxs.append(idx)
        pinned_routing[idx] = r
        pinned_coefs[idx] = float(slot.pin_routing_coef)
    else:
      # Legacy fallback: use the recovery routing fields.
      recov = getattr(batch, "recovery_active", None)
      if self.recovery_routing_coef > 0.0 and recov is not None:
        idx = self.recovery_expert_idx % num_experts
        pinned_idxs.append(idx)
        pinned_routing[idx] = recov
        pinned_coefs[idx] = float(self.recovery_routing_coef)

    # Per-pin diagnostics. These are pure logging (no gradient). Backwards-
    # compatible aliases for the legacy ``gate_recov_e2`` / ``gate_nonrecov_e2``
    # / ``recov_active_frac`` keys keep dashboards / regression scripts working.
    with torch.no_grad():
      legacy_rec_idx = self.recovery_expert_idx % num_experts
      for idx in set(pinned_idxs):
        r = pinned_routing.get(idx)
        if r is None:
          continue
        r_flat = r.flatten().float()
        g = gate_weights[:, idx]
        r_sum = r_flat.sum().clamp_min(1.0)
        nr_sum = (1.0 - r_flat).sum().clamp_min(1.0)
        gate_active = ((g * r_flat).sum() / r_sum).item()
        gate_inactive = ((g * (1.0 - r_flat)).sum() / nr_sum).item()
        diagnostics[f"gate_active_e{idx}"] = gate_active
        diagnostics[f"gate_inactive_e{idx}"] = gate_inactive
        diagnostics[f"active_frac_e{idx}"] = r_flat.mean().item()
        # Legacy aliases — only emit for the recovery expert idx so dashboards
        # tracking ``gate_recov_e2`` for the 3-teacher run keep working.
        if idx == legacy_rec_idx:
          diagnostics["gate_recov_e2"] = gate_active
          diagnostics["gate_nonrecov_e2"] = gate_inactive
          diagnostics["recov_active_frac"] = r_flat.mean().item()
      # Default zero values for the legacy keys when no recovery pin is active.
      diagnostics.setdefault("gate_recov_e2", 0.0)
      diagnostics.setdefault("gate_nonrecov_e2", 0.0)
      diagnostics.setdefault("recov_active_frac", 0.0)

    if pinned_idxs:
      lb_unpinned, lb_pinned = subset_aware_load_balancing_loss(
        gate_weights, pinned_idxs, pinned_routing
      )
      aux = self.load_balance_coef * lb_unpinned
      for idx, lb in lb_pinned.items():
        aux = aux + pinned_coefs[idx] * lb
        diagnostics[f"lb_pin_loss_e{idx}"] = lb.item()
      diagnostics["lb_loss"] = lb_unpinned.item()
      # Legacy alias.
      diagnostics["lb_recov_loss"] = (
        lb_pinned.get(legacy_rec_idx, torch.zeros((), device=gate_weights.device)).item()
      )
      return loss + aux, diagnostics

    lb = load_balancing_loss(gate_weights)
    diagnostics["lb_loss"] = lb.item()
    diagnostics["lb_recov_loss"] = 0.0
    return loss + self.load_balance_coef * lb, diagnostics
