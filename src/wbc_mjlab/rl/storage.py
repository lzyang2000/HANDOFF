"""Custom rollout storage for HANDOFF-style DAgger PPO."""

from __future__ import annotations

import torch
from tensordict import TensorDict


def _zeros_like_td(
  obs: TensorDict,
  num_transitions_per_env: int,
  device: str,
) -> TensorDict:
  return TensorDict(
    {
      key: torch.zeros(
        num_transitions_per_env,
        *value.shape,
        device=device,
        dtype=value.dtype,
      )
      for key, value in obs.items()
    },
    batch_size=[num_transitions_per_env, obs.batch_size[0]],
    device=device,
  )


class DaggerRolloutStorage:
  class Transition:
    def __init__(self) -> None:
      self.observations: TensorDict | None = None
      self.critic_observations: TensorDict | None = None
      self.teacher_observations: TensorDict | None = None
      self.actions: torch.Tensor | None = None
      self.rewards: torch.Tensor | None = None
      self.dones: torch.Tensor | None = None
      self.values: torch.Tensor | None = None
      self.actions_log_prob: torch.Tensor | None = None
      self.distribution_params: tuple[torch.Tensor, ...] | None = None
      self.hidden_states = (None, None)
      self.loco_observations: torch.Tensor | None = None
      self.blend_weights: torch.Tensor | None = None
      self.amp_observations: torch.Tensor | None = None
      self.recovery_active: torch.Tensor | None = None
      # Generic teacher slot inputs. Keyed by slot name (e.g. "amp",
      # "stair"). slot_obs values are flat tensors for single-group teachers
      # and TensorDicts for structured (multi-group) teachers like the
      # stair-depth-CNN teacher. slot_routing values are [B] or [B, 1]
      # per-env weights in [0, 1] used to gate that slot's KL contribution.
      self.slot_obs: dict[str, "torch.Tensor | TensorDict"] = {}
      self.slot_routing: dict[str, torch.Tensor] = {}

    def clear(self) -> None:
      self.__init__()

  class Batch:
    def __init__(
      self,
      observations: TensorDict,
      critic_observations: TensorDict | None,
      teacher_observations: TensorDict | None,
      actions: torch.Tensor,
      values: torch.Tensor,
      advantages: torch.Tensor,
      returns: torch.Tensor,
      old_actions_log_prob: torch.Tensor,
      old_distribution_params: tuple[torch.Tensor, ...],
      loco_observations: torch.Tensor | None,
      blend_weights: torch.Tensor | None,
      amp_observations: torch.Tensor | None = None,
      recovery_active: torch.Tensor | None = None,
      slot_obs: dict[str, "torch.Tensor | TensorDict"] | None = None,
      slot_routing: dict[str, torch.Tensor] | None = None,
    ) -> None:
      self.observations = observations
      self.critic_observations = critic_observations
      self.teacher_observations = teacher_observations
      self.actions = actions
      self.values = values
      self.advantages = advantages
      self.returns = returns
      self.old_actions_log_prob = old_actions_log_prob
      self.old_distribution_params = old_distribution_params
      self.loco_observations = loco_observations
      self.blend_weights = blend_weights
      self.amp_observations = amp_observations
      self.recovery_active = recovery_active
      self.slot_obs: dict[str, "torch.Tensor | TensorDict"] = slot_obs or {}
      self.slot_routing: dict[str, torch.Tensor] = slot_routing or {}
      self.hidden_states = (None, None)
      self.masks = None

  def __init__(
    self,
    num_envs: int,
    num_transitions_per_env: int,
    obs: TensorDict,
    actions_shape: tuple[int, ...] | list[int],
    device: str = "cpu",
    critic_obs: TensorDict | None = None,
    teacher_obs: TensorDict | None = None,
  ) -> None:
    self.device = device
    self.num_envs = num_envs
    self.num_transitions_per_env = num_transitions_per_env
    self.actions_shape = actions_shape

    self.observations = _zeros_like_td(obs, num_transitions_per_env, device)
    self.critic_observations = (
      _zeros_like_td(critic_obs, num_transitions_per_env, device)
      if critic_obs is not None
      else None
    )
    self.teacher_observations = (
      _zeros_like_td(teacher_obs, num_transitions_per_env, device)
      if teacher_obs is not None
      else None
    )

    self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
    self.actions = torch.zeros(
      num_transitions_per_env, num_envs, *actions_shape, device=device
    )
    self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()
    self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
    self.actions_log_prob = torch.zeros(
      num_transitions_per_env, num_envs, 1, device=device
    )
    self.distribution_params: tuple[torch.Tensor, ...] | None = None
    self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
    self.advantages = torch.zeros(
      num_transitions_per_env, num_envs, 1, device=device
    )

    self.loco_observations: torch.Tensor | None = None
    self.blend_weights: torch.Tensor | None = None
    self.amp_observations: torch.Tensor | None = None
    self.recovery_active: torch.Tensor | None = None
    # Generic per-slot rollout buffers, keyed by slot name. Allocated by
    # ``init_slot_obs_buffer`` / ``init_slot_routing_buffer`` from the runner
    # once the slot's obs example shape is known.
    self.slot_obs: dict[str, "torch.Tensor | TensorDict"] = {}
    self.slot_routing: dict[str, torch.Tensor] = {}
    self.step = 0

  def init_slot_obs_buffer(
    self,
    name: str,
    obs_example: "torch.Tensor | TensorDict",
  ) -> None:
    """Allocate a rollout buffer for one teacher slot's obs.

    For a flat ``torch.Tensor`` example, allocates ``[T, N, D]`` zeros. For a
    ``TensorDict`` example (multi-group teacher like the stair-depth CNN),
    allocates a nested ``[T, N, ...]`` TensorDict matching the example's
    per-key shapes via ``_zeros_like_td``.
    """
    if isinstance(obs_example, torch.Tensor):
      self.slot_obs[name] = torch.zeros(
        self.num_transitions_per_env,
        self.num_envs,
        obs_example.shape[-1],
        device=self.device,
        dtype=obs_example.dtype,
      )
    else:
      self.slot_obs[name] = _zeros_like_td(
        obs_example, self.num_transitions_per_env, self.device
      )

  def init_slot_routing_buffer(self, name: str) -> None:
    """Allocate a ``[T, N, 1]`` rollout buffer for one slot's routing signal."""
    self.slot_routing[name] = torch.zeros(
      self.num_transitions_per_env, self.num_envs, 1, device=self.device
    )

  def init_loco_buffers(self, loco_obs_dim: int) -> None:
    self.loco_observations = torch.zeros(
      self.num_transitions_per_env, self.num_envs, loco_obs_dim, device=self.device
    )
    self.blend_weights = torch.zeros(
      self.num_transitions_per_env, self.num_envs, 1, device=self.device
    )

  def init_amp_buffers(self, amp_obs_dim: int) -> None:
    self.amp_observations = torch.zeros(
      self.num_transitions_per_env, self.num_envs, amp_obs_dim, device=self.device
    )
    self.recovery_active = torch.zeros(
      self.num_transitions_per_env, self.num_envs, 1, device=self.device
    )

  def add_transition(self, transition: Transition) -> None:
    if self.step >= self.num_transitions_per_env:
      raise OverflowError("Rollout buffer overflow.")

    self.observations[self.step].copy_(transition.observations)
    if self.critic_observations is not None and transition.critic_observations is not None:
      self.critic_observations[self.step].copy_(transition.critic_observations)
    if self.teacher_observations is not None and transition.teacher_observations is not None:
      self.teacher_observations[self.step].copy_(transition.teacher_observations)

    self.actions[self.step].copy_(transition.actions)  # type: ignore[arg-type]
    self.rewards[self.step].copy_(transition.rewards.view(-1, 1))  # type: ignore[union-attr]
    self.dones[self.step].copy_(transition.dones.view(-1, 1))  # type: ignore[union-attr]
    self.values[self.step].copy_(transition.values)  # type: ignore[arg-type]
    self.actions_log_prob[self.step].copy_(
      transition.actions_log_prob.view(-1, 1)  # type: ignore[union-attr]
    )

    if self.distribution_params is None:
      self.distribution_params = tuple(
        torch.zeros(
          self.num_transitions_per_env,
          *param.shape,
          device=self.device,
          dtype=param.dtype,
        )
        for param in transition.distribution_params  # type: ignore[arg-type]
      )
    for index, param in enumerate(transition.distribution_params or ()):
      self.distribution_params[index][self.step].copy_(param)

    if self.loco_observations is not None and transition.loco_observations is not None:
      self.loco_observations[self.step].copy_(transition.loco_observations)
    if self.blend_weights is not None and transition.blend_weights is not None:
      self.blend_weights[self.step].copy_(transition.blend_weights.view(-1, 1))
    if self.amp_observations is not None and transition.amp_observations is not None:
      self.amp_observations[self.step].copy_(transition.amp_observations)
    if self.recovery_active is not None and transition.recovery_active is not None:
      self.recovery_active[self.step].copy_(transition.recovery_active.view(-1, 1))

    # Generic slot buffers. ``transition.slot_obs[name]`` may be a flat
    # tensor or a TensorDict; ``buf.copy_`` handles both (TensorDict.copy_
    # recurses by key).
    for name, buf in self.slot_obs.items():
      src = transition.slot_obs.get(name)
      if src is None:
        continue
      buf[self.step].copy_(src)
    for name, buf in self.slot_routing.items():
      src = transition.slot_routing.get(name)
      if src is None:
        continue
      buf[self.step].copy_(src.view(-1, 1))

    self.step += 1

  def clear(self) -> None:
    self.step = 0

  def mini_batch_generator(
    self,
    num_mini_batches: int,
    num_epochs: int = 8,
  ):
    batch_size = self.num_envs * self.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(
      num_mini_batches * mini_batch_size,
      requires_grad=False,
      device=self.device,
    )

    observations = self.observations.flatten(0, 1)
    critic_observations = (
      self.critic_observations.flatten(0, 1)
      if self.critic_observations is not None
      else None
    )
    teacher_observations = (
      self.teacher_observations.flatten(0, 1)
      if self.teacher_observations is not None
      else None
    )
    actions = self.actions.flatten(0, 1)
    values = self.values.flatten(0, 1)
    returns = self.returns.flatten(0, 1)
    advantages = self.advantages.flatten(0, 1)
    old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
    old_distribution_params = tuple(
      param.flatten(0, 1) for param in self.distribution_params or ()
    )
    loco_observations = (
      self.loco_observations.flatten(0, 1)
      if self.loco_observations is not None
      else None
    )
    blend_weights = (
      self.blend_weights.flatten(0, 1) if self.blend_weights is not None else None
    )
    amp_observations = (
      self.amp_observations.flatten(0, 1)
      if self.amp_observations is not None
      else None
    )
    recovery_active = (
      self.recovery_active.flatten(0, 1) if self.recovery_active is not None else None
    )
    # Flatten the per-slot buffers once outside the inner loop (matches
    # how ``observations`` is flattened above). Both ``torch.Tensor`` and
    # ``TensorDict`` support ``flatten(0, 1)`` so this works uniformly.
    slot_obs_flat: dict[str, "torch.Tensor | TensorDict"] = {
      name: buf.flatten(0, 1) for name, buf in self.slot_obs.items()
    }
    slot_routing_flat: dict[str, torch.Tensor] = {
      name: buf.flatten(0, 1) for name, buf in self.slot_routing.items()
    }

    for _ in range(num_epochs):
      for mini_batch in range(num_mini_batches):
        start = mini_batch * mini_batch_size
        stop = (mini_batch + 1) * mini_batch_size
        batch_idx = indices[start:stop]
        yield DaggerRolloutStorage.Batch(
          observations=observations[batch_idx],  # type: ignore[arg-type]
          critic_observations=(
            critic_observations[batch_idx]
            if critic_observations is not None
            else None
          ),
          teacher_observations=(
            teacher_observations[batch_idx]
            if teacher_observations is not None
            else None
          ),
          actions=actions[batch_idx],
          values=values[batch_idx],
          advantages=advantages[batch_idx],
          returns=returns[batch_idx],
          old_actions_log_prob=old_actions_log_prob[batch_idx],
          old_distribution_params=tuple(
            param[batch_idx] for param in old_distribution_params
          ),
          loco_observations=(
            loco_observations[batch_idx] if loco_observations is not None else None
          ),
          blend_weights=(
            blend_weights[batch_idx] if blend_weights is not None else None
          ),
          amp_observations=(
            amp_observations[batch_idx] if amp_observations is not None else None
          ),
          recovery_active=(
            recovery_active[batch_idx] if recovery_active is not None else None
          ),
          slot_obs={n: b[batch_idx] for n, b in slot_obs_flat.items()},
          slot_routing={n: b[batch_idx] for n, b in slot_routing_flat.items()},
        )

  def recurrent_mini_batch_generator(
    self,
    num_mini_batches: int,
    num_epochs: int = 8,
  ):
    del num_mini_batches, num_epochs
    raise NotImplementedError("Recurrent custom DAgger storage is not implemented.")
