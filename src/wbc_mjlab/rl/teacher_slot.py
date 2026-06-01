"""Runtime dataclass for an auxiliary teacher attached to the student via per-slot KL.

A ``TeacherSlot`` carries everything the algorithm and storage need at runtime to
incorporate one teacher into the student's loss: the loaded ``nn.Module``, which
obs groups feed it, which slice of the student's action vector it supervises,
how its KL is weighted and gated, and (for MoE) which expert it pins.

Existing per-teacher attributes on ``DaggerPPO`` (``teacher_actor``,
``loco_teacher_actor``, ``amp_teacher_actor``) are migrated into a list of
``TeacherSlot`` instances so that adding an Nth teacher (e.g. the stair-depth
teacher) is one config-list entry instead of a fresh attribute, storage field,
runner load, and KL block per teacher.

The config-side counterpart is ``WbcTeacherSlotCfg`` in
``wbc_mjlab.dagger_ppo_config``. The runner constructs ``TeacherSlot`` instances
from those configs by loading the actor checkpoint and resolving the obs groups
against the live env obs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn


@dataclass
class TeacherSlot:
  """A single auxiliary teacher attached to the student via per-slot KL.

  ``routing_signal_key`` names the obs group holding a ``[B, 1]`` per-env weight
  in ``[0, 1]``. Per-slot KL contribution is multiplied by this weight, so a
  slot is "off" on samples where its routing signal is 0.

  ``body_blend_role`` opts a slot into the special-cased body-blend KL block:
  ``"anchor"`` (WBC body, weighted by ``1 - blend``), ``"flat_specialist"``
  (loco, weighted by ``blend * (1 - stair_active)``), ``"stair_specialist"``
  (stair, weighted by ``blend * stair_active``). All body-blend slots share a
  single ``dagger_coef`` after the linear combination is gated-meaned together,
  matching the legacy ``(1-blend)*kl_wbc + blend*kl_loco`` formula at
  ``algorithms.py:417-422``. Slots with ``body_blend_role=None`` get a separate
  per-slot KL term gated by their own ``routing_signal_key`` (e.g. AMP).
  """

  name: str
  actor: nn.Module
  actor_cfg: dict[str, Any]
  obs_groups: dict[str, tuple[str, ...]]
  obs_set: str
  experiment_name: str | None = None
  proj_name: str | None = None
  checkpoint: str | int | None = None
  # MoE pinning. None = unpinned (the slot does not directly supervise any
  # expert's gate weight). e.g. AMP=2, Stair=3 in the 4-teacher config.
  expert_idx: int | None = None
  # Action slicing on the student side. ``slice(0, 15)`` for body-only loco /
  # stair teachers; ``slice(0, 29)`` for full body+arm teachers (WBC, AMP).
  dim_slice: slice = field(default_factory=lambda: slice(None, None))
  # KL coefficient anneal (mirrors the legacy ``dagger_coef``/``arm_kl_coef``/
  # ``amp_kl_coef`` cosine anneal pattern). ``kl_coef`` is mutable, set by the
  # algorithm each update.
  kl_coef_init: float = 0.0
  kl_coef: float = 0.0
  kl_coef_min: float = 0.0
  # Routing.
  routing_signal_key: str = ""
  # MoE direct supervision strength on this slot's pinned expert (analogous to
  # ``recovery_routing_coef`` for the AMP slot). 0 = no direct gate push.
  pin_routing_coef: float = 0.0
  # Out-of-regime gate-mass penalty (reserved; deferred).
  anti_route_coef: float = 0.0
  output_dim: int = 0
  # True if ``obs_groups`` resolves to >1 group (e.g. stair teacher's
  # ``(stair_teacher_actor, stair_teacher_actor_depth)``). Storage allocates a
  # TensorDict-shaped buffer in that case; flat tensor otherwise.
  is_structured: bool = False
  # Optional adapters for body-blend slots and the loco<->wbc_body coupling.
  # ``blend_obs_group`` names the per-env blend factor obs (sigmoid of
  # reference root velocity). Set on the loco slot.
  blend_obs_group: str | None = None
  # ``body_blend_role`` opts into the body-blend KL block (see class docstring).
  # One of: ``None`` (default; slot uses its own per-slot KL term),
  # ``"anchor"`` (WBC body), ``"flat_specialist"`` (loco),
  # ``"stair_specialist"`` (stair), or future ``"<regime>_specialist"`` strings.
  body_blend_role: str | None = None
  # True for the WBC arm-KL slot (slice [15:29]). Algorithm uses this to route
  # the slot through the dedicated arm-KL term instead of the body-blend block.
  is_arm_kl: bool = False
