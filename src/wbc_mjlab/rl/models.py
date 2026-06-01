"""HANDOFF-style observation models adapted to rsl_rl 5.x."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, resolve_nn_activation, unpad_trajectories

from wbc_mjlab.rl.encoders import DepthCNN2D, FutureMotionEncoder, TemporalConvEncoder
from wbc_mjlab.rl.moe_network import MixtureOfExperts


@dataclass(slots=True)
class _ObsGroupSpec:
  name: str
  shape: tuple[int, ...]

  @property
  def flat_dim(self) -> int:
    return math.prod(self.shape)

  @property
  def rank(self) -> int:
    return len(self.shape)


@dataclass(slots=True)
class _FlatObsLayout:
  motion_dim: int
  proprio_dim: int
  history_dim: int
  future_dim: int
  depth_dim: int = 0

  @property
  def total_dim(self) -> int:
    return (
      self.motion_dim
      + self.proprio_dim
      + self.history_dim
      + self.future_dim
      + self.depth_dim
    )


class _DepthCnnEncoder(nn.Module):
  """Compact 2D-conv depth encoder for the structured-depth obs mode.

  Stride-2 conv stack → adaptive avg-pool to 1×1 → linear projection. Used
  by ``_ObsModelBase`` when the obs_groups for a role expand into a
  ``[1-D flat, 3-D depth]`` pair (e.g. ``("actor", "actor_depth")``); the
  flat features are concatenated with this encoder's output before the MLP.
  Distinct from :class:`wbc_mjlab.rl.encoders.DepthCNN2D`, which targets
  the legacy single-group flat-tail layout.
  """

  def __init__(
    self,
    input_shape: tuple[int, int, int],
    channels: tuple[int, ...] | list[int],
    output_size: int,
    activation: str,
  ) -> None:
    super().__init__()
    c_in = input_shape[0]
    layers: list[nn.Module] = []
    for c_out in channels:
      layers.append(nn.Conv2d(c_in, int(c_out), kernel_size=3, stride=2, padding=1))
      layers.append(resolve_nn_activation(activation))
      c_in = int(c_out)
    self.cnn = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
    self.proj = nn.Linear(c_in, output_size)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.proj(self.cnn(x))


def _build_mlp(
  input_dim: int,
  output_dim: int,
  hidden_dims: tuple[int, ...] | list[int],
  activation: str,
  layer_norm: bool,
) -> nn.Sequential:
  layers: list[nn.Module] = []
  in_dim = input_dim
  for index, hidden_dim in enumerate(hidden_dims):
    layers.append(nn.Linear(in_dim, hidden_dim))
    if layer_norm and index == len(hidden_dims) - 1:
      layers.append(nn.LayerNorm(hidden_dim))
    layers.append(resolve_nn_activation(activation))
    in_dim = hidden_dim
  layers.append(nn.Linear(in_dim, output_dim))
  return nn.Sequential(*layers)


class _ObsModelBase(nn.Module):
  """Base model that supports flat or explicit sequence-shaped observation groups."""

  is_recurrent: bool = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    num_motion_observations: int = 0,
    num_motion_steps: int = 1,
    num_priop_observations: int = 0,
    num_history_steps: int = 0,
    num_future_observations: int = 0,
    num_future_steps: int = 0,
    motion_latent_dim: int = 64,
    history_latent_dim: int = 64,
    future_latent_dim: int = 64,
    future_encoder_dims: tuple[int, ...] | list[int] = (256, 128),
    future_dropout: float = 0.1,
    layer_norm: bool = False,
    tanh_encoder_output: bool = False,
    current_mimic_dim: int = 0,
    current_proprio_dim: int = 0,
    history_feature_dim: int = 0,
    history_length: int = 0,
    privileged_future_step_dim: int = 0,
    privileged_future_steps: int = 0,
    critic_current_dim: int = 0,
    critic_extras_dim: int = 0,
    num_depth_observations: int = 0,
    depth_shape: tuple[int, ...] = (1, 18, 32),
    depth_latent_dim: int = 64,
    depth_encoder_channels: tuple[int, int, int] = (8, 16, 32),
    **_: object,
  ) -> None:
    super().__init__()
    self.obs_set = obs_set
    self.obs_groups, self.obs_specs, self.obs_dim = self._get_obs_specs(
      obs, obs_groups, obs_set
    )
    self.output_dim = output_dim
    self.activation = activation
    self.tanh_encoder_output = tanh_encoder_output

    if obs_normalization:
      self.obs_normalizer: nn.Module = EmpiricalNormalization(self.obs_dim)
    else:
      self.obs_normalizer = nn.Identity()

    if distribution_cfg is not None:
      dist_cfg = copy.deepcopy(distribution_cfg)
      dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore[assignment]
      self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
      mlp_output_dim = self.distribution.input_dim
    else:
      self.distribution = None
      mlp_output_dim = output_dim

    self.depth_shape: tuple[int, ...] = tuple(depth_shape)
    self.depth_latent_dim = depth_latent_dim
    self.num_motion_steps = max(num_motion_steps, 1)
    self.num_history_steps = max(num_history_steps, 0)
    self.num_future_steps = max(num_future_steps, 0)

    # Resolve mode early so the structured-depth path can opt out of the
    # legacy flat-tail depth layout.
    self.mode = self._resolve_mode()

    # In any structured-depth mode the depth observation lives in its own
    # group as a ``[B, C, H, W]`` tensor, not concatenated into the flat
    # vector — keep the layout's ``depth_dim`` at zero so ``_encode_flat``
    # never tries to slice a depth tail off the end.
    layout_depth_dim = (
      0
      if self.mode in (
        "depth_structured",
        "actor_structured_depth",
        "critic_structured_depth",
      )
      else num_depth_observations
    )
    self.layout = _FlatObsLayout(
      motion_dim=num_motion_observations,
      proprio_dim=num_priop_observations,
      history_dim=num_priop_observations * num_history_steps,
      future_dim=num_future_observations,
      depth_dim=layout_depth_dim,
    )

    self.motion_encoder: TemporalConvEncoder | None = None
    self.history_encoder: TemporalConvEncoder | None = None
    self.future_encoder: FutureMotionEncoder | None = None
    # Legacy flat-tail depth encoder (DepthCNN2D); the structured mode
    # builds its own ``_DepthCnnEncoder`` further below and overwrites this
    # slot, so the type is ``nn.Module | None`` in practice.
    self.depth_encoder: nn.Module | None = None
    if self.mode != "depth_structured" and self.layout.depth_dim > 0:
      expected_depth_dim = math.prod(self.depth_shape)
      if self.layout.depth_dim != expected_depth_dim:
        raise ValueError(
          "num_depth_observations must equal prod(depth_shape), got "
          f"{self.layout.depth_dim} vs prod({self.depth_shape})={expected_depth_dim}"
        )
      self.depth_encoder = DepthCNN2D(
        input_shape=tuple(self.depth_shape),  # type: ignore[arg-type]
        output_size=depth_latent_dim,
        activation=activation,
        channels=depth_encoder_channels,
      )

    if self.mode == "actor_structured":
      self.actor_current_group = self.obs_specs[0].name
      self.actor_history_group = self.obs_specs[1].name
      self.actor_current_dim = self.obs_specs[0].flat_dim
      self.actor_history_length = self.obs_specs[1].shape[0]
      self.actor_history_feature_dim = math.prod(self.obs_specs[1].shape[1:])
      self._validate_expected_dim(
        "actor current",
        self.actor_current_dim,
        current_mimic_dim + current_proprio_dim,
      )
      self._validate_expected_dim(
        "actor history feature",
        self.actor_history_feature_dim,
        history_feature_dim,
      )
      self._validate_expected_dim(
        "actor history length",
        self.actor_history_length,
        history_length,
      )
      self.history_encoder = TemporalConvEncoder(
        input_size=self.actor_history_feature_dim,
        tsteps=self.actor_history_length,
        output_size=history_latent_dim,
        activation=activation,
      )
      feature_dim = self.actor_current_dim + history_latent_dim
    elif self.mode == "actor_structured_depth":
      # Three-group actor: current[1D] + history[2D] + depth[3D].
      # Mirrors the ``actor_structured`` history-encoder setup and adds a
      # depth CNN whose latent is concatenated alongside the proprio
      # latents before the MLP / MoE head.
      self.actor_current_group = self.obs_specs[0].name
      self.actor_history_group = self.obs_specs[1].name
      self.depth_group = self.obs_specs[2].name
      self.actor_current_dim = self.obs_specs[0].flat_dim
      self.actor_history_length = self.obs_specs[1].shape[0]
      self.actor_history_feature_dim = math.prod(self.obs_specs[1].shape[1:])
      self.depth_shape = self.obs_specs[2].shape
      self._validate_expected_dim(
        "actor current",
        self.actor_current_dim,
        current_mimic_dim + current_proprio_dim,
      )
      self._validate_expected_dim(
        "actor history feature",
        self.actor_history_feature_dim,
        history_feature_dim,
      )
      self._validate_expected_dim(
        "actor history length",
        self.actor_history_length,
        history_length,
      )
      self._validate_expected_dim(
        "actor depth", self.obs_specs[2].flat_dim, num_depth_observations
      )
      if depth_shape != (1, 18, 32) and tuple(depth_shape) != self.depth_shape:
        raise ValueError(
          f"Unexpected depth shape: configured {tuple(depth_shape)}, "
          f"obs reports {self.depth_shape}"
        )
      self.history_encoder = TemporalConvEncoder(
        input_size=self.actor_history_feature_dim,
        tsteps=self.actor_history_length,
        output_size=history_latent_dim,
        activation=activation,
      )
      self.depth_encoder = _DepthCnnEncoder(
        input_shape=self.depth_shape,  # type: ignore[arg-type]
        channels=depth_encoder_channels,
        output_size=depth_latent_dim,
        activation=activation,
      )
      feature_dim = self.actor_current_dim + history_latent_dim + depth_latent_dim
    elif self.mode == "critic_structured":
      self.critic_priv_group = self.obs_specs[0].name
      self.critic_current_group = self.obs_specs[1].name
      self.critic_extras_group = self.obs_specs[2].name
      self.critic_priv_steps = self.obs_specs[0].shape[0]
      self.critic_priv_step_dim = math.prod(self.obs_specs[0].shape[1:])
      self.critic_current_dim = self.obs_specs[1].flat_dim
      self.critic_extras_dim = self.obs_specs[2].flat_dim
      self._validate_expected_dim(
        "critic privileged future step",
        self.critic_priv_step_dim,
        privileged_future_step_dim,
      )
      self._validate_expected_dim(
        "critic privileged future length",
        self.critic_priv_steps,
        privileged_future_steps,
      )
      self._validate_expected_dim(
        "critic current",
        self.critic_current_dim,
        critic_current_dim,
      )
      self._validate_expected_dim(
        "critic extras",
        self.critic_extras_dim,
        critic_extras_dim,
      )
      self.motion_encoder = TemporalConvEncoder(
        input_size=self.critic_priv_step_dim,
        tsteps=self.critic_priv_steps,
        output_size=motion_latent_dim,
        activation=activation,
      )
      feature_dim = motion_latent_dim + self.critic_current_dim + self.critic_extras_dim
    elif self.mode == "critic_structured_depth":
      # Four-group critic: priv_future[2D] + current[1D] + extras[1D]
      # + depth[3D]. Mirrors the ``critic_structured`` setup and adds a
      # depth CNN — typically holds a privileged height_scan raycaster
      # (raycast against true terrain mesh) for the 4-teacher MoE
      # student's critic.
      self.critic_priv_group = self.obs_specs[0].name
      self.critic_current_group = self.obs_specs[1].name
      self.critic_extras_group = self.obs_specs[2].name
      self.depth_group = self.obs_specs[3].name
      self.critic_priv_steps = self.obs_specs[0].shape[0]
      self.critic_priv_step_dim = math.prod(self.obs_specs[0].shape[1:])
      self.critic_current_dim = self.obs_specs[1].flat_dim
      self.critic_extras_dim = self.obs_specs[2].flat_dim
      self.depth_shape = self.obs_specs[3].shape
      self._validate_expected_dim(
        "critic privileged future step",
        self.critic_priv_step_dim,
        privileged_future_step_dim,
      )
      self._validate_expected_dim(
        "critic privileged future length",
        self.critic_priv_steps,
        privileged_future_steps,
      )
      self._validate_expected_dim(
        "critic current", self.critic_current_dim, critic_current_dim
      )
      self._validate_expected_dim(
        "critic extras", self.critic_extras_dim, critic_extras_dim
      )
      self._validate_expected_dim(
        "critic depth", self.obs_specs[3].flat_dim, num_depth_observations
      )
      self.motion_encoder = TemporalConvEncoder(
        input_size=self.critic_priv_step_dim,
        tsteps=self.critic_priv_steps,
        output_size=motion_latent_dim,
        activation=activation,
      )
      self.depth_encoder = _DepthCnnEncoder(
        input_shape=self.depth_shape,  # type: ignore[arg-type]
        channels=depth_encoder_channels,
        output_size=depth_latent_dim,
        activation=activation,
      )
      feature_dim = (
        motion_latent_dim
        + self.critic_current_dim
        + self.critic_extras_dim
        + depth_latent_dim
      )
    elif self.mode == "depth_structured":
      # Two-group layout: ``obs_specs[0]`` is the flat features (proprio +
      # command + history + …), ``obs_specs[1]`` is the depth image.
      self.flat_group = self.obs_specs[0].name
      self.depth_group = self.obs_specs[1].name
      self.depth_shape = self.obs_specs[1].shape
      # ``num_depth_observations``, when set, is treated as a sanity check.
      self._validate_expected_dim(
        "depth", self.obs_specs[1].flat_dim, num_depth_observations
      )
      if depth_shape != (1, 18, 32) and tuple(depth_shape) != self.depth_shape:
        raise ValueError(
          f"Unexpected depth shape: configured {tuple(depth_shape)}, "
          f"obs reports {self.depth_shape}"
        )
      self.depth_encoder = _DepthCnnEncoder(
        input_shape=self.depth_shape,  # type: ignore[arg-type]
        channels=depth_encoder_channels,
        output_size=depth_latent_dim,
        activation=activation,
      )
      feature_dim = self.obs_specs[0].flat_dim + depth_latent_dim
    else:
      if self.layout.total_dim > self.obs_dim:
        raise ValueError(
          "Configured flat observation layout exceeds the concatenated TensorDict "
          f"size: layout={self.layout.total_dim}, obs_dim={self.obs_dim}"
        )
      if self.layout.motion_dim > 0:
        if self.layout.motion_dim % self.num_motion_steps != 0:
          raise ValueError(
            "num_motion_observations must divide evenly by num_motion_steps."
          )
        self.motion_encoder = TemporalConvEncoder(
          input_size=self.layout.motion_dim // self.num_motion_steps,
          tsteps=self.num_motion_steps,
          output_size=motion_latent_dim,
          activation=activation,
        )
      if self.layout.history_dim > 0 and self.num_history_steps > 0:
        self.history_encoder = TemporalConvEncoder(
          input_size=self.layout.proprio_dim,
          tsteps=self.num_history_steps,
          output_size=history_latent_dim,
          activation=activation,
        )
      if self.layout.future_dim > 0:
        self.future_encoder = FutureMotionEncoder(
          input_size=self.layout.future_dim,
          output_size=future_latent_dim,
          activation=activation,
          hidden_dims=future_encoder_dims,
          dropout=future_dropout,
        )
      feature_dim = self._get_flat_feature_dim(
        motion_latent_dim=motion_latent_dim,
        history_latent_dim=history_latent_dim,
        future_latent_dim=future_latent_dim,
      )

    self.mlp = _build_mlp(
      feature_dim,
      int(mlp_output_dim),
      hidden_dims,
      activation,
      layer_norm,
    )
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)

  def forward(
    self,
    obs: TensorDict | torch.Tensor,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    if isinstance(obs, TensorDict) and masks is not None:
      obs = unpad_trajectories(obs, masks)
    latent = self.get_latent(obs, masks, hidden_state)
    output = self.mlp(latent)
    if self.distribution is None:
      return output
    self.distribution.update(output)
    if stochastic_output:
      return self.distribution.sample()
    return self.distribution.deterministic_output(output)

  def get_latent(
    self,
    obs: TensorDict | torch.Tensor,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state
    flat = self._flatten_obs(obs)
    flat = self.obs_normalizer(flat)
    if self.mode == "flat":
      return self._encode_flat(flat)
    obs_td = self._unflatten_obs(flat)
    return self._encode_structured(obs_td)

  def reset(
    self,
    dones: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> None:
    del dones, hidden_state

  def get_hidden_state(self) -> HiddenState:
    return None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    del dones

  @property
  def output_mean(self) -> torch.Tensor:
    if self.distribution is None:
      raise AttributeError("Deterministic model has no output_mean.")
    return self.distribution.mean

  @property
  def output_std(self) -> torch.Tensor:
    if self.distribution is None:
      raise AttributeError("Deterministic model has no output_std.")
    return self.distribution.std

  @property
  def output_entropy(self) -> torch.Tensor:
    if self.distribution is None:
      raise AttributeError("Deterministic model has no output_entropy.")
    return self.distribution.entropy

  @property
  def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
    if self.distribution is None:
      raise AttributeError("Deterministic model has no distribution parameters.")
    return self.distribution.params

  def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    if self.distribution is None:
      raise AttributeError("Deterministic model has no log probabilities.")
    return self.distribution.log_prob(outputs)

  def get_kl_divergence(
    self,
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
  ) -> torch.Tensor:
    if self.distribution is None:
      raise AttributeError("Deterministic model has no KL divergence.")
    return self.distribution.kl_divergence(old_params, new_params)

  def as_jit(self) -> nn.Module:
    return _TorchObsModel(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _OnnxObsModel(self, verbose)

  def update_normalization(self, obs: TensorDict | torch.Tensor) -> None:
    if isinstance(self.obs_normalizer, EmpiricalNormalization):
      self.obs_normalizer.update(self._flatten_obs(obs))  # type: ignore[arg-type]

  def _resolve_mode(self) -> str:
    if (
      self.obs_set == "actor"
      and len(self.obs_specs) == 2
      and self.obs_specs[0].rank == 1
      and self.obs_specs[1].rank == 2
    ):
      return "actor_structured"
    # Three-group actor layout: current[1D] + history[2D] + depth[3D]. Used
    # by the depth-aware MoE student (4-teacher cfg) so the gate + experts
    # can see both the 11-frame proprio history and the front-camera depth
    # image. New mode rather than reusing ``depth_structured`` because
    # we don't want to lose the temporal-conv history encoder.
    if (
      self.obs_set == "actor"
      and len(self.obs_specs) == 3
      and self.obs_specs[0].rank == 1
      and self.obs_specs[1].rank == 2
      and self.obs_specs[2].rank == 3
    ):
      return "actor_structured_depth"
    if (
      self.obs_set == "critic"
      and len(self.obs_specs) == 3
      and self.obs_specs[0].rank == 2
      and self.obs_specs[1].rank == 1
      and self.obs_specs[2].rank == 1
    ):
      return "critic_structured"
    # Four-group critic layout: priv_future[2D] + current[1D] + extras[1D]
    # + depth[3D]. Used by the depth-aware MoE student's privileged critic.
    # The depth group typically holds a height_scan raycaster image, not
    # the camera depth — gives the critic the privileged terrain signal.
    if (
      self.obs_set == "critic"
      and len(self.obs_specs) == 4
      and self.obs_specs[0].rank == 2
      and self.obs_specs[1].rank == 1
      and self.obs_specs[2].rank == 1
      and self.obs_specs[3].rank == 3
    ):
      return "critic_structured_depth"
    # Two-group layout with a flat features group + a 3-D depth group
    # (``[1, H, W]``). Used by Loco-Teacher-Stair-…-DepthCNN for both actor
    # (``actor`` + ``actor_depth``) and critic (``critic`` + ``critic_depth``).
    if (
      len(self.obs_specs) == 2
      and self.obs_specs[0].rank == 1
      and self.obs_specs[1].rank == 3
    ):
      return "depth_structured"
    return "flat"

  def _get_obs_specs(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
  ) -> tuple[list[str], list[_ObsGroupSpec], int]:
    active_obs_groups = obs_groups[obs_set]
    specs: list[_ObsGroupSpec] = []
    total_dim = 0
    for obs_group in active_obs_groups:
      shape = tuple(obs[obs_group].shape[1:])
      spec = _ObsGroupSpec(name=obs_group, shape=shape)
      specs.append(spec)
      total_dim += spec.flat_dim
    return active_obs_groups, specs, total_dim

  def _flatten_obs(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
      return obs
    return torch.cat(
      [obs[group].reshape(obs[group].shape[0], -1) for group in self.obs_groups],
      dim=-1,
    )

  def _unflatten_obs(self, flat: torch.Tensor) -> TensorDict:
    batch = flat.shape[0]
    items: dict[str, torch.Tensor] = {}
    cursor = 0
    for spec in self.obs_specs:
      next_cursor = cursor + spec.flat_dim
      items[spec.name] = flat[:, cursor:next_cursor].reshape(batch, *spec.shape)
      cursor = next_cursor
    return TensorDict(items, batch_size=[batch], device=flat.device)

  def _encode_structured(self, obs: TensorDict) -> torch.Tensor:
    if self.mode == "actor_structured":
      current = obs[self.actor_current_group]
      history = obs[self.actor_history_group]
      if current.dim() != 2 or history.dim() != 3:
        raise ValueError(
          "Actor structured observations must be [B, D] + [B, H, D], got "
          f"{tuple(current.shape)} and {tuple(history.shape)}"
        )
      history_latent = self.history_encoder(history)
      latent = torch.cat((history_latent, current), dim=-1)
    elif self.mode == "critic_structured":
      priv_future = obs[self.critic_priv_group]
      current = obs[self.critic_current_group]
      extras = obs[self.critic_extras_group]
      if priv_future.dim() != 3 or current.dim() != 2 or extras.dim() != 2:
        raise ValueError(
          "Critic structured observations must be [B, T, D] + [B, D] + [B, D], got "
          f"{tuple(priv_future.shape)}, {tuple(current.shape)}, {tuple(extras.shape)}"
        )
      priv_latent = self.motion_encoder(priv_future)
      latent = torch.cat((priv_latent, current, extras), dim=-1)
    elif self.mode == "depth_structured":
      flat = obs[self.flat_group]
      depth = obs[self.depth_group]
      if flat.dim() != 2 or depth.dim() != 4:
        raise ValueError(
          "Depth structured observations must be [B, D] + [B, C, H, W], got "
          f"{tuple(flat.shape)} and {tuple(depth.shape)}"
        )
      assert self.depth_encoder is not None
      depth_latent = self.depth_encoder(depth)
      latent = torch.cat((flat, depth_latent), dim=-1)
    elif self.mode == "actor_structured_depth":
      current = obs[self.actor_current_group]
      history = obs[self.actor_history_group]
      depth = obs[self.depth_group]
      if current.dim() != 2 or history.dim() != 3 or depth.dim() != 4:
        raise ValueError(
          "Actor structured-depth observations must be [B, D] + [B, H, D] + "
          f"[B, C, H, W], got {tuple(current.shape)}, {tuple(history.shape)}, "
          f"{tuple(depth.shape)}"
        )
      history_latent = self.history_encoder(history)
      assert self.depth_encoder is not None
      depth_latent = self.depth_encoder(depth)
      latent = torch.cat((history_latent, current, depth_latent), dim=-1)
    elif self.mode == "critic_structured_depth":
      priv_future = obs[self.critic_priv_group]
      current = obs[self.critic_current_group]
      extras = obs[self.critic_extras_group]
      depth = obs[self.depth_group]
      if (
        priv_future.dim() != 3
        or current.dim() != 2
        or extras.dim() != 2
        or depth.dim() != 4
      ):
        raise ValueError(
          "Critic structured-depth observations must be [B, T, D] + [B, D] + "
          f"[B, D] + [B, C, H, W], got {tuple(priv_future.shape)}, "
          f"{tuple(current.shape)}, {tuple(extras.shape)}, {tuple(depth.shape)}"
        )
      priv_latent = self.motion_encoder(priv_future)
      assert self.depth_encoder is not None
      depth_latent = self.depth_encoder(depth)
      latent = torch.cat((priv_latent, current, extras, depth_latent), dim=-1)
    else:
      raise RuntimeError(f"Structured encoder called in unsupported mode: {self.mode}")

    if self.tanh_encoder_output:
      latent = torch.tanh(latent)
    return latent

  def _encode_flat(self, flat: torch.Tensor) -> torch.Tensor:
    motion_end = self.layout.motion_dim
    proprio_end = motion_end + self.layout.proprio_dim
    history_end = proprio_end + self.layout.history_dim
    future_end = history_end + self.layout.future_dim
    # Depth observations are placed at the END of the flat vector so
    # they can be sliced off and reshaped to [B, C, H, W] for the CNN.
    depth_start = flat.shape[1] - self.layout.depth_dim

    motion = flat[:, :motion_end]
    proprio = flat[:, motion_end:proprio_end]
    history = flat[:, proprio_end:history_end]
    future = flat[:, history_end:future_end]
    remainder = flat[:, future_end:depth_start]
    depth = flat[:, depth_start:] if self.layout.depth_dim > 0 else None

    features: list[torch.Tensor] = []
    if motion.numel() > 0 and self.motion_encoder is not None:
      features.append(self.motion_encoder(motion))
      single_motion = motion[:, : self.layout.motion_dim // self.num_motion_steps]
      features.append(single_motion)
    if history.numel() > 0 and self.history_encoder is not None:
      features.append(self.history_encoder(history))
    if future.numel() > 0 and self.future_encoder is not None:
      features.append(self.future_encoder(future))
    if proprio.numel() > 0:
      features.append(proprio)
    if remainder.numel() > 0:
      features.append(remainder)
    if depth is not None:
      if self.depth_encoder is not None:
        features.append(self.depth_encoder(depth))
      else:
        features.append(depth)
    latent = torch.cat(features, dim=-1) if features else flat
    if self.tanh_encoder_output:
      latent = torch.tanh(latent)
    return latent

  def _get_flat_feature_dim(
    self,
    motion_latent_dim: int,
    history_latent_dim: int,
    future_latent_dim: int,
  ) -> int:
    dim = self.obs_dim - self.layout.total_dim
    if self.layout.motion_dim > 0:
      dim += motion_latent_dim + self.layout.motion_dim // self.num_motion_steps
    if self.layout.history_dim > 0:
      dim += history_latent_dim
    if self.layout.future_dim > 0:
      dim += future_latent_dim
    dim += self.layout.proprio_dim
    if self.layout.depth_dim > 0:
      dim += self.depth_latent_dim if self.depth_encoder is not None else self.layout.depth_dim
    return dim

  @staticmethod
  def _validate_expected_dim(
    name: str,
    actual: int,
    expected: int,
  ) -> None:
    if expected > 0 and actual != expected:
      raise ValueError(f"Unexpected {name} dim: expected {expected}, got {actual}")


class ActorCriticFuture(_ObsModelBase):
  """HANDOFF future-motion model adapted to the rsl_rl 5.x model API."""


class ActorCriticFutureResidual(ActorCriticFuture):
  """Residual variant that adds a lightweight corrective head on top of a base actor."""

  def __init__(
    self,
    *args: object,
    residual_scale: float = 1.0,
    residual_loco_only: bool = False,
    residual_loco_dofs: int = 15,
    residual_use_base_action_input: bool = False,
    **kwargs: object,
  ) -> None:
    self.residual_scale = float(residual_scale)
    self.residual_loco_only = bool(residual_loco_only)
    self.residual_loco_dofs = int(residual_loco_dofs)
    self.residual_use_base_action_input = bool(residual_use_base_action_input)
    super().__init__(*args, **kwargs)

    if self.distribution is None:
      self.base_actor: ActorCriticFuture | None = None
      self.residual_head = None
      return

    self.base_actor = copy.deepcopy(ActorCriticFuture(*args, **kwargs))
    for param in self.base_actor.parameters():
      param.requires_grad = False

    self.residual_head = nn.Sequential(
      nn.LazyLinear(128),
      resolve_nn_activation(self.activation),
      nn.Linear(128, self.output_dim),
    )

  def forward(
    self,
    obs: TensorDict | torch.Tensor,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    if self.distribution is None or self.base_actor is None or self.residual_head is None:
      return super().forward(obs, masks, hidden_state, stochastic_output)

    if isinstance(obs, TensorDict) and masks is not None:
      obs = unpad_trajectories(obs, masks)
    latent = self.get_latent(obs, masks, hidden_state)
    base_actions = self.base_actor(obs, masks, hidden_state, stochastic_output=False)
    residual_input = latent
    if self.residual_use_base_action_input:
      residual_input = torch.cat((latent, base_actions), dim=-1)
    residual = self.residual_head(residual_input)
    if self.residual_loco_only:
      mask = torch.zeros_like(residual)
      mask[:, : self.residual_loco_dofs] = 1.0
      residual = residual * mask
    output = base_actions + self.residual_scale * residual
    self.distribution.update(output)
    if stochastic_output:
      return self.distribution.sample()
    return self.distribution.deterministic_output(output)

  def get_residual_regularization(
    self, obs: TensorDict | torch.Tensor
  ) -> torch.Tensor:
    if self.base_actor is None or self.residual_head is None:
      return torch.zeros((), device=next(self.parameters()).device)
    latent = self.get_latent(obs)
    base_actions = self.base_actor(obs, stochastic_output=False)
    residual_input = latent
    if self.residual_use_base_action_input:
      residual_input = torch.cat((latent, base_actions), dim=-1)
    residual = self.residual_head(residual_input)
    return residual.pow(2).mean()

  def load_base_actor_from_checkpoint(
    self,
    checkpoint_path: str,
    map_location: str | torch.device | None = None,
  ) -> tuple[dict, list[str], list[str]]:
    if self.base_actor is None:
      raise RuntimeError("Residual model has no base actor.")
    loaded = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    actor_state = loaded.get("actor_state_dict", loaded.get("model_state_dict", {}))
    missing, unexpected = self.base_actor.load_state_dict(actor_state, strict=False)
    return loaded, missing, unexpected


class _TorchObsModel(nn.Module):
  """TorchScript-friendly wrapper for custom models with flat external inputs."""

  def __init__(self, model: _ObsModelBase) -> None:
    super().__init__()
    cached_distribution = None
    if model.distribution is not None:
      cached_distribution = model.distribution._distribution
      model.distribution._distribution = None
    try:
      self.model = copy.deepcopy(model)
    finally:
      if model.distribution is not None:
        model.distribution._distribution = cached_distribution
    self.input_size = self.model.obs_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.model(x, stochastic_output=False)

  @torch.jit.export
  def reset(self) -> None:
    pass


class _OnnxObsModel(_TorchObsModel):
  """ONNX-friendly wrapper for custom models with flat external inputs."""

  is_recurrent: bool = False

  def __init__(self, model: _ObsModelBase, verbose: bool) -> None:
    del verbose
    super().__init__(model)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


class ActorCriticFutureMoE(ActorCriticFuture):
  """ActorCriticFuture with a Mixture-of-Experts head.

  Builds all encoders from ActorCriticFuture, then replaces the flat MLP head
  (self.mlp) with a MixtureOfExperts module. The gate sees the fused encoded
  latent, not raw observations.
  """

  def __init__(
    self,
    obs,
    obs_groups,
    role: str,
    num_outputs: int,
    num_experts: int = 4,
    expert_hidden_dims: list[int] | None = None,
    gate_hidden_dims: list[int] | None = None,
    **kwargs,
  ) -> None:
    super().__init__(obs, obs_groups, role, num_outputs, **kwargs)
    input_dim = self.mlp[0].in_features
    self.mlp = MixtureOfExperts(
      input_dim=input_dim,
      output_dim=num_outputs,
      num_experts=num_experts,
      expert_hidden_dims=expert_hidden_dims or [256, 128],
      gate_hidden_dims=gate_hidden_dims or [128, 64],
    )
