"""Runner for AMP-teacher tasks.

AMP env steps publish an ``"amp"`` observation group in the env's TensorDict.
``AmpPPO`` already reads that key inside ``act()`` and ``process_env_step()``,
so the rollout loop in ``MjlabOnPolicyRunner`` (inherited from upstream
``OnPolicyRunner``) needs no override here. We do override ``save()`` to also
export an ONNX of the actor — mirroring ``mjlab.tasks.velocity.rl.runner`` so
the AMP teacher's checkpoint produces a teacher-compatible ONNX usable by
``DaggerOnPolicyRunner._load_optional_teacher`` in a future student.
"""

from __future__ import annotations

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner


class AmpOnPolicyRunner(MjlabOnPolicyRunner):
  """MjlabOnPolicyRunner specialized for AMP-teacher training.

  The discriminator + amp_normalizer state is threaded through
  ``MjlabOnPolicyRunner.save/load`` automatically because ``AmpPPO`` overrides
  ``save()``/``load()`` to include them.
  """

  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
