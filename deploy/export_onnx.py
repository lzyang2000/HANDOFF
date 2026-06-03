import os
import sys
from pathlib import Path

# When invoked as `python deploy/export_onnx.py`, sys.path[0] is the deploy/
# directory, so `from deploy.common.X import ...` (pulled in transitively via
# wbc_mjlab.g1_constants_custom's NOMINAL_COMMAND) fails. Put the repo root
# on sys.path so that namespace import resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import glob
import torch
from dataclasses import asdict


def _resolve_motion_file(configured: str) -> str:
    """Return an existing motion file for env construction during export.

    Prefers ``configured`` if it exists; otherwise grabs a single enriched
    ``.pkl`` from a common local root. PklMotionLib accepts a single .pkl (it
    need not be a yaml dataset), and the pkl must be enriched — carry
    ``body_pos_w``/``body_quat_w`` (see enrich_pkl.py). The motion content does
    not affect the exported actor; we only need one clip so PklMotionLib doesn't
    hard-fail. Dataset paths baked into the config point at the origin machine's
    handoff dirs, which may not exist here.
    """
    if configured and os.path.exists(configured):
        return configured

    for root in (
        os.path.expanduser("~/twist2/seed_g1_enriched_pkl"),
        os.path.expanduser("~/twist2"),
        os.path.expanduser("~/handoff"),
    ):
        matches = sorted(glob.glob(os.path.join(root, "**", "*.pkl"), recursive=True))
        if matches:
            print(f"  Resolved motion file for export: {matches[0]}")
            return matches[0]

    raise FileNotFoundError(
        "export_onnx could not find an enriched .pkl to build the env. "
        f"Configured path was {configured!r} (missing). Searched under "
        "~/twist2 and ~/handoff. Point motion_cmd.motion_file at an enriched "
        ".pkl (see enrich_pkl.py)."
    )


def main():
    if len(sys.argv) < 3:
        print("Usage: python export_onnx.py <task_id> <checkpoint_path>")
        sys.exit(1)

    task_id = sys.argv[1]
    checkpoint_path = sys.argv[2]

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print(f"Loading task: {task_id}")
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    
    env_cfg = load_env_cfg(task_id, play=True)
    rl_cfg = load_rl_cfg(task_id)
    runner_cls = load_runner_cls(task_id)
    
    # Convert dataclass config to dict for the runner
    # Some older versions might use a simple dict, but here it is a dataclass.
    if hasattr(rl_cfg, "__dataclass_fields__"):
        rl_dict = asdict(rl_cfg)
    else:
        rl_dict = rl_cfg

    print(f"Loading checkpoint: {checkpoint_path}")
    device = "cpu"
    
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    
    print("Creating environment to resolve observation space...")
    env_cfg.scene.num_envs = 1
    if hasattr(env_cfg, 'viewer'):
        env_cfg.viewer.viewer = "auto"

    # Export only builds the env to resolve the observation space; the motion
    # content is irrelevant to the exported actor, but PklMotionLib hard-fails
    # if no motion loads. The dataset paths baked into the task config / training
    # run point at the origin machine's handoff dirs (/home/yangl/handoff/...),
    # which may not exist here. Resolve any single existing .pkl instead of
    # hard-coding one machine-specific path.
    if "motion" in env_cfg.commands:
        motion_cmd = env_cfg.commands["motion"]
        if hasattr(motion_cmd, "motion_file"):
            motion_cmd.motion_file = _resolve_motion_file(
                getattr(motion_cmd, "motion_file", "")
            )
    
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env)
    
    # Teachers are only needed during training (KL distillation); disable them
    # so export doesn't fail on observation-dimension mismatches (e.g. nobv).
    rl_dict["teacher_experiment_name"] = None
    rl_dict["loco_teacher_experiment_name"] = None
    rl_dict["amp_teacher_experiment_name"] = None

    # Auto-detect MoE expert count from the checkpoint so a 2-expert ckpt can
    # be exported via a task whose runner_cfg defaults to 3 experts (and vice
    # versa). Looks for `mlp.experts.<i>.*` keys in the actor state dict.
    if "num_experts" in rl_dict.get("actor", {}):
      import re

      _ckpt_peek = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
      _actor_state = _ckpt_peek.get(
        "actor_state_dict", _ckpt_peek.get("model_state_dict", {})
      )
      _expert_idxs = set()
      for _k in _actor_state.keys() if hasattr(_actor_state, "keys") else []:
        _m = re.match(r"mlp\.experts\.(\d+)\.", _k)
        if _m:
          _expert_idxs.add(int(_m.group(1)))
      if _expert_idxs:
        _ckpt_num_experts = max(_expert_idxs) + 1
        if rl_dict["actor"]["num_experts"] != _ckpt_num_experts:
          print(
            f"  Overriding actor.num_experts: "
            f"{rl_dict['actor']['num_experts']} -> {_ckpt_num_experts} "
            f"(detected from checkpoint)"
          )
          rl_dict["actor"]["num_experts"] = _ckpt_num_experts
      del _ckpt_peek, _actor_state

    print(f"Instantiating runner: {runner_cls.__name__}")
    # Pass rl_dict (the dictionary) instead of the dataclass object
    runner = runner_cls(env, rl_dict, device=device)
    
    print("Loading weights into runner...")
    runner.load(checkpoint_path)
    
    export_dir = Path(checkpoint_path).parent
    # The MjlabOnPolicyRunner uses the folder name for the .onnx file
    filename = f"{export_dir.name}.onnx"
    
    print(f"Exporting to: {export_dir / filename}")
    runner.export_policy_to_onnx(str(export_dir), filename=filename)
    print("Done!")

if __name__ == "__main__":
    main()
