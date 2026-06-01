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

import torch
from dataclasses import asdict

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

    # Provide a default motion file so export works without a full dataset.
    _DEFAULT_MOTION_FILE = "/home/yangl/handoff/wbc_handoff_data/OMOMO_g1_GMR/sub1_clothesstand_000.pkl"
    if "motion" in env_cfg.commands:
        motion_cmd = env_cfg.commands["motion"]
        if hasattr(motion_cmd, "motion_file"):
            motion_cmd.motion_file = _DEFAULT_MOTION_FILE
    
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
