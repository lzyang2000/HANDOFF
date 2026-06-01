"""Enrich PKL motion files with world-frame body positions and orientations.

Takes a HANDOFF YAML dataset file, runs MuJoCo forward kinematics on every
referenced PKL to compute body_pos_w and body_quat_w, and outputs enriched
PKLs + a new YAML pointing to them.

Optionally applies capture-point correction so the static CoM projection
stays inside a shrunk support polygon. Two interchangeable solvers are
provided (``--method {lbfgsb,cbf}``) and two eligibility triggers
(``--trigger {squat,standing}``); see :mod:`wbc_mjlab.stability_offline`
for the full derivation.

Usage:
    uv run python -m wbc_mjlab.scripts.enrich_pkl \
        --dataset /path/to/handoff_dataset.yaml \
        --output-dir /path/to/enriched/ \
        --method cbf --trigger standing
"""

from __future__ import annotations

import argparse
import os
import pickle
import multiprocessing as mp
from pathlib import Path

import mujoco
import numpy as np
import yaml
from tqdm import tqdm

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from wbc_mjlab.stability_offline import (
  build_body_name_map,
  correct_capture_point,
  needs_squat_correction,
)


# ---------------------------------------------------------------------------
# Numpy-compat unpickler (remaps numpy._core.* → numpy.core.* when loading
# NumPy-2-era pickles under NumPy 1.x)
# ---------------------------------------------------------------------------

class _NumpyCompatUnpickler(pickle.Unpickler):
  _MODULE_MAP = {
    "numpy._core.multiarray": "numpy.core.multiarray",
    "numpy._core.numeric": "numpy.core.numeric",
  }

  def find_class(self, module, name):
    module = self._MODULE_MAP.get(module, module)
    return super().find_class(module, name)


def _load_pickle_compat(file_obj):
  return _NumpyCompatUnpickler(file_obj).load()


# ---------------------------------------------------------------------------
# Main enrichment
# ---------------------------------------------------------------------------

def enrich_single_pkl(
  input_path: str,
  output_path: str,
  xml_path: str,
  *,
  safety_margin: float = 0.10,
  capture_point: bool = True,
  urdf_path: str = "",
  method: str = "lbfgsb",
  trigger: str = "squat",
) -> str | None:
  """Enrich a single PKL file with MuJoCo FK data. Returns error string or None."""
  try:
    with open(input_path, "rb") as f:
      data = _load_pickle_compat(f)
  except Exception as e:
    return f"Failed to load {input_path}: {e}"

  root_pos = np.asarray(data["root_pos"])  # [T, 3]
  root_rot = np.asarray(data["root_rot"])  # [T, 4] in [x,y,z,w]
  dof_pos = np.asarray(data["dof_pos"])    # [T, 29]
  link_body_list = data["link_body_list"]
  T = root_pos.shape[0]

  # Convert root_rot from [x,y,z,w] → [w,x,y,z] for MuJoCo
  root_rot_wxyz = root_rot[:, [3, 0, 1, 2]]

  model = mujoco.MjModel.from_xml_path(xml_path)
  mj_data = mujoco.MjData(model)
  pkl_to_mj = build_body_name_map(model, link_body_list)

  n_bodies = len(link_body_list)
  body_pos_w = np.zeros((T, n_bodies, 3), dtype=np.float32)
  body_quat_w = np.zeros((T, n_bodies, 4), dtype=np.float32)
  body_quat_w[:, :, 0] = 1.0  # identity [w,x,y,z]

  for t in range(T):
    mj_data.qpos[:3] = root_pos[t]
    mj_data.qpos[3:7] = root_rot_wxyz[t]
    mj_data.qpos[7:] = dof_pos[t]
    mujoco.mj_kinematics(model, mj_data)
    for pkl_idx, mj_idx in pkl_to_mj.items():
      body_pos_w[t, pkl_idx] = mj_data.xpos[mj_idx]
      body_quat_w[t, pkl_idx] = mj_data.xquat[mj_idx]

  # Save enriched PKL (preserve all original keys + add new ones)
  enriched = dict(data)
  enriched["body_pos_w"] = body_pos_w
  enriched["body_quat_w"] = body_quat_w

  # Capture-point correction.
  # For "squat" trigger, the cheap root-height check skips obviously non-squat
  # PKLs. For "standing" trigger, we always run the full detection because
  # every PKL may have standing frames.
  if capture_point:
    should_run = (trigger == "standing") or needs_squat_correction(enriched)
    if should_run:
      try:
        correct_capture_point(
          enriched, urdf_path, safety_margin, xml_path,
          method=method, trigger=trigger,
        )
      except Exception as e:
        return f"Capture-point correction failed for {input_path}: {e}"

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  with open(output_path, "wb") as f:
    pickle.dump(enriched, f)

  return None


def _worker(args: tuple) -> tuple[str, str | None]:
  """Multiprocessing worker. Returns (input_path, error_or_none)."""
  (input_path, output_path, xml_path, safety_margin, capture_point,
   urdf_path, method, trigger) = args
  err = enrich_single_pkl(
    input_path, output_path, xml_path,
    safety_margin=safety_margin,
    capture_point=capture_point,
    urdf_path=urdf_path,
    method=method,
    trigger=trigger,
  )
  return (input_path, err)


def main() -> None:
  parser = argparse.ArgumentParser(description="Enrich PKL dataset with MuJoCo FK")
  parser.add_argument("--dataset", required=True, help="Path to HANDOFF dataset YAML file")
  parser.add_argument("--output-dir", required=True, help="Output directory for enriched PKLs + YAML")
  parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers (default: 16)")
  parser.add_argument("--safety-margin", type=float, default=0.10,
                      help="Support polygon shrink margin in metres (default: 0.10)")
  parser.add_argument("--no-capture-point", action="store_true",
                      help="Skip capture-point squat correction")
  parser.add_argument("--urdf", type=str, default="",
                      help="Path to G1 URDF for Pinocchio (auto-resolved if empty)")
  parser.add_argument("--method", type=str, choices=["lbfgsb", "cbf"], default="lbfgsb",
                      help="Correction solver: 'lbfgsb' (default, L-BFGS-B optimisation) "
                           "or 'cbf' (closed-form CBF projection)")
  parser.add_argument("--trigger", type=str, choices=["squat", "standing"], default="squat",
                      help="Which frames to correct: 'squat' (default, root height < 0.65m + "
                           "near-stationary) or 'standing' (both feet on ground + pelvis "
                           "near-stationary, any height)")
  args = parser.parse_args()

  with open(args.dataset) as f:
    config = yaml.safe_load(f)

  root_path = config["root_path"]
  motions = config["motions"]
  output_dir = Path(args.output_dir)
  xml_path = str(G1_XML)

  # Resolve URDF path
  urdf_path = args.urdf
  if not urdf_path:
    candidate = Path(__file__).resolve().parents[3] / "deploy" / "assets" / "g1_29dof_rev_1_0.urdf"
    if candidate.exists():
      urdf_path = str(candidate)
    else:
      urdf_path = ""
      if not args.no_capture_point:
        print(f"Warning: URDF not found at {candidate}, disabling capture-point correction")
        args.no_capture_point = True

  capture_point = not args.no_capture_point

  # Build work list
  work_items: list[tuple] = []
  for entry in motions:
    rel_file = entry["file"]
    input_path = os.path.join(root_path, rel_file)
    output_path = str(output_dir / rel_file)
    work_items.append((input_path, output_path, xml_path, args.safety_margin,
                       capture_point, urdf_path, args.method, args.trigger))

  print(f"Enriching {len(work_items)} PKL files → {output_dir}")
  if capture_point:
    print(f"  Capture-point correction: ON (safety_margin={args.safety_margin}m, "
          f"method={args.method}, trigger={args.trigger})")
  else:
    print(f"  Capture-point correction: OFF")

  # Process in parallel
  errors: list[str] = []
  n_workers = min(args.workers, len(work_items))
  ctx = mp.get_context("spawn")
  with ctx.Pool(processes=n_workers) as pool:
    for input_path, err in tqdm(
      pool.imap_unordered(_worker, work_items, chunksize=1),
      total=len(work_items),
      desc="Enriching PKLs",
    ):
      if err is not None:
        errors.append(err)

  if errors:
    print(f"\n{len(errors)} errors:")
    for e in errors[:20]:
      print(f"  {e}")
    if len(errors) > 20:
      print(f"  ... and {len(errors) - 20} more")

  # Write new YAML pointing to enriched PKLs
  new_config = {"root_path": str(output_dir), "motions": motions}
  output_yaml = output_dir / "dataset.yaml"
  output_dir.mkdir(parents=True, exist_ok=True)
  with open(output_yaml, "w") as f:
    yaml.dump(new_config, f, default_flow_style=False)

  n_ok = len(work_items) - len(errors)
  print(f"\nDone: {n_ok}/{len(work_items)} enriched. YAML: {output_yaml}")


if __name__ == "__main__":
  main()
