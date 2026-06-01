"""Convert SEED G1 CSVs to enriched HANDOFF PKL format in a single pass.

Reads CSV motion files, runs MuJoCo FK to compute local_body_pos,
body_pos_w, and body_quat_w, writes enriched PKLs, and generates
dataset YAML files (unfiltered + quality-filtered) with metadata descriptions.

Optionally applies the offline capture-point correction from
``wbc_mjlab.stability_offline`` to every filtered PKL after conversion, so
the emitted filtered dataset is safety-ready in one pipeline pass.

Usage:
    uv run python -m wbc_mjlab.scripts.seed_enrich \
        --csv-dir ~/handoff/seed/g1/csv \
        --metadata ~/handoff/seed/seed_metadata_v003.csv \
        --output-dir ~/handoff/seed_g1_enriched_pkl \
        --fps 30 \
        --apply-correction --method cbf --trigger standing
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
import mujoco as _mujoco

mujoco = cast(Any, _mujoco)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 38-body list expected by HANDOFF MotionLib
MUJOCO_BODY_NAMES_38 = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "left_toe_link",              # synthetic
    "pelvis_contour_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "right_toe_link",             # synthetic
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "head_link",                  # synthetic
    "head_mocap",                 # synthetic
    "imu_in_torso",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
]

# Offsets for 4 synthetic bodies
SYNTHETIC_BODY_OFFSETS: Dict[int, Tuple[int, np.ndarray]] = {
    7:  (6,  np.array([0.0927,  0.0175, -0.0191], dtype=np.float32)),
    15: (14, np.array([0.0925, -0.0036, -0.0225], dtype=np.float32)),
    19: (18, np.array([-0.0090, -0.0006, -0.0427], dtype=np.float32)),
    20: (18, np.array([0.0000,  0.0059,  0.3774], dtype=np.float32)),
}

DOF_COLUMNS = [
    "left_hip_pitch_joint_dof",
    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_pitch_joint_dof",
    "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof",
    "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_pitch_joint_dof",
    "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof",
    "waist_roll_joint_dof",
    "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof",
    "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof",
    "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",
    "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
]

REJECT_KEYWORDS = ["dancing", "stunts", "injure", "crutch", "kneel", "crawl", "on_all_fours"]
BYPASS_HEIGHT_KEYWORDS = ["jump"]

DEFAULT_MIN_ROOT_HEIGHT = 0.4
DEFAULT_MAX_ROOT_HEIGHT = 0.9
DEFAULT_MAX_TILT_DEG = 100.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def should_skip_csv(
    csv_path: Path,
    meta: Dict,
    reject_keywords: List[str],
) -> Optional[str]:
    """Pre-filter before conversion based on keyword matching."""
    name_lower = csv_path.stem.lower()
    cat_lower = meta.get("category", "").lower()
    for kw in reject_keywords:
        if kw in name_lower or kw in cat_lower:
            return f"keyword '{kw}'"
    return None


def check_motion_quality(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    min_height: float,
    max_height: float,
    max_tilt_deg: float,
    skip_height: bool = False,
) -> Optional[str]:
    """Post-conversion quality filter on root height and torso orientation."""
    if not skip_height:
        z = root_pos[:, 2]
        if z.min() < min_height:
            return f"root_z_min={z.min():.3f} < {min_height}"
        if z.max() > max_height:
            return f"root_z_max={z.max():.3f} > {max_height}"

    rots = Rotation.from_quat(root_rot)  # xyzw
    up_world = rots.apply(np.array([0.0, 0.0, 1.0]))
    cos_tilt = np.clip(up_world[:, 2], -1.0, 1.0)
    tilt_deg = np.rad2deg(np.arccos(cos_tilt))
    if tilt_deg.max() > max_tilt_deg:
        return f"max_tilt={tilt_deg.max():.1f}° > {max_tilt_deg}°"

    return None


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def build_body_mapping(model: Any) -> Dict[int, int]:
    """Map 38-body list indices to MuJoCo model body IDs (real bodies only)."""
    name_to_id: Dict[str, int] = {}
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name:
            name_to_id[name] = bid

    mapping = {}
    for idx, body_name in enumerate(MUJOCO_BODY_NAMES_38):
        if body_name in name_to_id:
            mapping[idx] = name_to_id[body_name]
    return mapping


def get_description(meta: Dict, fallback_stem: str) -> str:
    """Build a single description string from metadata."""
    desc = meta.get("content_natural_desc_1", "")
    if isinstance(desc, str) and desc.strip():
        return desc.strip()
    desc = meta.get("category", "")
    if isinstance(desc, str) and desc.strip():
        return desc.strip()
    return fallback_stem.split("__")[0].replace("_", " ")


# ---------------------------------------------------------------------------
# Worker process globals (loaded once per worker via Pool initializer)
# ---------------------------------------------------------------------------

_worker_model: Any = None
_worker_data: Any = None
_worker_body_mapping: Dict[int, int] = {}


def _init_worker(xml_path: str) -> None:
    global _worker_model, _worker_data, _worker_body_mapping
    _worker_model = mujoco.MjModel.from_xml_path(xml_path)
    _worker_data = mujoco.MjData(_worker_model)
    _worker_body_mapping = build_body_mapping(_worker_model)


# ---------------------------------------------------------------------------
# Core conversion + enrichment
# ---------------------------------------------------------------------------


def convert_and_enrich_one_csv(
    csv_path: Path,
    output_path: Path,
    fps: int,
    native_fps: int = 120,
) -> Optional[str]:
    """Convert a single SEED CSV to an enriched HANDOFF PKL. Returns error or None."""
    try:
        # CSV columns: Frame(0), root_tX(1), root_tY(2), root_tZ(3),
        #              root_rX(4), root_rY(5), root_rZ(6), 29 DOFs(7..35)
        raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)

        # Root position: cm → meters
        root_pos = raw[:, 1:4] / 100.0

        # Root rotation: Euler degrees → xyzw quaternion
        root_rot = Rotation.from_euler("xyz", raw[:, 4:7], degrees=True).as_quat().astype(np.float64)

        # Joint angles: degrees → radians
        dof_pos = np.deg2rad(raw[:, 7:])

        T = root_pos.shape[0]

        # Subsample frames when target fps differs from native capture rate so
        # that the stored fps label correctly describes the frame spacing.
        if fps != native_fps and native_fps > 0:
            T_new = max(2, round(T * fps / native_fps))
            indices = np.round(np.linspace(0, T - 1, T_new)).astype(int)
            root_pos = root_pos[indices]
            root_rot = root_rot[indices]
            dof_pos  = dof_pos[indices]
            T = root_pos.shape[0]
        n_bodies = len(MUJOCO_BODY_NAMES_38)

        model = _worker_model
        data = _worker_data
        body_mapping = _worker_body_mapping

        local_body_pos = np.zeros((T, n_bodies, 3), dtype=np.float32)
        body_pos_w = np.zeros((T, n_bodies, 3), dtype=np.float32)
        body_quat_w = np.zeros((T, n_bodies, 4), dtype=np.float32)
        body_quat_w[:, :, 0] = 1.0  # identity quaternion [w,x,y,z]

        root_rot_wxyz = xyzw_to_wxyz(root_rot)

        for t in range(T):
            data.qpos[:3] = root_pos[t]
            data.qpos[3:7] = root_rot_wxyz[t]
            data.qpos[7:] = dof_pos[t]
            mujoco.mj_kinematics(model, data)

            for idx_38, bid in body_mapping.items():
                local_body_pos[t, idx_38] = data.xpos[bid] - root_pos[t]
                body_pos_w[t, idx_38] = data.xpos[bid]
                body_quat_w[t, idx_38] = data.xquat[bid]

            for syn_idx, (src_idx, offset) in SYNTHETIC_BODY_OFFSETS.items():
                local_body_pos[t, syn_idx] = local_body_pos[t, src_idx] + offset

        motion_dict = {
            "fps": fps,
            "root_pos": root_pos.astype(np.float32),
            "root_rot": root_rot.astype(np.float32),
            "dof_pos": dof_pos.astype(np.float32),
            "local_body_pos": local_body_pos,
            "link_body_list": MUJOCO_BODY_NAMES_38,
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(motion_dict, f)

        return None

    except Exception as e:
        return f"{csv_path.name}: {e}"


def _worker_convert(args: Tuple[Path, Path, int, int]) -> Optional[str]:
    return convert_and_enrich_one_csv(*args)


# ---------------------------------------------------------------------------
# Post-conversion CBF / L-BFGS-B correction pass
# ---------------------------------------------------------------------------

def _correction_worker(args: Tuple[Path, str, float, str, str, str]) -> Optional[str]:
    """Apply stability_offline.correct_capture_point to a single PKL in place."""
    pkl_path, urdf_path, safety_margin, xml_path, method, trigger = args
    try:
        from wbc_mjlab.stability_offline import correct_capture_point
        with open(pkl_path, "rb") as f:
            motion = pickle.load(f)
        correct_capture_point(
            motion, urdf_path, safety_margin, xml_path,
            method=method, trigger=trigger,
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(motion, f)
        return None
    except Exception as e:
        return f"{pkl_path.name}: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SEED G1 CSVs to enriched HANDOFF PKLs (single-pass)"
    )
    parser.add_argument("--csv-dir", type=Path, default=Path.home() / "handoff/seed/g1/csv")
    parser.add_argument("--metadata", type=Path, default=Path.home() / "handoff/seed/seed_metadata_v003.csv")
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "handoff/seed_g1_enriched_pkl")
    parser.add_argument("--output-yaml", type=Path, default=None,
                        help="Unfiltered YAML (default: {output-dir}/seed_dataset.yaml)")
    parser.add_argument("--filtered-yaml", type=Path, default=None,
                        help="Filtered YAML (default: {output-dir}/seed_dataset_filtered.yaml)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--native-fps", type=int, default=120,
                        help="Capture rate of the source CSVs (default 120 Hz for SEED G1)")
    parser.add_argument("--max-files", type=int, default=-1, help="-1 for all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Overwrite existing PKLs")
    parser.add_argument("--min-height", type=float, default=DEFAULT_MIN_ROOT_HEIGHT)
    parser.add_argument("--max-height", type=float, default=DEFAULT_MAX_ROOT_HEIGHT)
    parser.add_argument("--max-tilt", type=float, default=DEFAULT_MAX_TILT_DEG)

    # Offline stability correction (applied only to filtered entries).
    parser.add_argument("--apply-correction", action="store_true",
                        help="After YAMLs are written, run stability correction in place on "
                             "every PKL in the filtered dataset.")
    parser.add_argument("--method", type=str, choices=["lbfgsb", "cbf"], default="cbf",
                        help="Correction solver (used with --apply-correction). Default: cbf")
    parser.add_argument("--trigger", type=str, choices=["squat", "standing"], default="standing",
                        help="Eligibility trigger (used with --apply-correction). Default: standing")
    parser.add_argument("--safety-margin", type=float, default=0.10,
                        help="Support polygon shrink margin in metres (default: 0.10)")
    parser.add_argument("--urdf", type=str, default="",
                        help="Path to G1 URDF for Pinocchio (auto-resolved if empty)")

    args = parser.parse_args()

    if args.output_yaml is None:
        args.output_yaml = args.output_dir / "seed_dataset.yaml"
    if args.filtered_yaml is None:
        args.filtered_yaml = args.output_dir / "seed_dataset_filtered.yaml"

    xml_path = str(G1_XML)

    # Resolve URDF path. When ``WBC_ATTACH_PAYLOADS`` is truthy, prefer the
    # weighted variant so Pinocchio CoM / capture-point uses the deployed
    # hardware masses (Jetson on back + Dex1-1 hands).
    from wbc_mjlab.g1_constants_custom import ATTACH_PAYLOADS  # noqa: PLC0415
    assets_dir = Path(__file__).resolve().parents[3] / "deploy" / "assets"
    urdf_path = args.urdf
    if not urdf_path:
        if ATTACH_PAYLOADS:
            candidate = assets_dir / "g1_29dof_rev_1_0_with_payloads.urdf"
            print(f"WBC_ATTACH_PAYLOADS set: using weighted URDF {candidate.name}")
        else:
            candidate = assets_dir / "g1_29dof_rev_1_0.urdf"
        if candidate.exists():
            urdf_path = str(candidate)
        elif args.apply_correction:
            print(f"Warning: URDF not found at {candidate}, disabling --apply-correction")
            args.apply_correction = False

    # Load metadata, indexed by move_g1_path
    meta_lookup: Dict[str, Dict] = {}
    if args.metadata.exists():
        with open(args.metadata, newline="", encoding="utf-8") as mf:
            reader = csv.DictReader(mf)
            for row in reader:
                key = row.get("move_g1_path", "")
                if key:
                    meta_lookup[key] = dict(row)
        print(f"Loaded metadata for {len(meta_lookup)} motions")

    # Collect and optionally limit CSV files
    csv_files = sorted(args.csv_dir.rglob("*.csv"))
    if args.max_files > 0:
        csv_files = csv_files[: args.max_files]
    print(f"Found {len(csv_files)} CSV files")

    # Build task list, skipping existing PKLs unless --force
    tasks: List[Tuple[Path, Path, int, int]] = []
    for csv_path in csv_files:
        rel = csv_path.relative_to(args.csv_dir)
        out_path = args.output_dir / rel.with_suffix(".pkl")
        if out_path.exists() and not args.force:
            continue
        tasks.append((csv_path, out_path, args.fps, args.native_fps))

    print(f"Converting {len(tasks)} files ({len(csv_files) - len(tasks)} already exist, skipped)")

    # Convert with multiprocessing
    errors: List[str] = []
    if tasks:
        n_workers = min(args.workers, len(tasks))
        with Pool(processes=n_workers, initializer=_init_worker, initargs=(xml_path,)) as pool:
            for result in tqdm(
                pool.imap_unordered(_worker_convert, tasks, chunksize=32),
                total=len(tasks),
                desc="Converting",
            ):
                if result is not None:
                    errors.append(result)
    else:
        print("No new files to convert — skipping to YAML generation.")

    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    # -----------------------------------------------------------------------
    # Generate YAML configs (unfiltered + filtered)
    # -----------------------------------------------------------------------
    all_pkls = sorted(args.output_dir.rglob("*.pkl"))
    all_entries: List[Dict] = []
    filtered_entries: List[Dict] = []
    filter_rejected: List[Tuple[str, str]] = []

    for pkl_path in all_pkls:
        rel = pkl_path.relative_to(args.output_dir)
        rel_csv = "g1/csv/" + str(rel.with_suffix(".csv"))
        meta = meta_lookup.get(rel_csv, {})
        desc = get_description(meta, pkl_path.stem)

        try:
            with pkl_path.open("rb") as f:
                motion = pickle.load(f)
        except Exception as e:
            filter_rejected.append((str(rel), f"load error: {e}"))
            continue

        entry = {
            "file": str(rel),
            "weight": 1.0,
            "description": desc,
            "fps": int(motion["fps"]),
            "num_frames": int(motion["root_pos"].shape[0]),
        }
        all_entries.append(entry)

        # Keyword pre-filter
        skip = should_skip_csv(pkl_path, meta, REJECT_KEYWORDS)
        if skip:
            filter_rejected.append((str(rel), skip))
            continue

        # Quality post-filter
        try:
            name_lower = pkl_path.stem.lower()
            cat_lower = meta.get("category", "").lower()
            bypass_height = any(kw in name_lower or kw in cat_lower for kw in BYPASS_HEIGHT_KEYWORDS)
            reason = check_motion_quality(
                motion["root_pos"], motion["root_rot"],
                args.min_height, args.max_height, args.max_tilt,
                skip_height=bypass_height,
            )
            if reason:
                filter_rejected.append((str(rel), reason))
                continue
        except Exception as e:
            filter_rejected.append((str(rel), f"quality check error: {e}"))
            continue

        filtered_entries.append(entry)

    # Write unfiltered YAML
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating unfiltered YAML at {args.output_yaml}")
    yaml_config = {"root_path": str(args.output_dir), "motions": all_entries}
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with args.output_yaml.open("w") as f:
        yaml.safe_dump(yaml_config, f, sort_keys=False)

    # Write filtered YAML
    print(f"Generating filtered YAML at {args.filtered_yaml}")
    filtered_config = {"root_path": str(args.output_dir), "motions": filtered_entries}
    args.filtered_yaml.parent.mkdir(parents=True, exist_ok=True)
    with args.filtered_yaml.open("w") as f:
        yaml.safe_dump(filtered_config, f, sort_keys=False)

    print(f"\nConverted: {len(all_entries)} total PKLs, {len(filtered_entries)} passed filter, "
          f"{len(filter_rejected)} rejected, {len(errors)} conversion errors.")
    if filter_rejected:
        print(f"\nSample rejections:")
        for name, reason in filter_rejected[:15]:
            print(f"  {name}: {reason}")

    # -----------------------------------------------------------------------
    # Optional: stability correction pass on the filtered entries
    # -----------------------------------------------------------------------
    if args.apply_correction and filtered_entries:
        print(f"\nApplying stability correction to {len(filtered_entries)} filtered PKLs "
              f"(method={args.method}, trigger={args.trigger}, "
              f"safety_margin={args.safety_margin}m)...")
        correction_tasks = [
            (args.output_dir / entry["file"], urdf_path, args.safety_margin,
             xml_path, args.method, args.trigger)
            for entry in filtered_entries
        ]
        correction_errors: List[str] = []
        n_workers = min(args.workers, len(correction_tasks))
        with Pool(processes=n_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_correction_worker, correction_tasks, chunksize=1),
                total=len(correction_tasks),
                desc="Correcting",
            ):
                if result is not None:
                    correction_errors.append(result)
        if correction_errors:
            print(f"\n{len(correction_errors)} correction errors:")
            for e in correction_errors[:20]:
                print(f"  {e}")
            if len(correction_errors) > 20:
                print(f"  ... and {len(correction_errors) - 20} more")
        print(f"Corrected {len(correction_tasks) - len(correction_errors)}/"
              f"{len(correction_tasks)} filtered PKLs.")


if __name__ == "__main__":
    main()
