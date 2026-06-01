# HANDOFF

This is the framework for the HANDOFF paper (*Humanoid Agentic Task-Space
Whole-Body Control via Distilled Complementary Teachers*): the **fully-stable,
seed-dataset, AMP-augmented three-teacher Mixture-of-Experts student** for the
Unitree G1 (29 DoF) in MuJoCo. The student is one planner-friendly whole-body
controller that consumes a compact 10-D task-space command — planar base
velocity, root height, and bilateral pelvis-frame wrist targets
`[v_x, v_y, ω_z, z, p_L, p_R]` — and is distilled, under a context-conditioned
mixture-of-experts gating scheme, from three complementary specialists: a
whole-body motion-tracking teacher (trained on CoP-safety-filtered retargeted
clips), a locomotion teacher, and an AMP fall-recovery teacher.

The repository registers only four
tasks — the three teachers plus the MoE student — and the source keeps only the
term functions (observations / rewards / terminations / commands / events /
curricula / actions) those tasks actually reference.

## The four tasks

| Stage | Task id | Trains |
|------|---------|--------|
| Teacher 1 | `Wbc-Teacher-Flat-Unitree-G1-Stable` | stable WBC (whole-body-control) teacher |
| Teacher 2 | `Loco-Teacher-Flat-Unitree-G1-NoBV-Stable` | stable non-privileged (NoBV) locomotion teacher |
| Teacher 3 | `Amp-Teacher-Flat-Unitree-G1` | AMP recovery teacher |
| Student | `Wbc-Hand-Dual-Teacher-Flat-Unitree-G1-MoE-UniCmd-NoBV-AMP-Stable` | 3-teacher MoE student (split-KL distillation) |

"Stable" = whole-body stability rewards (CoM-in-support-polygon,
capture-point-in-support-polygon, ankle/hip/step, linear & angular momentum
change penalties) layered on the standard teacher reward mix. "Fully-stable" =
the student trains under those rewards **and** distills from teachers that were
themselves trained with them, so the KL targets are consistent with the env the
student sees.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) for package management.
- The **seed motion dataset** (an external artifact, not in this repo):
  `/home/yangl/handoff/seed_g1_cbf_standing_payload/seed_dataset_filtered.yaml`
  Every training script points at this file via `--env.commands.motion.motion-file`.
- A CUDA GPU. All scripts take the GPU id as the first positional arg (default 0)
  and forward everything else to `uv run train`.

```bash
uv sync     # install mjlab + torch (and transitive deps) into .venv
```

## Data: preprocessing & safety filtering

The `seed_dataset_filtered.yaml` that every training script consumes is not a
raw mocap dump — it is the output of a preprocessing + safety-filtering pipeline
that turns raw SEED G1 motion-capture CSVs into stability-corrected reference
clips. The pipeline lives in two pieces:

- `src/wbc_mjlab/scripts/seed_enrich.py` — the driver (CSV → enriched + filtered PKLs + YAMLs)
- `src/wbc_mjlab/stability_offline.py` — the offline capture-point / CoP correction library

```bash
WBC_ATTACH_PAYLOADS=1 uv run python -m wbc_mjlab.scripts.seed_enrich \
    --csv-dir ~/handoff/seed/g1/csv \
    --metadata ~/handoff/seed/seed_metadata_v003.csv \
    --output-dir ~/handoff/seed_g1_cbf_standing_payload \
    --fps 30 \
    --apply-correction --method cbf --trigger standing
```

`WBC_ATTACH_PAYLOADS=1` is **load-bearing for the correction**: the
capture-point / CoP solver builds its CoM from a URDF, and only with the env
var set does it use the payload-weighted `g1_29dof_rev_1_0_with_payloads.urdf`
(Jetson on the back + Dex1-1 hands) instead of the bare `g1_29dof_rev_1_0.urdf`.
Since every training task runs payloads-on, the filter must too — otherwise the
corrected clips would be statically stable for a robot that isn't the one being
trained. (Enrichment FK in step 1 is mass-independent, so the env var only
affects the correction in step 4.)

It runs in one pass and does four things:

1. **Enrichment.** Each CSV is loaded and MuJoCo forward kinematics is run on the
   G1 model to add the world-frame body fields HANDOFF's `MotionLib` expects
   (`local_body_pos`, `body_pos_w`, `body_quat_w`), then written back as an
   enriched PKL.
2. **Keyword pre-filter.** Clips whose name/category match `REJECT_KEYWORDS`
   (`dancing`, `stunts`, `injure`, `crutch`, `kneel`, `crawl`, `on_all_fours`)
   are dropped — these are motions a standing, payload-laden WBC G1 cannot or
   should not imitate. (`jump` clips bypass the height check below.)
3. **Quality post-filter.** After conversion, a clip is rejected if its root
   height leaves `[0.4, 0.9] m` or its torso tilt ever exceeds `100°` — i.e. the
   subject went to the ground or inverted.
4. **CoP / capture-point safety correction** (`--apply-correction`). For every
   quasi-static "standing" frame the solver runs Pinocchio FK, builds a
   conservatively shrunk support polygon from the foot-contact corners, and
   computes the signed distance `h` between the CoM projection and the polygon
   boundary. Where `h` falls below a safety target it solves a minimum-effort
   correction over a 7-DoF subspace (hip pitch, ankle pitch/roll, waist pitch)
   that restores `h ≥ h_tgt` while **re-anchoring the floating base so foot
   placements are preserved**. Two interchangeable solvers are available
   (`--method {lbfgsb,cbf}`); the CBF one reuses the same analytic `∇h` and
   contact-consistent CoM Jacobian as the online training/deploy-time CBFs. A
   temporal blending mask smooths the boundaries of corrected segments.

Two YAMLs are emitted: `seed_dataset.yaml` (unfiltered — all converted clips)
and `seed_dataset_filtered.yaml` (passed both filters, correction applied). The
latter is the one the training scripts point at.

To apply the same correction to an *existing* HANDOFF YAML dataset (PKLs that are
already enriched) rather than raw CSVs, use the sibling driver:

```bash
uv run python -m wbc_mjlab.scripts.enrich_pkl \
    --dataset /path/to/handoff_dataset.yaml \
    --output-dir /path/to/enriched/ \
    --method cbf --trigger standing
```

## Training pipeline (run in order)

Sim-to-real hardware payloads (Jetson on the back + Dex1-1 hands) are attached
to every task by default; set `WBC_ATTACH_PAYLOADS=0` to disable.

```bash
# 1. Stable WBC teacher
./train_wbc_teacher_stable_seed.sh <GPU>
#    -> logs/rsl_rl/g1_wbc_teacher_stable_seed/<run>/

# 2. Stable NoBV locomotion teacher
./train_loco_teacher_nobv_stable_seed.sh <GPU>
#    -> logs/rsl_rl/g1_loco_teacher_nobv_stable_seed/<run>/

# 3. AMP recovery teacher (no -Stable variant exists; the AMP reward stack is
#    independent of the stability rewards)
./train_amp_teacher_flat.sh <GPU>
#    -> logs/rsl_rl/g1_amp_teacher_flat/<run>/

# 4. Point the student at the teacher checkpoints, then train it.
#    Edit ckpt_setup_stable_seed.sh and fill WBC_TEACHER_EXP / LOCO_TEACHER_NOBV_EXP
#    with the run-dir prefixes ("<date>_<time>_<run-name>") produced by steps 1-2.
#    (The AMP-teacher slot defaults are in ckpt_setup_stable_seed_amp.sh.)
./train_moe_student_unicmd_nobv_fullstable_seed_amp.sh <GPU>
#    -> logs/rsl_rl/g1_hand_moe_flat_unicmd_nobv_fullstable_seed_amp/<run>/
```

### Checkpoint wiring

`train_moe_student_unicmd_nobv_fullstable_seed_amp.sh` sources
`ckpt_setup_stable_seed_amp.sh`, which sources `ckpt_setup_stable_seed.sh`:

```
train_moe_student_unicmd_nobv_fullstable_seed_amp.sh
└─ ckpt_setup_stable_seed_amp.sh   (fills AMP_TEACHER_* slot)
   └─ ckpt_setup_stable_seed.sh    (fills WBC_TEACHER_* + LOCO_TEACHER_NOBV_* slots)
```

You only ever *run* the four `train_*.sh` scripts; you *edit* the timestamps in
`ckpt_setup_stable_seed.sh` once between steps 3 and 4.

> Note: hand-student runs degrade past ~20k iters. Scripts may run longer for
> visibility, but export from a deliberately picked checkpoint (e.g.
> `model_20000.pt`), not `model_-1`.

## Evaluating a checkpoint

Use mjlab's built-in native-viewer playback (no extra scripts needed):

```bash
uv run play Wbc-Hand-Dual-Teacher-Flat-Unitree-G1-MoE-UniCmd-NoBV-AMP-Stable \
  --checkpoint logs/rsl_rl/g1_hand_moe_flat_unicmd_nobv_fullstable_seed_amp/<run>/model_20000.pt
```

The same works for any of the four task ids.

## Deployment (sim2sim & sim2real)

Beyond the native-viewer playback above, the student policy can be driven
through the runtime stack under `deploy/` — the same ONNX policy node talking to
either MuJoCo (sim2sim) or the physical Unitree G1 (sim2real), with an xbox pad
or a VR headset on the command bus. Both flows export ONNX from the latest
`g1_hand_moe_flat_unicmd_nobv_fullstable_seed_amp` run automatically (or take an
explicit `.pt` / `.onnx` path), then launch the policy, the sim/hardware node,
and the chosen controller.

Both scripts require **ROS 2** (any of humble/iron/jazzy/rolling — auto-detected
and sourced) for the controller↔policy command topics; the high-rate
policy↔sim/hardware loop is UDP. Run `uv sync` first so the deploy deps
(`onnxruntime`, `inputs`, `pin`, `televuer`, `websockets`) are present.

### sim2sim (MuJoCo)

```bash
bash deploy/play_sim_hand.sh                 # latest run, prompts for controller + run mode
bash deploy/play_sim_hand.sh model_20000.pt  # auto-export a specific checkpoint, then run
bash deploy/play_sim_hand.sh model.onnx      # run a pre-exported ONNX
```

Controller prompt: `none`, `keyboard`, `xbox` (local pad), `vr` (dds_xr headset
teleop), or `xbox server` (browser gamepad at `http://localhost:8765`). Run-mode
prompt: `sync` (sim+policy in one physics-synchronized process, default) or
`split` (policy and sim as separate UDP processes).

### sim2real (physical G1)

```bash
bash deploy/play_real_hand.sh                # latest run, prompts for net iface + grippers + controller
bash deploy/play_real_hand.sh model_20000.pt
```

One-time hardware setup builds the vendored Unitree SDK Python binding and the
Dex1-1 gripper service (both pinned git submodules):

```bash
git submodule update --init deploy/real/unitree_sdk2_wrapper deploy/real/dex1_1_service
bash deploy/install_unitree_sdk.sh           # builds + installs unitree_interface .so and dex1_1_gripper_server
```

The script sets up policy-network routing (`deploy/real/setup_route.sh`), prompts
for the robot DDS interface (default `enP2p1s0`) and whether Dex1-1 grippers are
connected, then launches the gripper service (if present), the hand policy, the
hardware node, and the chosen controller (`none` / `keyboard` / `xbox` / `vr`).

> Always verify a checkpoint in sim2sim before running it on hardware — a wrong
> obs/action layout shows up as a soft glitch in sim but a fall on the real G1.

## Repo layout

```
src/wbc_mjlab/
  __init__.py            # registers the 4 tasks (single source of truth)
  config.py              # env configs for the WBC / loco / student tasks
  amp_config.py          # env + runner config for the AMP teacher
  rl_cfg.py              # RSL-RL runner configs
  observations.py rewards.py terminations.py commands.py
  actions.py events.py curriculums.py amp_mdp.py   # MDP term functions
  g1_constants_custom.py # G1 PD gains, action scales, payload specs
  pkl_motion_lib.py      # motion-clip loader
  stability_offline.py   # offline CoP/capture-point safety-correction solvers
  scripts/               # data prep: seed_enrich.py (CSV->filtered), enrich_pkl.py
  rl/                    # DAgger + AMP runners, PPO algorithms, MoE network
deploy/                  # runtime stack for sim2sim / sim2real of the student
  play_sim_hand.sh       # sim2sim playback (xbox / vr / keyboard)
  play_real_hand.sh      # sim2real deployment on the physical G1
  install_unitree_sdk.sh # builds the vendored SDK + Dex1-1 gripper service
  export_onnx.py         # checkpoint .pt -> actor ONNX
  policy/hand_policy.py  # ONNX policy node (UDP loop)
  sim/                   # MuJoCo nodes: sim_node.py, sim_policy_node.py (sync)
  real/                  # hardware_node.py + unitree_sdk2_wrapper, dex1_1_service (submodules)
  controller/            # xbox_node.py, dds_xr_node.py (vr), keyboard_node.py
  common/command.py      # G1 command-vector layout (imported by g1_constants_custom)
  assets/                # G1 MuJoCo XML + URDF + meshes
```

`src/wbc_mjlab/__init__.py` is the entry index: mjlab discovers tasks via the
`mjlab.tasks` entry point in `pyproject.toml`. Training/playback go through
mjlab's `uv run train` / `uv run play` CLI.

## Citation

If you use this code or build on HANDOFF, please cite:

```bibtex
@article{yang2026handoff,
  title   = {HANDOFF: Humanoid Agentic Task-Space Whole-Body Control via Distilled Complementary Teachers},
  author  = {Yang, Lizhi and Li, Junheng and Poddar, Nehar and Hou, Yiling and Huh, Gio and Griffin, Robert and Gkioxari, Georgia and Ames, Aaron D.},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026},
  note    = {arXiv ID to be added}
}
```
