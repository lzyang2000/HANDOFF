"""Offline capture-point stability correction — shared library.

Provides the two interchangeable solvers that restore the static CoP into a
conservatively shrunk support polygon on quasi-static mocap frames:

  * ``_correct_single_frame`` (L-BFGS-B nonlinear least squares)
  * ``_cbf_correct_frame``   (position-space CBF Newton projection with
                              line search, smoothness clamp, warm-start)

Both solvers share the same Pinocchio-based FK helper, floating-base
re-anchoring, foot-contact geometry, eligibility detection, and temporal
blending.  The public entry point is ``correct_capture_point``.

Used by both ``scripts/enrich_pkl.py`` (PKL → corrected PKL) and
``scripts/seed_enrich.py`` (CSV → corrected PKL).
"""

from __future__ import annotations

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Detection / target / blending constants
# ---------------------------------------------------------------------------

_SQUAT_HEIGHT_THRESH: float = 0.65
_SQUAT_SPEED_THRESH: float = 0.2
_SQUAT_MIN_CONSECUTIVE: int = 15
_H_TARGET: float = 0.04
_BLEND_FRAMES: int = 10

# Standing-mode detection parameters (used when trigger="standing").
_STAND_SPEED_THRESH: float = 0.2       # m/s pelvis XY speed (same as squat)
_STAND_FOOT_Z_THRESH: float = 0.08     # m — max foot z above estimated ground
_STAND_FEET_Z_DIFF_THRESH: float = 0.05  # m — max L-R foot z difference
_STAND_MIN_CONSECUTIVE: int = 8        # shorter than squat (standing can be brief)

# ---------------------------------------------------------------------------
# G1 foot contact geometry
# ---------------------------------------------------------------------------

_FOOT_CONTACTS_LOCAL = np.array(
  [[-0.05, 0.025, -0.03], [-0.05, -0.025, -0.03],
   [0.12, 0.03, -0.03], [0.12, -0.03, -0.03]], dtype=np.float64,
)

# ---------------------------------------------------------------------------
# L-BFGS-B solver constants
# ---------------------------------------------------------------------------

_HIP_PITCH_IDX = [0, 6]
_ANKLE_PITCH_IDX = [4, 10]
_ANKLE_ROLL_IDX = [5, 11]
_WAIST_PITCH_IDX = 14

_OPT_BOUNDS = [
  (-0.6, 0.6), (-0.6, 0.6),    # hip pitch L/R
  (-0.5, 0.5), (-0.5, 0.5),    # ankle pitch L/R
  (-0.3, 0.3), (-0.3, 0.3),    # ankle roll L/R
  (-0.8, 0.8),                  # waist pitch
]
_REG_WEIGHTS = np.array([1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 3.0])
_TEMPORAL_SMOOTH_WEIGHT = 2.0

# ---------------------------------------------------------------------------
# CBF solver constants
# ---------------------------------------------------------------------------

# Offline uses 7 DOF (hip pitch + ankle pitch/roll + waist pitch) — same as
# the L-BFGS-B optimizer — because severe violations need hip pitch authority.
# α = 1.0 (per-step discrete-time rate) means "reach h_target in one step".
# The training/deploy CBFs use α = 0.5 (halfway each step) — that's appropriate
# for continuous safety but too slow for offline per-frame feasibility.
_CBF_ALPHA = 1.0
_CBF_ADJ_INDICES = np.array([0, 4, 5, 6, 10, 11, 14], dtype=np.intp)
#  0 = left_hip_pitch,   4 = left_ankle_pitch,   5 = left_ankle_roll,
#  6 = right_hip_pitch, 10 = right_ankle_pitch, 11 = right_ankle_roll,
# 14 = waist_pitch
_CBF_NON_ADJ_MASK = ~np.isin(np.arange(29), _CBF_ADJ_INDICES)
# Per-iteration step cap — generous because Newton line search handles overshoot.
_CBF_VEL_LIMITS = np.array([
  30.0,  # hip pitch L
  30.0,  # ankle pitch L
  30.0,  # ankle roll L
  30.0,  # hip pitch R
  30.0,  # ankle pitch R
  30.0,  # ankle roll R
  30.0,  # waist pitch
])
# Frame-to-frame smoothness cap — bounds how much each adjustable joint can
# change from the previous corrected frame. Prevents jitter when consecutive
# frames converge to different local CBF solutions.
_CBF_SMOOTH_VEL_LIMIT = 1.0  # rad/s — corresponds to 0.033 rad per frame at 30 Hz
_CBF_MAX_REFINE_ITERS = 20
_CBF_LINE_SEARCH_ITERS = 6  # backtracking line search depth per iteration
_GRAVITY = 9.81

# ---------------------------------------------------------------------------
# Policy joint names (needed for Pinocchio index mapping)
# ---------------------------------------------------------------------------

_POLICY_JOINT_NAMES = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


# ---------------------------------------------------------------------------
# Pinocchio helper (shared by both solvers and by standing detection)
# ---------------------------------------------------------------------------

class PinocchioHelper:
  """Thin wrapper caching Pinocchio model + index maps for repeated FK calls."""

  def __init__(self, urdf_path: str):
    import pinocchio as pin
    self._pin = pin
    self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    self.data = self.model.createData()
    self.q = pin.neutral(self.model)
    self.q_indices = np.array(
      [int(self.model.joints[self.model.getJointId(n)].idx_q) for n in _POLICY_JOINT_NAMES],
      dtype=np.int32,
    )
    self.lf_id = self.model.getFrameId("left_ankle_roll_link")
    self.rf_id = self.model.getFrameId("right_ankle_roll_link")

  def fk(self, rp: np.ndarray, rq_xyzw: np.ndarray, jp: np.ndarray) -> None:
    pin = self._pin
    self.q[:] = pin.neutral(self.model)
    self.q[0:3] = rp
    self.q[3:7] = rq_xyzw / np.linalg.norm(rq_xyzw)
    self.q[self.q_indices] = jp
    pin.forwardKinematics(self.model, self.data, self.q)
    pin.updateFramePlacements(self.model, self.data)
    pin.centerOfMass(self.model, self.data, self.q)

  def feet_mid_se3(self):
    from scipy.spatial.transform import Rotation as R, Slerp
    pL = self.data.oMf[self.lf_id].translation.copy()
    pR = self.data.oMf[self.rf_id].translation.copy()
    RL = self.data.oMf[self.lf_id].rotation
    RR = self.data.oMf[self.rf_id].rotation
    slerp = Slerp([0, 1], R.concatenate([R.from_matrix(RL), R.from_matrix(RR)]))
    return (pL + pR) / 2.0, slerp(0.5).as_matrix()

  def reanchor_root(self, rp, rq, jp_orig, jp_new):
    from scipy.spatial.transform import Rotation as R
    self.fk(rp, rq, jp_orig); mo, ro = self.feet_mid_se3()
    self.fk(rp, rq, jp_new); mn, rn = self.feet_mid_se3()
    Rc = ro @ rn.T; tc = mo - Rc @ mn
    rp_n = Rc @ rp + tc
    rq_n = (R.from_matrix(Rc) * R.from_quat(rq)).as_quat()
    if np.dot(rq_n, rq) < 0:
      rq_n = -rq_n
    return rp_n, rq_n

  def h_static(self, rp, rq, jp, safety_margin):
    self.fk(rp, rq, jp)
    com = np.asarray(self.data.com[0])
    fL = self.data.oMf[self.lf_id]; fR = self.data.oMf[self.rf_id]
    cL = (fL.rotation @ _FOOT_CONTACTS_LOCAL.T).T + fL.translation
    cR = (fR.rotation @ _FOOT_CONTACTS_LOCAL.T).T + fR.translation
    ac = np.vstack([cL, cR])
    return min(
      com[0] - ac[:, 0].min() - safety_margin,
      ac[:, 0].max() - safety_margin - com[0],
      com[1] - ac[:, 1].min() - safety_margin,
      ac[:, 1].max() - safety_margin - com[1],
    )


# ---------------------------------------------------------------------------
# MuJoCo body-transform recomputation (used by both enrichment scripts)
# ---------------------------------------------------------------------------

def build_body_name_map(
  model: mujoco.MjModel, pkl_link_body_list: list[str]
) -> dict[int, int]:
  """Map PKL link_body_list indices → MuJoCo body indices, by name."""
  mj_name_to_idx: dict[str, int] = {}
  for i in range(model.nbody):
    mj_name_to_idx[model.body(i).name] = i

  pkl_to_mj: dict[int, int] = {}
  for pkl_idx, name in enumerate(pkl_link_body_list):
    if name in mj_name_to_idx:
      pkl_to_mj[pkl_idx] = mj_name_to_idx[name]
  return pkl_to_mj


def recompute_body_transforms_mujoco(
  enriched: dict,
  xml_path: str,
) -> None:
  """Recompute body_pos_w and body_quat_w from (possibly corrected) root/dof data."""
  root_pos = np.asarray(enriched["root_pos"])
  root_rot = np.asarray(enriched["root_rot"])  # xyzw
  dof_pos = np.asarray(enriched["dof_pos"])
  link_body_list = enriched["link_body_list"]
  T = root_pos.shape[0]

  root_rot_wxyz = root_rot[:, [3, 0, 1, 2]]

  model = mujoco.MjModel.from_xml_path(xml_path)
  mj_data = mujoco.MjData(model)
  pkl_to_mj = build_body_name_map(model, link_body_list)

  n_bodies = len(link_body_list)
  body_pos_w = np.zeros((T, n_bodies, 3), dtype=np.float32)
  body_quat_w = np.zeros((T, n_bodies, 4), dtype=np.float32)
  body_quat_w[:, :, 0] = 1.0  # identity wxyz

  for t in range(T):
    mj_data.qpos[:3] = root_pos[t]
    mj_data.qpos[3:7] = root_rot_wxyz[t]
    mj_data.qpos[7:] = dof_pos[t]
    mujoco.mj_kinematics(model, mj_data)
    for pkl_idx, mj_idx in pkl_to_mj.items():
      body_pos_w[t, pkl_idx] = mj_data.xpos[mj_idx]
      body_quat_w[t, pkl_idx] = mj_data.xquat[mj_idx]

  enriched["body_pos_w"] = body_pos_w
  enriched["body_quat_w"] = body_quat_w


# ---------------------------------------------------------------------------
# Eligibility detectors
# ---------------------------------------------------------------------------

def needs_squat_correction(data: dict) -> bool:
  """Cheap root-height fast-path: skip PKLs with no sustained squat segment."""
  root_pos = np.asarray(data["root_pos"])
  T = root_pos.shape[0]
  if T < _SQUAT_MIN_CONSECUTIVE + 2:
    return False
  fps = float(data.get("fps", 30.0))
  vel = np.zeros_like(root_pos)
  vel[1:] = np.diff(root_pos, axis=0) * fps
  speed = np.linalg.norm(vel[:, :2], axis=1)
  is_sq = (root_pos[:, 2] < _SQUAT_HEIGHT_THRESH) & (speed < _SQUAT_SPEED_THRESH)
  run = 0
  for s in is_sq:
    if s:
      run += 1
      if run >= _SQUAT_MIN_CONSECUTIVE:
        return True
    else:
      run = 0
  return False


def _detect_standing_frames(
  data: dict,
  helper: PinocchioHelper,
) -> np.ndarray:
  """Return boolean mask: True where both feet are on ground AND pelvis is near-stationary.

  Uses Pinocchio FK to obtain foot world-frame z-coordinates. The ground
  plane is estimated from the 10th percentile of per-frame min(L, R) foot
  z — robust against brief airborne frames.
  """
  rpos = np.asarray(data["root_pos"], dtype=np.float64)
  rrot = np.asarray(data["root_rot"], dtype=np.float64)  # xyzw
  dpos = np.asarray(data["dof_pos"], dtype=np.float64)
  fps = float(data.get("fps", 30.0))
  T = rpos.shape[0]

  vel = np.zeros_like(rpos)
  vel[1:] = np.diff(rpos, axis=0) * fps
  speed = np.linalg.norm(vel[:, :2], axis=1)

  foot_z = np.zeros((T, 2))
  for t in range(T):
    helper.fk(rpos[t], rrot[t], dpos[t])
    foot_z[t, 0] = helper.data.oMf[helper.lf_id].translation[2]
    foot_z[t, 1] = helper.data.oMf[helper.rf_id].translation[2]

  min_foot_z = np.min(foot_z, axis=1)
  ground_z = float(np.percentile(min_foot_z, 10))

  both_on_ground = (
    (foot_z[:, 0] < ground_z + _STAND_FOOT_Z_THRESH) &
    (foot_z[:, 1] < ground_z + _STAND_FOOT_Z_THRESH)
  )
  feet_close = np.abs(foot_z[:, 0] - foot_z[:, 1]) < _STAND_FEET_Z_DIFF_THRESH
  stationary = speed < _STAND_SPEED_THRESH
  return both_on_ground & feet_close & stationary


# ---------------------------------------------------------------------------
# L-BFGS-B solver
# ---------------------------------------------------------------------------

def _correct_single_frame(
  helper: PinocchioHelper,
  rp: np.ndarray,
  rq: np.ndarray,
  jp: np.ndarray,
  safety_margin: float,
  prev_x: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Optimise joint corrections for one frame. Returns (rp, rq, jp, x)."""
  from scipy.optimize import minimize

  jp_orig = jp.copy()

  def _apply(x):
    j = jp_orig.copy()
    j[_HIP_PITCH_IDX[0]] += x[0]; j[_HIP_PITCH_IDX[1]] += x[1]
    j[_ANKLE_PITCH_IDX[0]] += x[2]; j[_ANKLE_PITCH_IDX[1]] += x[3]
    j[_ANKLE_ROLL_IDX[0]] += x[4]; j[_ANKLE_ROLL_IDX[1]] += x[5]
    j[_WAIST_PITCH_IDX] += x[6]
    return j

  def _obj(x):
    jn = _apply(x)
    rp2, rq2 = helper.reanchor_root(rp, rq, jp_orig, jn)
    h = helper.h_static(rp2, rq2, jn, safety_margin)
    cost = max(0.0, _H_TARGET - h) ** 2 * 10000.0
    cost += float(np.sum(_REG_WEIGHTS * x ** 2))
    if prev_x is not None:
      cost += float(np.sum((x - prev_x) ** 2)) * _TEMPORAL_SMOOTH_WEIGHT
    return cost

  x0 = prev_x if prev_x is not None else np.zeros(7)
  res = minimize(_obj, x0, method="L-BFGS-B", bounds=_OPT_BOUNDS,
                 options={"maxiter": 300, "ftol": 1e-12})
  x = res.x
  jp_f = _apply(x)
  rp_f, rq_f = helper.reanchor_root(rp, rq, jp_orig, jp_f)
  return rp_f, rq_f, jp_f, x


# ---------------------------------------------------------------------------
# CBF solver
# ---------------------------------------------------------------------------

def _cbf_project_with_box(
  a: np.ndarray, rhs: float, v_proposed: np.ndarray,
) -> np.ndarray:
  """Greedy CBF projection with box constraints (mirrors cbf_filter.py logic)."""
  v = v_proposed.copy()
  free = np.ones(len(v), dtype=bool)

  for _ in range(len(v)):
    a_free = a[free]
    a_free_sq = float(a_free @ a_free)
    if a_free_sq < 1e-16:
      break

    margin = float(a @ v) + rhs
    if margin >= -1e-8:
      break

    lam = -margin / (a_free_sq + 1e-12)
    v[free] = v[free] + lam * a[free]

    violations = np.abs(v) > _CBF_VEL_LIMITS
    if not np.any(violations & free):
      break
    v = np.clip(v, -_CBF_VEL_LIMITS, _CBF_VEL_LIMITS)
    free &= ~violations

  return np.clip(v, -_CBF_VEL_LIMITS, _CBF_VEL_LIMITS)


def _cbf_correct_frame(
  helper: PinocchioHelper,
  estimator,  # G1CapturePointEstimator
  rp: np.ndarray,
  rq_xyzw: np.ndarray,
  jp: np.ndarray,
  com_vel_xy: np.ndarray,
  prev_jp: np.ndarray,
  prev_jp_orig: np.ndarray,
  safety_margin: float,
  dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Position-space CBF projection for offline per-frame correction.

  Each iteration computes a Newton step toward h = h_target:
    δq_adj = a · (h_target - h) / ||a||²,  a = ∇h · J_cc[:2, adj]

  This is the offline position-space analogue of the online velocity-space
  CBF used in training/deployment. Velocity-space CBF with α=0.5 only
  pushes h halfway per step (and the capture-point lookahead makes the
  per-step correction even smaller), which is inappropriate for offline
  per-frame feasibility.
  """
  from deploy.common.capture_point import signed_distance_gradient

  jp_orig = jp.copy()

  # Warm-start adjustable joints at (prev_corrected + mocap_delta) so the
  # corrected trajectory inherits the mocap's motion on top of the previous
  # correction. This is the key to frame-to-frame smoothness: without this,
  # each frame re-solves from the raw mocap, causing the CBF to flip between
  # local solutions.
  mocap_delta_adj = jp_orig[_CBF_ADJ_INDICES] - prev_jp_orig[_CBF_ADJ_INDICES]
  jp_cur = jp.copy()
  jp_cur[_CBF_ADJ_INDICES] = prev_jp[_CBF_ADJ_INDICES] + mocap_delta_adj
  rp_cur, rq_cur = helper.reanchor_root(rp, rq_xyzw, jp_orig, jp_cur)

  def _evaluate_h(rp_, rq_xyzw_, jp_):
    rq_wxyz_ = np.array([rq_xyzw_[3], rq_xyzw_[0], rq_xyzw_[1], rq_xyzw_[2]])
    return estimator.evaluate(
      rp_, rq_wxyz_,
      np.array([0.0, 0.0, 0.0]),
      np.array([0.0, 0.0, 0.0]),
      jp_, np.zeros(29),
    )

  for iteration in range(_CBF_MAX_REFINE_ITERS):
    cp_data = _evaluate_h(rp_cur, rq_cur, jp_cur)
    h = cp_data.h_now
    if h >= _H_TARGET - 1e-3:
      break

    polygon = cp_data.shrunk_polygon_xy
    if polygon.shape[0] < 3:
      polygon = cp_data.raw_polygon_xy
    if polygon.shape[0] < 3:
      break

    J_cc = estimator.compute_contact_consistent_com_jacobian()
    nabla_h = signed_distance_gradient(cp_data.capture_point_xy, polygon)

    # Position-space gradient: a[i] = dh/dq_i (metres per radian).
    a_full = nabla_h @ J_cc[:2, :]
    a = a_full[_CBF_ADJ_INDICES]
    a_norm_sq = float(a @ a)
    if a_norm_sq < 1e-16:
      break

    # Full Newton step toward h = h_target.
    lam = (_H_TARGET - h) / (a_norm_sq + 1e-12)
    delta_q_full = lam * a

    # Smoothness clamp: limit the CBF *correction* (on top of the mocap
    # evolution) to vel_smooth · dt per frame. The warm-start already
    # inherits the mocap's motion; this caps how far the correction drifts.
    max_delta = _CBF_SMOOTH_VEL_LIMIT * dt
    warm_start_adj = prev_jp[_CBF_ADJ_INDICES] + mocap_delta_adj
    jp_min = warm_start_adj - max_delta
    jp_max = warm_start_adj + max_delta

    # Backtracking line search: accept smallest t ∈ {1, 0.5, 0.25, ...}
    # that produces an h improvement (monotone decrease in |h - h_target|).
    step = 1.0
    best_rp, best_rq, best_jp = rp_cur, rq_cur, jp_cur
    best_improvement = 0.0
    for _ in range(_CBF_LINE_SEARCH_ITERS):
      jp_try = jp_cur.copy()
      jp_try[_CBF_ADJ_INDICES] = np.clip(
        jp_cur[_CBF_ADJ_INDICES] + step * delta_q_full, jp_min, jp_max,
      )
      jp_try[_CBF_NON_ADJ_MASK] = jp_orig[_CBF_NON_ADJ_MASK]
      rp_try, rq_try = helper.reanchor_root(rp, rq_xyzw, jp_orig, jp_try)
      h_try = _evaluate_h(rp_try, rq_try, jp_try).h_now
      improvement = h_try - h
      if improvement > best_improvement:
        best_improvement = improvement
        best_rp, best_rq, best_jp = rp_try, rq_try, jp_try
        if h_try >= _H_TARGET - 1e-3:
          break
      step *= 0.5

    if best_improvement <= 1e-5:
      break  # line search failed to make progress
    rp_cur, rq_cur, jp_cur = best_rp, best_rq, best_jp

  return rp_cur, rq_cur, jp_cur


# ---------------------------------------------------------------------------
# Public orchestrator: detect eligible frames, run chosen solver, blend, FK
# ---------------------------------------------------------------------------

def correct_capture_point(
  data: dict,
  urdf_path: str,
  safety_margin: float,
  xml_path: str,
  *,
  method: str = "lbfgsb",
  trigger: str = "squat",
) -> dict:
  """Apply capture-point correction to eligible frames, mutating *data* in-place.

  Parameters
  ----------
  data : dict
      Motion dict with ``root_pos`` (T,3), ``root_rot`` (T,4 xyzw), ``dof_pos``
      (T,29), optional ``fps``. Mutated to include corrected poses and
      re-computed ``body_pos_w`` / ``body_quat_w``.
  urdf_path : str
      Path to the G1 URDF (for Pinocchio).
  safety_margin : float
      Inward shrink margin on the support polygon, in metres.
  xml_path : str
      Path to the G1 MuJoCo XML (for body-transform recomputation).
  method : {"lbfgsb", "cbf"}
      ``"lbfgsb"`` — L-BFGS-B nonlinear least-squares per frame.
      ``"cbf"``     — CBF Newton projection with line search and warm-start.
  trigger : {"squat", "standing"}
      ``"squat"``    — correct only frames with root height < 0.65 m and
      near-stationary pelvis.
      ``"standing"`` — correct every frame where both feet are on the ground
      and pelvis is near-stationary, regardless of root height.
  """
  from scipy.spatial.transform import Rotation as R, Slerp
  from scipy.ndimage import uniform_filter1d

  rpos = np.asarray(data["root_pos"], dtype=np.float64)
  rrot = np.asarray(data["root_rot"], dtype=np.float64)
  dpos = np.asarray(data["dof_pos"], dtype=np.float64)
  fps = float(data.get("fps", 30.0))
  T = rpos.shape[0]
  dt = 1.0 / fps

  # Pinocchio helper is used by all branches below (re-anchoring, h_static
  # check, foot-z detection).
  helper = PinocchioHelper(urdf_path)

  # --- detect eligible frames (with erosion) ---
  vel = np.zeros_like(rpos); vel[1:] = np.diff(rpos, axis=0) * fps
  speed = np.linalg.norm(vel[:, :2], axis=1)
  if trigger == "standing":
    raw_eligible = _detect_standing_frames(data, helper)
    min_run = _STAND_MIN_CONSECUTIVE
  else:  # "squat"
    raw_eligible = (rpos[:, 2] < _SQUAT_HEIGHT_THRESH) & (speed < _SQUAT_SPEED_THRESH)
    min_run = _SQUAT_MIN_CONSECUTIVE

  # Erode to remove isolated detections.
  sq = raw_eligible.copy()
  for i in range(T):
    if raw_eligible[i] and raw_eligible[max(0, i - 3):min(T, i + 4)].sum() < 5:
      sq[i] = False

  # Drop runs shorter than min_run.
  if sq.any():
    in_run = False
    run_start = 0
    for i in range(T):
      if sq[i] and not in_run:
        in_run = True; run_start = i
      elif not sq[i] and in_run:
        if i - run_start < min_run:
          sq[run_start:i] = False
        in_run = False
    if in_run and T - run_start < min_run:
      sq[run_start:T] = False

  if not sq.any():
    return data

  # --- per-frame correction ---
  c_pos, c_rot, c_dof = rpos.copy(), rrot.copy(), dpos.copy()

  if method == "cbf":
    from deploy.common.capture_point import G1CapturePointEstimator
    estimator = G1CapturePointEstimator(
      urdf_path=urdf_path,
      safety_margin=safety_margin,
    )
    # prev_jp: previous corrected joints; prev_jp_orig: previous original mocap
    # joints. We need both so the warm-start preserves mocap's frame-to-frame
    # motion on top of the previous correction.
    prev_jp = dpos[0].copy()
    prev_jp_orig = dpos[0].copy()
    for t in range(T):
      if not sq[t]:
        prev_jp = dpos[t].copy()
        prev_jp_orig = dpos[t].copy()
        continue
      if helper.h_static(rpos[t], rrot[t], dpos[t], safety_margin) < _H_TARGET:
        c_pos[t], c_rot[t], c_dof[t] = _cbf_correct_frame(
          helper, estimator,
          rpos[t], rrot[t], dpos[t],
          vel[t, :2],  # COM velocity approximation
          prev_jp, prev_jp_orig, safety_margin, dt,
        )
      prev_jp = c_dof[t].copy()
      prev_jp_orig = dpos[t].copy()
  else:
    # Original L-BFGS-B method.
    prev_x: np.ndarray | None = None
    for t in range(T):
      if not sq[t]:
        prev_x = None
        continue
      if helper.h_static(rpos[t], rrot[t], dpos[t], safety_margin) < _H_TARGET:
        c_pos[t], c_rot[t], c_dof[t], prev_x = _correct_single_frame(
          helper, rpos[t], rrot[t], dpos[t], safety_margin, prev_x,
        )
      else:
        prev_x = np.zeros(7)

  # --- blend at squat boundaries ---
  blend = np.zeros(T)
  ranges: list[tuple[int, int]] = []
  in_s = False
  for i in range(T):
    if sq[i] and not in_s:
      start = i; in_s = True
    elif not sq[i] and in_s:
      ranges.append((start, i - 1)); in_s = False
  if in_s:
    ranges.append((start, T - 1))

  for s, e in ranges:
    blend[s:e + 1] = 1.0
    for i in range(_BLEND_FRAMES):
      if s + i <= e: blend[s + i] = min(blend[s + i], (i + 1) / _BLEND_FRAMES)
      if e - i >= s: blend[e - i] = min(blend[e - i], (i + 1) / _BLEND_FRAMES)
  blend = uniform_filter1d(blend, size=5)
  blend = np.clip(blend, 0.0, 1.0)

  f_pos, f_rot, f_dof = rpos.copy(), rrot.copy(), dpos.copy()
  for t in range(T):
    w = blend[t]
    if w < 1e-6:
      continue
    f_pos[t] = rpos[t] * (1 - w) + c_pos[t] * w
    f_dof[t] = dpos[t] * (1 - w) + c_dof[t] * w
    slerp = Slerp([0, 1], R.concatenate([R.from_quat(rrot[t]), R.from_quat(c_rot[t])]))
    f_rot[t] = slerp(w).as_quat()
    if np.dot(f_rot[t], rrot[t]) < 0:
      f_rot[t] = -f_rot[t]

  data["root_pos"] = f_pos.astype(np.float32)
  data["root_rot"] = f_rot.astype(np.float32)
  data["dof_pos"] = f_dof.astype(np.float32)

  # Recompute body transforms from corrected data
  recompute_body_transforms_mujoco(data, xml_path)
  return data
