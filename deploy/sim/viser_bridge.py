"""Viser bridge for the sim node.

Hosts a single ``viser.ViserServer`` that renders:
  * the robot loaded from the payload URDF,
  * molmo marker spheres in the 3D scene,
  * the RGB camera panel with molmo overlay dots composited in,
  * the molmo prompt text input + action buttons.

All viser imports happen lazily inside ``ViserBridge`` so the sim node can
skip this module entirely when viser is disabled.
"""

from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


def _parse_floats(s: str | None, n: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if s is None:
        return default
    vals = tuple(float(x) for x in s.split())
    if len(vals) < n:
        vals = vals + default[len(vals):]
    return vals[:n]


def _euler_xyz_to_wxyz(euler: tuple[float, ...]) -> tuple[float, float, float, float]:
    """MuJoCo eulerseq-xyz angles (rad) → wxyz quaternion."""
    rx, ry, rz = float(euler[0]), float(euler[1]), float(euler[2])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rx @ Ry @ Rz
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return (0.25 / s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return ((R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s)
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return ((R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s)
    s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
    return ((R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s)


def _quat_mul(
    a: tuple[float, ...], b: tuple[float, ...]
) -> tuple[float, float, float, float]:
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bw, bx, by, bz = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _rotate_by_wxyz(wxyz: tuple[float, ...], v: tuple[float, ...]) -> np.ndarray:
    """Rotate 3-vector v by quaternion wxyz."""
    w, x, y, z = float(wxyz[0]), float(wxyz[1]), float(wxyz[2]), float(wxyz[3])
    qv = np.array([x, y, z])
    p = np.array([float(v[0]), float(v[1]), float(v[2])])
    return p + 2.0 * w * np.cross(qv, p) + 2.0 * np.cross(qv, np.cross(qv, p))


def _wxyz_to_rotmat(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = float(wxyz[0]), float(wxyz[1]), float(wxyz[2]), float(wxyz[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )

try:
    import cv2  # only used to draw overlay dots on the RGB composite
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False


class ViserBridge:
    def __init__(
        self,
        urdf_path: Path,
        joint_names: Sequence[str],
        port: int,
        *,
        on_submit_prompt: Callable[[str], None],
        on_resubmit: Callable[[], None],
        on_toggle_voice: Callable[[], None],
        on_reset_molmo: Callable[[], None],
        on_reset_sim: Callable[[], None] | None = None,
        initial_prompt: str = "",
        robot_hz: float = 30.0,
        image_hz: float = 15.0,
        gripper_joint_names: Sequence[str] | None = None,
    ) -> None:
        import viser
        from viser.extras import ViserUrdf

        self._server = viser.ViserServer(port=port)

        # --- 3D scene: robot + marker spheres ---
        self._base = self._server.scene.add_frame("/pelvis", show_axes=False)
        self._urdf = ViserUrdf(
            self._server,
            urdf_or_path=Path(urdf_path),
            root_node_name="/pelvis",
            load_collision_meshes=False,
        )
        self._server.scene.add_grid(
            "/floor",
            width=10.0,
            height=10.0,
            cell_size=0.5,
            section_size=1.0,
        )
        # Brighter ambient so textured graspnet meshes (large albedo range)
        # don't render too dark under the default key light alone.
        try:
            self._server.scene.add_light_ambient(
                "/scene_ambient", color=(1.0, 1.0, 1.0), intensity=0.6,
            )
        except Exception:
            pass  # older viser builds lack this API
        urdf_joint_order = self._urdf.get_actuated_joint_names()
        body_names = list(joint_names)
        gripper_names = list(gripper_joint_names) if gripper_joint_names else []
        combined_names = body_names + gripper_names
        missing = [n for n in urdf_joint_order if n not in combined_names]
        if missing:
            raise ValueError(
                f"URDF actuated joints missing from caller-provided lists: {missing}"
            )
        self._joint_reorder = np.array(
            [combined_names.index(n) for n in urdf_joint_order],
            dtype=np.int64,
        )
        self._body_joint_count = len(body_names)
        self._gripper_joint_count = len(gripper_names)
        self._urdf_joint_count = len(urdf_joint_order)
        self._server.scene.add_frame("/molmo", show_axes=False)
        self._molmo_handles: list = []

        # Planner visualization roots + handles
        self._server.scene.add_frame("/planner", show_axes=False)
        self._loco_path_handle = None
        self._loco_goal_handle = None
        self._loco_path_last: np.ndarray | None = None
        self._ee_left_handle = None
        self._ee_right_handle = None
        self._esdf_voxel_handle = None
        self._ffs_obb_handle = None
        # Last-drawn extent (meters). publish_obb_body recreates the box
        # handle only when the new extent diverges by > 1 cm; in-between
        # pose updates mutate handle.position / handle.wxyz in place.
        self._ffs_obb_last_extent: np.ndarray | None = None
        # Per-hand wireframe boxes drawn at the early-close trigger volume
        # (target offset from gripper midpoint in wrist-local coords must
        # land inside this box for early-close-into-grasp to fire). Created
        # lazily on first publish_grasp_boxes() call.
        self._grasp_box_handle_l = None
        self._grasp_box_handle_r = None
        self._grasp_box_last_dims: tuple[float, float, float] | None = None
        # Solid spheres at each gripper midpoint so the user can see the
        # box's reference center clearly — wireframe alone is depth-
        # ambiguous when viewed from oblique angles.
        self._grasp_mid_handle_l = None
        self._grasp_mid_handle_r = None

        # Handles for world bodies whose pose is streamed each tick (free joints).
        self._movable_handles: dict[str, "viser.SceneNodeHandle"] = {}

        # --- Sim health indicator (policy connected? robot height?) ---
        self._sim_status_md = self._server.gui.add_markdown(
            "**Sim:** starting…"
        )

        # --- 2D panel: RGB (depth dropped — was ~frozen and expensive to colorize) ---
        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
        self._rgb_handle = self._server.gui.add_image(placeholder, label="RGB")
        self._status_md = self._server.gui.add_markdown("**Status:** Ready")
        self._planner_status_md = self._server.gui.add_markdown("**Waypoint Planner:** idle")
        self._plan_md = self._server.gui.add_markdown("**Plan:** _(no prompt submitted)_")

        # --- Prompt input + action buttons ---
        with self._server.gui.add_folder("Prompt"):
            self._prompt_text = self._server.gui.add_text(
                "Prompt", initial_value=initial_prompt, multiline=True
            )
            submit_btn = self._server.gui.add_button("Submit prompt")
            resubmit_btn = self._server.gui.add_button("Re-submit current")
        with self._server.gui.add_folder("Controls"):
            self._voice_btn = self._server.gui.add_button("Toggle voice recording")
            reset_btn = self._server.gui.add_button("Reset molmo")
            reset_sim_btn = self._server.gui.add_button("Reset sim")
        with self._server.gui.add_folder("Visualization"):
            self._show_paths_cb = self._server.gui.add_checkbox(
                "Show planner paths", initial_value=False
            )
            self._show_voxels_cb = self._server.gui.add_checkbox(
                "Show ESDF voxels", initial_value=False
            )

        def _handle_submit() -> None:
            text = (self._prompt_text.value or "").strip()
            if text:
                self._status_md.content = f"**Status:** submitted prompt: {text}"
            on_submit_prompt(self._prompt_text.value)

        def _handle_resubmit() -> None:
            text = (self._prompt_text.value or "").strip()
            if text:
                self._status_md.content = f"**Status:** submitted prompt: {text}"
            on_resubmit()

        submit_btn.on_click(lambda _: _handle_submit())
        resubmit_btn.on_click(lambda _: _handle_resubmit())
        self._voice_btn.on_click(lambda _: on_toggle_voice())
        reset_btn.on_click(lambda _: on_reset_molmo())
        if on_reset_sim is not None:
            reset_sim_btn.on_click(lambda _: on_reset_sim())
        self._show_paths_cb.on_update(lambda _: self._apply_path_visibility())
        self._show_voxels_cb.on_update(lambda _: self._apply_voxel_visibility())

        # Throttles
        self._robot_period = 1.0 / max(robot_hz, 1e-3)
        self._image_period = 1.0 / max(image_hz, 1e-3)
        self._last_robot_t = 0.0
        self._last_image_t = 0.0

        # Latest overlay state (composited on top of the next RGB push)
        self._state_lock = threading.Lock()
        self._overlay_points: list[tuple[int, int]] = []
        self._overlay_status = "Ready"
        self._voice_recording = False
        self._latest_rgb: np.ndarray | None = None

    # ---------- control loop hooks ----------

    def publish_robot(
        self,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
        joint_pos: np.ndarray,
        gripper_pos: np.ndarray | None = None,
    ) -> None:
        now = time.perf_counter()
        if now - self._last_robot_t < self._robot_period:
            return
        self._last_robot_t = now
        self._base.position = (float(root_pos[0]), float(root_pos[1]), float(root_pos[2]))
        self._base.wxyz = (
            float(root_quat_wxyz[0]),
            float(root_quat_wxyz[1]),
            float(root_quat_wxyz[2]),
            float(root_quat_wxyz[3]),
        )
        body = np.asarray(joint_pos, dtype=np.float32)
        if self._gripper_joint_count == 0:
            combined = body
        else:
            if gripper_pos is None:
                gripper = np.zeros(self._gripper_joint_count, dtype=np.float32)
            else:
                gripper = np.asarray(gripper_pos, dtype=np.float32)
            combined = np.concatenate([body, gripper])
        cfg = combined[self._joint_reorder]
        self._urdf.update_cfg(cfg)

    def publish_objects(
        self,
        poses: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Update movable body poses. ``poses[name] = (xyz, wxyz)``."""
        for name, (xyz, wxyz) in poses.items():
            handle = self._movable_handles.get(name)
            if handle is None:
                continue
            handle.position = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
            handle.wxyz = (
                float(wxyz[0]),
                float(wxyz[1]),
                float(wxyz[2]),
                float(wxyz[3]),
            )

    def load_mjcf_scene(
        self,
        mjcf_path: Path | str | None = None,
        *,
        model: "mujoco.MjModel | None" = None,
        skip_bodies: Iterable[str] = (),
        skip_geoms: Iterable[str] = (),
        prefix: str = "/scene",
    ) -> list[str]:
        """Render the static scene + movable objects in viser.

        If ``model`` is given, walks the compiled ``MjModel`` directly — this
        sees runtime-injected bodies (e.g. graspnet objects added by
        :func:`deploy.sim.graspnet_scene.inject_graspnet_objects`) that aren't
        in the on-disk XML. Mesh geoms are rendered as textured trimesh objects
        when the geom's material has a 2D texture; otherwise as a flat-color
        mesh. Collision geoms (group == 3) are skipped.

        If only ``mjcf_path`` is given (legacy), parses the XML's
        ``<worldbody>`` and renders ``box``/``sphere``/``cylinder`` primitives.

        Free-jointed bodies are registered in ``self._movable_handles`` so
        :meth:`publish_objects` can update their pose live.

        Returns:
            Body names registered as movable.
        """
        if model is not None:
            return self._load_scene_from_model(
                model,
                skip_bodies=skip_bodies,
                skip_geoms=skip_geoms,
                prefix=prefix,
            )
        assert mjcf_path is not None, "Provide either model or mjcf_path"

        tree = ET.parse(str(mjcf_path))
        worldbodies = tree.getroot().findall("worldbody")
        if not worldbodies:
            return []

        self._server.scene.add_frame(prefix, show_axes=False)
        skip_body_set = set(skip_bodies)
        skip_geom_set = set(skip_geoms)
        movable: list[str] = []

        for worldbody in worldbodies:
            for geom in worldbody.findall("geom"):
                gname = geom.get("name", "")
                if gname in skip_geom_set:
                    continue
                self._add_mjcf_geom(f"{prefix}/{gname or 'geom'}", geom, (0.0, 0.0, 0.0))

            for body in worldbody.findall("body"):
                name = body.get("name", "")
                if not name or name in skip_body_set:
                    continue
                bpos = _parse_floats(body.get("pos"), 3, (0.0, 0.0, 0.0))
                if body.get("quat"):
                    body_wxyz = _parse_floats(body.get("quat"), 4, (1.0, 0.0, 0.0, 0.0))
                else:
                    body_wxyz = _euler_xyz_to_wxyz(
                        _parse_floats(body.get("euler"), 3, (0.0, 0.0, 0.0))
                    )
                body_prefix = f"{prefix}/{name}"

                if body.find("freejoint") is not None:
                    geom = body.find("geom")
                    if geom is None:
                        continue
                    handle = self._add_mjcf_geom(body_prefix, geom, bpos, body_wxyz)
                    if handle is not None:
                        self._movable_handles[name] = handle
                        movable.append(name)
                else:
                    self._server.scene.add_frame(body_prefix, show_axes=False)
                    for geom in body.findall("geom"):
                        gname = geom.get("name", "geom")
                        self._add_mjcf_geom(f"{body_prefix}/{gname}", geom, bpos, body_wxyz)

        return movable

    def _load_scene_from_model(
        self,
        model,
        *,
        skip_bodies: Iterable[str] = (),
        skip_geoms: Iterable[str] = (),
        prefix: str = "/scene",
    ) -> list[str]:
        """Walk an MjModel and render worldbody-level bodies + their geoms."""
        import mujoco as _mj

        self._server.scene.add_frame(prefix, show_axes=False)
        skip_body_set = set(skip_bodies)
        skip_geom_set = set(skip_geoms)
        movable: list[str] = []

        # Worldbody geoms (children of body 0, before any sub-body)
        wb_geom_start = int(model.body_geomadr[0])
        wb_geom_num = int(model.body_geomnum[0])
        for gid in range(wb_geom_start, wb_geom_start + wb_geom_num):
            gname = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_GEOM, gid) or ""
            if gname in skip_geom_set:
                continue
            if int(model.geom_group[gid]) == 3:
                continue
            self._add_model_geom(f"{prefix}/{gname or f'geom{gid}'}", model, gid)

        # All bodies whose parent is body 0 (the world).
        for bid in range(1, model.nbody):
            if int(model.body_parentid[bid]) != 0:
                continue
            bname = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_BODY, bid) or ""
            if not bname or bname in skip_body_set:
                continue
            body_pos = tuple(float(x) for x in model.body_pos[bid])
            body_wxyz = tuple(float(x) for x in model.body_quat[bid])
            body_prefix = f"{prefix}/{bname}"

            # Check freejoint
            jntnum = int(model.body_jntnum[bid])
            jntadr = int(model.body_jntadr[bid])
            is_free = jntnum >= 1 and int(model.jnt_type[jntadr]) == int(_mj.mjtJoint.mjJNT_FREE)

            geom_start = int(model.body_geomadr[bid])
            geom_num = int(model.body_geomnum[bid])
            if geom_num <= 0:
                if not is_free:
                    self._server.scene.add_frame(body_prefix, show_axes=False)
                continue

            primary_handle = None
            for gid in range(geom_start, geom_start + geom_num):
                gname = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"
                if gname in skip_geom_set:
                    continue
                if int(model.geom_group[gid]) == 3:  # collision-only
                    continue
                # Static body: bake the body's fixed pose into each geom.
                # Free body: keep geom at origin; the movable_handle drives world pose.
                handle = self._add_model_geom(
                    f"{body_prefix}/{gname}" if not is_free else body_prefix,
                    model,
                    gid,
                    body_pos=body_pos if not is_free else (0.0, 0.0, 0.0),
                    body_wxyz=body_wxyz if not is_free else (1.0, 0.0, 0.0, 0.0),
                )
                if is_free and primary_handle is None and handle is not None:
                    primary_handle = handle

            if is_free and primary_handle is not None:
                self._movable_handles[bname] = primary_handle
                movable.append(bname)
            elif not is_free:
                # Non-free body: render as a static frame; geoms above used body pose.
                pass

        return movable

    def _add_model_geom(
        self,
        path: str,
        model,
        geom_id: int,
        body_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        body_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ):
        """Render a single geom from the compiled MjModel."""
        import mujoco as _mj

        gtype = int(model.geom_type[geom_id])
        size = model.geom_size[geom_id]
        rgba = model.geom_rgba[geom_id]
        # geom-local pose relative to the body
        gpos_local = tuple(float(x) for x in model.geom_pos[geom_id])
        gquat_local = tuple(float(x) for x in model.geom_quat[geom_id])
        # Compose body * geom-local pose
        gpos_world = _rotate_by_wxyz(body_wxyz, gpos_local)
        pos = (
            body_pos[0] + float(gpos_world[0]),
            body_pos[1] + float(gpos_world[1]),
            body_pos[2] + float(gpos_world[2]),
        )
        wxyz = _quat_mul(body_wxyz, gquat_local)
        color = (float(rgba[0]), float(rgba[1]), float(rgba[2]))

        if gtype == int(_mj.mjtGeom.mjGEOM_PLANE):
            return None  # ground plane — rely on viser's built-in grid
        if gtype == int(_mj.mjtGeom.mjGEOM_BOX):
            # Textured box: render as a 6-face mesh so the material's texture
            # actually shows in viser. Plain box (no texture) → add_box.
            is_free_body = body_pos == (0.0, 0.0, 0.0) and body_wxyz == (1.0, 0.0, 0.0, 0.0)
            tex_handle = self._add_textured_box(
                path, model, geom_id,
                is_free=is_free_body,
                body_pos=body_pos,
                body_wxyz=body_wxyz,
            )
            if tex_handle is not None:
                return tex_handle
            return self._server.scene.add_box(
                path,
                dimensions=(2 * float(size[0]), 2 * float(size[1]), 2 * float(size[2])),
                color=color,
                position=pos,
                wxyz=wxyz,
            )
        if gtype == int(_mj.mjtGeom.mjGEOM_SPHERE):
            return self._server.scene.add_icosphere(
                path,
                radius=float(size[0]),
                color=color,
                position=pos,
                wxyz=wxyz,
            )
        if gtype == int(_mj.mjtGeom.mjGEOM_CYLINDER):
            return self._server.scene.add_cylinder(
                path,
                radius=float(size[0]),
                height=2 * float(size[1]),
                color=color,
                position=pos,
                wxyz=wxyz,
            )
        if gtype == int(_mj.mjtGeom.mjGEOM_MESH):
            # For free bodies (body_pos==0, body_wxyz==identity passed in by
            # _load_scene_from_model), we bake geom_local into vertices and
            # render at identity. publish_objects then drives the handle to
            # the body's world pose. For static bodies we use the composed
            # geom-world pose.
            is_free_body = body_pos == (0.0, 0.0, 0.0) and body_wxyz == (1.0, 0.0, 0.0, 0.0)
            if is_free_body:
                return self._add_model_mesh(
                    path, model, geom_id,
                    pos=(0.0, 0.0, 0.0), wxyz=(1.0, 0.0, 0.0, 0.0),
                    body_local=True,
                )
            return self._add_model_mesh(path, model, geom_id, pos, wxyz)
        return None

    def _add_textured_box(
        self,
        path: str,
        model,
        geom_id: int,
        *,
        is_free: bool,
        body_pos: tuple[float, float, float],
        body_wxyz: tuple[float, float, float, float],
    ):
        """Render a primitive BOX geom as a 6-face textured mesh.

        Returns the handle if the geom has a material+texture bound; else None
        (caller falls back to a flat-color :meth:`add_box`).
        Mirrors the body_local baking logic of :meth:`_add_model_mesh` so the
        cube renders correctly in viser whether the body is static (table-
        mounted) or free (movable via ``publish_objects``).
        """
        import mujoco as _mj
        try:
            import trimesh
            from PIL import Image
        except ImportError:
            return None

        mat_id = int(model.geom_matid[geom_id])
        if mat_id < 0:
            return None
        try:
            rgb_role = int(_mj.mjtTextureRole.mjTEXROLE_RGB)
            tex_id = int(model.mat_texid[mat_id, rgb_role])
        except Exception:
            tex_id = -1
        if tex_id < 0:
            return None

        # Pull the texture image. For CUBE textures MuJoCo stacks the 6 faces
        # vertically (height = 6 × width); we use the first face since our box
        # texture has the same image on all 6 faces.
        w = int(model.tex_width[tex_id])
        h = int(model.tex_height[tex_id])
        nchan = int(model.tex_nchannel[tex_id])
        tex_off = int(model.tex_adr[tex_id])
        img_arr = np.asarray(
            model.tex_data[tex_off : tex_off + w * h * nchan], dtype=np.uint8
        ).reshape(h, w, nchan)
        if h == 6 * w:
            img_arr = img_arr[:w]  # first face
        mode = {3: "RGB", 4: "RGBA"}.get(nchan, "L")
        img = Image.fromarray(img_arr if nchan != 1 else img_arr[..., 0], mode)

        # Geom-local extents and pose.
        size = model.geom_size[geom_id]
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        geom_local_pos = np.asarray(model.geom_pos[geom_id], dtype=np.float64)
        geom_local_quat = np.asarray(model.geom_quat[geom_id], dtype=np.float64)

        # 24 vertices (4 per face × 6 faces), each face's UVs span the full
        # texture so the entire image appears on each face.
        faces_def = [
            # (+X)
            [( sx, -sy, -sz), ( sx,  sy, -sz), ( sx,  sy,  sz), ( sx, -sy,  sz)],
            # (-X)
            [(-sx,  sy, -sz), (-sx, -sy, -sz), (-sx, -sy,  sz), (-sx,  sy,  sz)],
            # (+Y)
            [( sx,  sy, -sz), (-sx,  sy, -sz), (-sx,  sy,  sz), ( sx,  sy,  sz)],
            # (-Y)
            [(-sx, -sy, -sz), ( sx, -sy, -sz), ( sx, -sy,  sz), (-sx, -sy,  sz)],
            # (+Z)
            [(-sx, -sy,  sz), ( sx, -sy,  sz), ( sx,  sy,  sz), (-sx,  sy,  sz)],
            # (-Z)
            [(-sx,  sy, -sz), ( sx,  sy, -sz), ( sx, -sy, -sz), (-sx, -sy, -sz)],
        ]
        verts = np.array([v for face in faces_def for v in face], dtype=np.float64)
        tris = []
        for i in range(6):
            b = i * 4
            tris.append([b, b + 1, b + 2])
            tris.append([b, b + 2, b + 3])
        faces_idx = np.array(tris, dtype=np.int32)
        # Per-face UVs covering full image. trimesh expects bottom-left origin;
        # MuJoCo textures top-left → flip V.
        face_uv = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        uvs = np.tile(face_uv, (6, 1))

        # Bake geom-local pose into vertices so the resulting mesh is in
        # body-local frame (matches _add_model_mesh's body_local=True path).
        R_glocal = _wxyz_to_rotmat(geom_local_quat)
        verts_body = verts @ R_glocal.T + geom_local_pos

        tm = trimesh.Trimesh(vertices=verts_body, faces=faces_idx, process=False)
        pbr = trimesh.visual.material.PBRMaterial(
            baseColorTexture=img,
            emissiveTexture=img,
            emissiveFactor=(0.35, 0.35, 0.35),
            metallicFactor=0.0,
            roughnessFactor=0.7,
        )
        tm.visual = trimesh.visual.TextureVisuals(uv=uvs, material=pbr)

        if is_free:
            # Handle at identity; publish_objects drives world pose. Verts are
            # in body-local frame with geom_local baked in.
            return self._server.scene.add_mesh_trimesh(
                path, mesh=tm, position=(0.0, 0.0, 0.0), wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        # Static body: verts are in body-local; transform by body's world pose
        # only (do NOT pass the composed geom-world pose — geom_local is
        # already baked into the vertices).
        return self._server.scene.add_mesh_trimesh(
            path, mesh=tm, position=body_pos, wxyz=body_wxyz,
        )

    def _add_model_mesh(self, path, model, geom_id, pos, wxyz, body_local: bool = False):
        """Render a mesh geom with its texture (if any) via trimesh.

        When ``body_local=True``, bake the geom's body-local transform
        (``geom_pos``/``geom_quat``) into the mesh vertices so the resulting
        mesh is in body-local coords. The caller can then drive a single
        handle transform with the body's world pose (``data.xpos``/``xquat``)
        — needed for free-jointed bodies whose handle gets overwritten by
        :meth:`publish_objects`.
        """
        import mujoco as _mj
        try:
            import trimesh
            from PIL import Image
        except ImportError:
            return None

        mesh_id = int(model.geom_dataid[geom_id])
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        face_adr = int(model.mesh_faceadr[mesh_id])
        face_num = int(model.mesh_facenum[mesh_id])
        verts = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(model.mesh_face[face_adr : face_adr + face_num], dtype=np.int32).reshape(-1, 3)

        # Pull MuJoCo's vertex normals when they match 1:1 with verts (graspnet/lab_table do).
        # This gives smooth shading in viser instead of flat per-face fallback.
        normal_num = int(model.mesh_normalnum[mesh_id])
        normal_adr = int(model.mesh_normaladr[mesh_id])
        vnormals = None
        if normal_num == vert_num and normal_adr >= 0:
            vnormals = np.asarray(
                model.mesh_normal[normal_adr : normal_adr + normal_num], dtype=np.float64
            ).reshape(-1, 3)

        if body_local:
            # Bake the geom's body-local pose into the vertices so the mesh
            # is body-frame native. Caller will set the handle transform to
            # the body's WORLD pose.
            geom_local_pos = np.asarray(model.geom_pos[geom_id], dtype=np.float64)
            geom_local_quat = np.asarray(model.geom_quat[geom_id], dtype=np.float64)
            R_glocal = _wxyz_to_rotmat(geom_local_quat)
            verts = verts @ R_glocal.T + geom_local_pos
            if vnormals is not None:
                vnormals = vnormals @ R_glocal.T

        tm = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=vnormals,
            process=False,
        )

        # Texture binding (if material → 2D texture exists).
        mat_id = int(model.geom_matid[geom_id])
        tc_adr = int(model.mesh_texcoordadr[mesh_id])
        tc_num = int(model.mesh_texcoordnum[mesh_id])
        if mat_id >= 0 and tc_adr >= 0 and tc_num > 0:
            try:
                rgb_role = int(_mj.mjtTextureRole.mjTEXROLE_RGB)
                tex_id = int(model.mat_texid[mat_id, rgb_role])
            except Exception:
                tex_id = -1
            if tex_id >= 0:
                w = int(model.tex_width[tex_id])
                h = int(model.tex_height[tex_id])
                nchan = int(model.tex_nchannel[tex_id])
                tex_off = int(model.tex_adr[tex_id])
                img_arr = np.asarray(
                    model.tex_data[tex_off : tex_off + w * h * nchan], dtype=np.uint8
                ).reshape(h, w, nchan)
                if nchan == 3:
                    img = Image.fromarray(img_arr, "RGB")
                elif nchan == 4:
                    img = Image.fromarray(img_arr, "RGBA")
                else:
                    img = Image.fromarray(img_arr[..., 0], "L")
                # MuJoCo texture origin is top-left; trimesh expects bottom-left → flip V.
                uvs = np.asarray(
                    model.mesh_texcoord[tc_adr : tc_adr + tc_num], dtype=np.float64
                ).reshape(-1, 2).copy()
                uvs[:, 1] = 1.0 - uvs[:, 1]
                # mesh_facetexcoord exists when UVs differ from vertex indexing,
                # but graspnet meshes have 1:1 vert↔UV, so direct attachment works.
                if uvs.shape[0] == verts.shape[0]:
                    # Use PBR material with mild self-emission so viser's default
                    # lighting doesn't over-darken the texture albedo.
                    pbr = trimesh.visual.material.PBRMaterial(
                        baseColorTexture=img,
                        emissiveTexture=img,
                        emissiveFactor=(0.35, 0.35, 0.35),
                        metallicFactor=0.0,
                        roughnessFactor=0.7,
                    )
                    tm.visual = trimesh.visual.TextureVisuals(uv=uvs, material=pbr)
                else:
                    # Fallback: flat color
                    tm.visual.face_colors = (200, 200, 200, 255)
            else:
                tm.visual.face_colors = tuple(int(c * 255) for c in model.geom_rgba[geom_id])
        else:
            tm.visual.face_colors = tuple(int(c * 255) for c in model.geom_rgba[geom_id])

        return self._server.scene.add_mesh_trimesh(path, mesh=tm, position=pos, wxyz=wxyz)

    def _add_mjcf_geom(
        self,
        path: str,
        geom: ET.Element,
        body_pos: tuple[float, float, float],
        body_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ):
        """Render a single MJCF ``<geom>`` as a viser primitive.

        Returns the created handle, or ``None`` if the geom type is unsupported.
        """
        gtype = geom.get("type", "sphere")
        gpos_local = _parse_floats(geom.get("pos"), 3, (0.0, 0.0, 0.0))
        rgba = _parse_floats(geom.get("rgba"), 4, (0.7, 0.7, 0.7, 1.0))
        gpos_world = _rotate_by_wxyz(body_wxyz, gpos_local)
        pos = (
            body_pos[0] + float(gpos_world[0]),
            body_pos[1] + float(gpos_world[1]),
            body_pos[2] + float(gpos_world[2]),
        )
        color = (rgba[0], rgba[1], rgba[2])

        if gtype == "box":
            size = _parse_floats(geom.get("size"), 3, (0.01, 0.01, 0.01))
            return self._server.scene.add_box(
                path,
                dimensions=(2 * size[0], 2 * size[1], 2 * size[2]),
                color=color,
                position=pos,
                wxyz=body_wxyz,
            )
        if gtype == "sphere":
            size = _parse_floats(geom.get("size"), 1, (0.01,))
            return self._server.scene.add_icosphere(
                path,
                radius=size[0],
                color=color,
                position=pos,
            )
        if gtype == "cylinder":
            size = _parse_floats(geom.get("size"), 2, (0.01, 0.01))
            return self._server.scene.add_cylinder(
                path,
                radius=size[0],
                height=2 * size[1],
                color=color,
                position=pos,
                wxyz=body_wxyz,
            )
        return None

    def publish_markers(
        self, markers: Iterable[tuple[int, np.ndarray]]
    ) -> None:
        markers = list(markers)
        # Keep pool size in sync; handle counts are tiny so rebuild is fine.
        for h in self._molmo_handles:
            h.remove()
        self._molmo_handles = []
        for i, (mtype, pos) in enumerate(markers):
            t = int(mtype)
            if t == 2:
                # Raw 2D-back-projected anchor — small yellow point so we
                # can see precisely whether it lands inside the early-close
                # wireframe box. Bumped down from 0.06 m for visual clarity.
                color = (1.0, 0.95, 0.1)
                radius = 0.015
            elif t == 0:
                color = (1.0, 0.0, 0.0)
                radius = 0.02
            else:
                color = (0.0, 1.0, 0.0)
                radius = 0.02
            h = self._server.scene.add_icosphere(
                f"/molmo/m{i}",
                radius=radius,
                color=color,
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
            )
            self._molmo_handles.append(h)

    def publish_obb_world(
        self,
        center: np.ndarray | None,
        wxyz: np.ndarray | None,
        extent: np.ndarray | None,
    ) -> None:
        """Render the FFS OBB as a wireframe box in WORLD frame.

        All three args are in world (odom) frame — the same frame
        ``ffs_node`` publishes on ``/molmo/ffs/obb_pose``. Attached at the
        scene root (``/ffs_obb``) rather than under ``/pelvis`` so the
        wireframe stays fixed in the scene as the pelvis pitches / rolls
        during walking, instead of shimmying with the robot.

        Flicker-safe: main viser tick runs at ~60 Hz, OBB updates at ~10 Hz,
        so most calls are identical-geometry redraws. We use ``add_box`` so
        the handle's ``position`` and ``wxyz`` are settable — translate /
        rotate updates mutate in place with no remove+re-add. We only
        recreate the handle when the extent actually changes (EMA smoothing
        eventually locks it, so recreation is rare after seed).
        """
        if center is None or wxyz is None or extent is None:
            if self._ffs_obb_handle is not None:
                self._clear_handle("_ffs_obb_handle")
                self._ffs_obb_last_extent = None
            return
        c = np.asarray(center, dtype=np.float32).reshape(3)
        q = np.asarray(wxyz, dtype=np.float32).reshape(4)
        e = np.asarray(extent, dtype=np.float32).reshape(3)
        if (
            not np.all(np.isfinite(c))
            or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(e))
            or (e <= 0.0).any()
        ):
            return

        pos = (float(c[0]), float(c[1]), float(c[2]))
        wxyz_t = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

        # Recreate the handle only when the extent moves by > 1 cm in any
        # axis — pose updates inside that tolerance go through settable
        # properties and avoid the remove+re-add that caused the flicker.
        need_recreate = (
            self._ffs_obb_handle is None
            or self._ffs_obb_last_extent is None
            or np.any(np.abs(e - self._ffs_obb_last_extent) > 0.01)
        )

        if not need_recreate:
            try:
                self._ffs_obb_handle.position = pos
                self._ffs_obb_handle.wxyz = wxyz_t
                return
            except Exception:
                # Handle went stale (client reconnect, etc); fall through
                # to the recreate path.
                self._clear_handle("_ffs_obb_handle")

        if self._ffs_obb_handle is not None:
            self._clear_handle("_ffs_obb_handle")

        self._ffs_obb_handle = self._server.scene.add_box(
            "/ffs_obb",
            dimensions=(float(e[0]), float(e[1]), float(e[2])),
            wireframe=True,
            color=(0.2, 1.0, 0.9),
            position=pos,
            wxyz=wxyz_t,
        )
        self._ffs_obb_last_extent = e.copy()

    def publish_grasp_boxes(
        self,
        left_pose: tuple[np.ndarray, np.ndarray] | None,
        right_pose: tuple[np.ndarray, np.ndarray] | None,
        dims: tuple[float, float, float],
    ) -> None:
        """Render per-hand wireframe boxes at the early-close trigger volume.

        Each ``*_pose`` is ``(world_pos, world_wxyz)`` for the box centroid
        — caller (sim_node) computes this by taking the wrist link's world
        pose, translating along the wrist-local +x axis to the gripper
        midpoint, and offsetting by ``(X_FWD - X_BACK)/2`` along +x so the
        box centroid sits at the geometric center of the asymmetric
        ``[-X_BACK, +X_FWD]`` × ``[-Y, Y]`` × ``[-Z, Z]`` volume. ``dims``
        is the world-frame size tuple ``(X_BACK + X_FWD, 2*Y, 2*Z)``. Pass
        None to clear a side. Box orientation matches the wrist's frame.
        """
        def _publish_one(handle_attr: str, color: tuple[float, float, float], pose):
            if pose is None:
                if getattr(self, handle_attr, None) is not None:
                    self._clear_handle(handle_attr)
                return
            pos, wxyz = pose
            pos_t = (float(pos[0]), float(pos[1]), float(pos[2]))
            wxyz_t = (float(wxyz[0]), float(wxyz[1]), float(wxyz[2]), float(wxyz[3]))
            need_recreate = (
                getattr(self, handle_attr, None) is None
                or self._grasp_box_last_dims is None
                or any(abs(d - ld) > 1e-4 for d, ld in zip(dims, self._grasp_box_last_dims))
            )
            if not need_recreate:
                try:
                    h = getattr(self, handle_attr)
                    h.position = pos_t
                    h.wxyz = wxyz_t
                    return
                except Exception:
                    self._clear_handle(handle_attr)
            if getattr(self, handle_attr, None) is not None:
                self._clear_handle(handle_attr)
            handle = self._server.scene.add_box(
                f"/grasp_box/{handle_attr.split('_')[-1]}",
                dimensions=(float(dims[0]), float(dims[1]), float(dims[2])),
                wireframe=True,
                color=color,
                position=pos_t,
                wxyz=wxyz_t,
            )
            setattr(self, handle_attr, handle)

        _publish_one("_grasp_box_handle_l", (0.2, 0.9, 1.0), left_pose)
        _publish_one("_grasp_box_handle_r", (1.0, 0.5, 0.2), right_pose)
        self._grasp_box_last_dims = (float(dims[0]), float(dims[1]), float(dims[2]))

    def publish_grasp_midpoints(
        self,
        left_pos: np.ndarray | None,
        right_pos: np.ndarray | None,
    ) -> None:
        """Solid spheres at each gripper finger-pad midpoint so the box's
        reference center is unambiguous from any viewing angle.
        """
        for attr, color, pos in (
            ("_grasp_mid_handle_l", (0.2, 0.9, 1.0), left_pos),
            ("_grasp_mid_handle_r", (1.0, 0.5, 0.2), right_pos),
        ):
            if pos is None:
                if getattr(self, attr, None) is not None:
                    self._clear_handle(attr)
                continue
            pos_t = (float(pos[0]), float(pos[1]), float(pos[2]))
            handle = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.position = pos_t
                    continue
                except Exception:
                    self._clear_handle(attr)
            handle = self._server.scene.add_icosphere(
                f"/grasp_mid/{attr.split('_')[-1]}",
                radius=0.015,
                color=color,
                position=pos_t,
            )
            setattr(self, attr, handle)

    # ---------- planner path / voxel visualization ----------

    @staticmethod
    def _waypoints_to_segments(pts: np.ndarray) -> np.ndarray:
        """Convert (N, 3) polyline into viser's (N-1, 2, 3) segment layout."""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] < 2:
            return np.zeros((0, 2, 3), dtype=np.float32)
        starts = pts[:-1]
        ends = pts[1:]
        return np.stack([starts, ends], axis=1)

    def _clear_handle(self, attr_name: str) -> None:
        handle = getattr(self, attr_name, None)
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass
        setattr(self, attr_name, None)

    def _apply_path_visibility(self) -> None:
        visible = bool(self._show_paths_cb.value)
        for attr in (
            "_loco_path_handle",
            "_loco_goal_handle",
            "_ee_left_handle",
            "_ee_right_handle",
        ):
            h = getattr(self, attr, None)
            if h is not None:
                try:
                    h.visible = visible
                except Exception:
                    pass

    def _apply_voxel_visibility(self) -> None:
        visible = bool(self._show_voxels_cb.value)
        h = getattr(self, "_esdf_voxel_handle", None)
        if h is not None:
            try:
                h.visible = visible
            except Exception:
                pass

    def publish_loco_path(self, waypoints_xy: np.ndarray | None) -> None:
        """Render the world-frame base path as a yellow polyline at z=0.05.

        Called every sim tick (e.g. 50 Hz) but the underlying ROS path
        only refreshes at PLANNER_PLAN_RATE_HZ. Re-adding the viser line
        segment on every sim tick queues redundant websocket traffic,
        which makes the path visibly lag the actual planner output.
        Short-circuit when the waypoints haven't changed so the line
        stays stable between planner ticks and updates promptly when
        the planner does publish something new.
        """
        if waypoints_xy is None:
            if self._loco_path_last is None:
                return
            self._clear_handle("_loco_path_handle")
            self._clear_handle("_loco_goal_handle")
            self._loco_path_last = None
            return
        pts = np.asarray(waypoints_xy, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 1:
            if self._loco_path_last is None:
                return
            self._clear_handle("_loco_path_handle")
            self._clear_handle("_loco_goal_handle")
            self._loco_path_last = None
            return
        if (
            self._loco_path_last is not None
            and self._loco_path_last.shape == pts.shape
            and np.array_equal(self._loco_path_last, pts)
        ):
            return
        self._clear_handle("_loco_path_handle")
        self._clear_handle("_loco_goal_handle")
        pts_3d = np.concatenate(
            [pts, np.full((pts.shape[0], 1), 0.05, dtype=np.float32)], axis=1
        )
        segments = self._waypoints_to_segments(pts_3d)
        if segments.shape[0] > 0:
            self._loco_path_handle = self._server.scene.add_line_segments(
                "/planner/loco_path",
                points=segments,
                colors=(1.0, 0.85, 0.0),
                line_width=3.0,
                visible=bool(self._show_paths_cb.value),
            )
        goal_pt = pts_3d[-1]
        self._loco_goal_handle = self._server.scene.add_icosphere(
            "/planner/loco_goal_snapped",
            radius=0.045,
            color=(0.0, 0.95, 1.0),
            position=(float(goal_pt[0]), float(goal_pt[1]), float(goal_pt[2])),
            visible=bool(self._show_paths_cb.value),
        )
        self._loco_path_last = pts.copy()

    def publish_ee_path(
        self,
        left_waypoints_xyz: np.ndarray | None,
        right_waypoints_xyz: np.ndarray | None,
    ) -> None:
        """Render per-hand EE paths in the pelvis frame (blue=left, red=right)."""
        for attr, wps, name, color in (
            ("_ee_left_handle",  left_waypoints_xyz,  "/pelvis/planner_ee_left",  (0.2, 0.5, 1.0)),
            ("_ee_right_handle", right_waypoints_xyz, "/pelvis/planner_ee_right", (1.0, 0.3, 0.3)),
        ):
            self._clear_handle(attr)
            if wps is None:
                continue
            pts = np.asarray(wps, dtype=np.float32).reshape(-1, 3)
            if pts.shape[0] < 2:
                continue
            segments = self._waypoints_to_segments(pts)
            if segments.shape[0] == 0:
                continue
            handle = self._server.scene.add_line_segments(
                name,
                points=segments,
                colors=color,
                line_width=3.0,
                visible=bool(self._show_paths_cb.value),
            )
            setattr(self, attr, handle)

    def publish_esdf_voxels(
        self,
        centers_world: np.ndarray | None,
        voxel_m: float = 0.05,
    ) -> None:
        """Render ESDF-occupied voxel centers as a translucent orange point cloud."""
        self._clear_handle("_esdf_voxel_handle")
        if centers_world is None:
            return
        pts = np.asarray(centers_world, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 0:
            return
        # Viser point clouds have no alpha, so we simulate translucency
        # with a paler color + soft round shape + sub-voxel point size
        # so adjacent voxels have a visible gap between them.
        colors = np.tile(
            np.array([[1.0, 0.75, 0.5]], dtype=np.float32), (pts.shape[0], 1)
        )
        self._esdf_voxel_handle = self._server.scene.add_point_cloud(
            "/planner/esdf_voxels",
            points=pts,
            colors=colors,
            point_size=float(voxel_m) * 0.2,
            point_shape="rounded",
            visible=bool(self._show_voxels_cb.value),
        )

    # ---------- camera / overlay hooks (called from ROS / MolmoCameraBridge) ----------

    def push_rgb(self, rgb: np.ndarray) -> None:
        with self._state_lock:
            self._latest_rgb = np.ascontiguousarray(rgb)
        self._maybe_push_rgb()

    def set_overlay(
        self,
        points: Iterable[tuple[int, int]],
        status: str,
        voice_recording: bool,
    ) -> None:
        with self._state_lock:
            self._overlay_points = [(int(x), int(y)) for x, y in points]
            self._overlay_status = str(status)
            self._voice_recording = bool(voice_recording)
        self._status_md.content = (
            f"**Status:** {status}"
            + ("  \n**Voice:** recording" if voice_recording else "")
        )
        self._maybe_push_rgb()

    def set_prompt_text(self, text: str) -> None:
        self._prompt_text.value = str(text)

    def set_sim_status(self, text: str) -> None:
        """Update the top-of-panel sim/policy health indicator."""
        self._sim_status_md.content = f"**Sim:** {text}"

    def set_planner_status(self, text: str) -> None:
        """Update the planner_node status line (planning_loco / no_path / ...)."""
        self._planner_status_md.content = f"**Waypoint Planner:** {text}"

    def set_plan(self, plan_text: str) -> None:
        """Render the agent's task-decomposition plan as a persistent panel.

        Empty input is a no-op so the last-displayed plan stays after status
        updates that aren't submissions. Subtasks (one per non-empty line)
        render as a numbered markdown list.
        """
        text = (plan_text or "").strip()
        if not text:
            return
        items = [line.strip() for line in text.splitlines() if line.strip()]
        if not items:
            return
        lines = [f"{i + 1}. {item}" for i, item in enumerate(items)]
        self._plan_md.content = "**Plan:**  \n" + "  \n".join(lines)

    def stop(self) -> None:
        try:
            self._server.stop()
        except Exception:
            pass

    # ---------- internals ----------

    def _maybe_push_rgb(self) -> None:
        now = time.perf_counter()
        if now - self._last_image_t < self._image_period:
            return
        with self._state_lock:
            if self._latest_rgb is None:
                return
            frame = self._latest_rgb.copy()
            points = list(self._overlay_points)
        if _HAVE_CV2:
            h, w = frame.shape[:2]
            for i, (px, py) in enumerate(points):
                if not (0 <= px < w and 0 <= py < h):
                    continue
                color = (0, 255, 255) if i == 0 else (0, 255, 0)
                cv2.circle(frame, (px, py), 7, color, -1)
                cv2.circle(frame, (px, py), 10, (255, 255, 255), 2)
        self._rgb_handle.image = frame
        self._last_image_t = now
