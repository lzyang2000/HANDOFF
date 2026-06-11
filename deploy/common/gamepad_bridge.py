"""Browser-based gamepad bridge shared by the mjlab play flow and sim2sim.

Serves an HTML page that polls the Web Gamepad API, streams absolute state
(vx, vy, yaw, height, hand offsets) over WebSocket to the server,
and exposes the aggregated state via ``GamepadBridgeServer.get_state()``.

Used by:
  * ``src/wbc_mjlab/scripts/play_dual_student_with_xbox.py`` — reads state
    to override the student actor's observations.
  * ``deploy/controller/xbox_node.py --source server`` — reads state and
    republishes as ``Float32MultiArray`` on ``/g1/command``.

The browser integrates continuous axes (triggers → height, D-pad → hand
offset) locally; the server clamps to limits and is
authoritative. Overlay resets to nominal on client disconnect.
"""

from __future__ import annotations

import http.server
import json
import os
import socket
import socketserver
import threading
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class BridgeConfig:
  # Locomotion limits (m/s, rad/s).
  max_vx: float
  max_vy: float
  max_yaw: float
  # Torso height limits.
  height_min: float
  height_max: float
  # Right-hand offset limits (body frame). Left limits mirror y.
  hand_pos_limit_xyz: Sequence[float]
  hand_neg_limit_xyz: Sequence[float]
  left_hand_pos_limit_xyz: Sequence[float]
  left_hand_neg_limit_xyz: Sequence[float]
  # Defaults (what ``reset_state()`` returns to).
  default_height: float
  default_hand_x: float
  default_hand_y: float
  default_hand_z: float
  # Gamepad UI constants (browser-side integration).
  deadzone: float
  tick_hz: float
  gamepad_hand_speed: float
  gamepad_height_speed: float
  # Network.
  host: str = "0.0.0.0"
  http_port: int = 8765
  ws_port: int = 8766
  label: str = "gamepad-bridge"


def bridge_config_from_env(cfg: BridgeConfig) -> BridgeConfig:
  """Apply ``XBOX_BRIDGE_HOST|HTTP_PORT|WS_PORT`` overrides to a config."""
  host = os.environ.get("XBOX_BRIDGE_HOST", cfg.host)
  http_port = int(os.environ.get("XBOX_BRIDGE_HTTP_PORT", str(cfg.http_port)))
  ws_port = int(os.environ.get("XBOX_BRIDGE_WS_PORT", str(http_port + 1)))
  cfg.host = host
  cfg.http_port = http_port
  cfg.ws_port = ws_port
  return cfg


class GamepadBridgeServer:
  """HTTP + WebSocket gamepad bridge with a thread-safe state dict.

  Callers read via ``get_state()`` (returns a copy) and/or compose compound
  updates under ``with bridge.lock: bridge.state[...] = ...``. On client
  disconnect the overlay is reset to the configured defaults.
  """

  def __init__(self, cfg: BridgeConfig):
    self.cfg = cfg
    self.lock = threading.Lock()
    self.state: dict = self._make_defaults()
    self._started = False
    self._clients: set = set()
    self._clients_lock = threading.Lock()

  def _default_right_offset(self) -> np.ndarray:
    c = self.cfg
    return np.array(
      [c.default_hand_x, c.default_hand_y, c.default_hand_z], dtype=np.float32
    )

  def _default_left_offset(self) -> np.ndarray:
    c = self.cfg
    return np.array(
      [c.default_hand_x, -c.default_hand_y, c.default_hand_z], dtype=np.float32
    )

  def _make_defaults(self) -> dict:
    c = self.cfg
    return {
      "vx": 0.0,
      "vy": 0.0,
      "yaw": 0.0,
      "height": c.default_height,
      "right_hand": self._default_right_offset(),
      "left_hand": self._default_left_offset(),
      "loco_active": True,
      "height_active": True,
      "hands_active": True,
    }

  def get_state(self) -> dict:
    with self.lock:
      return {
        k: (v.copy() if isinstance(v, np.ndarray) else v)
        for k, v in self.state.items()
      }

  def reset_state(self, broadcast: bool = True) -> None:
    """Reset overlay to configured defaults (overlay remains active).

    The connected browser is authoritative — it overwrites server state on
    every WS message — so a server-side reset is meaningless on its own.
    When ``broadcast`` is True we also push ``{"reset": true}`` to all
    clients so they reset their local integrated state too.
    """
    with self.lock:
      self.state.update(self._make_defaults())
    if broadcast:
      self._broadcast_reset()

  def _broadcast_reset(self) -> None:
    with self._clients_lock:
      clients = list(self._clients)
    payload = json.dumps({"reset": True})
    for ws in clients:
      try:
        ws.send(payload)
      except Exception:
        pass

  def start(self) -> None:
    if self._started:
      return
    self._started = True
    self._serve_http()
    self._serve_ws()
    ip = _local_ip_hint()
    print(
      f"[{self.cfg.label}] Open  http://{ip}:{self.cfg.http_port}  in a "
      "browser with the controller connected. (Any OS; Web Gamepad API.)"
    )

  # ------------------------------------------------------------ HTTP
  def _serve_http(self) -> None:
    html_bytes = self._html_page().encode("utf-8")
    cfg = self.cfg

    class _Handler(http.server.BaseHTTPRequestHandler):
      def do_GET(h) -> None:  # noqa: N802,N805
        h.send_response(200)
        h.send_header("Content-Type", "text/html; charset=utf-8")
        h.send_header("Content-Length", str(len(html_bytes)))
        h.send_header("Cache-Control", "no-store")
        h.end_headers()
        h.wfile.write(html_bytes)

      def log_message(h, *args, **kwargs) -> None:  # noqa: N805
        return

    class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
      daemon_threads = True
      allow_reuse_address = True

    try:
      server = _Server((cfg.host, cfg.http_port), _Handler)
    except OSError as exc:
      print(f"[{cfg.label}] HTTP bind {cfg.host}:{cfg.http_port} failed: {exc}")
      return
    threading.Thread(
      target=server.serve_forever, name=f"{cfg.label}-http", daemon=True
    ).start()

  # ------------------------------------------------------------ WebSocket
  def _serve_ws(self) -> None:
    try:
      from websockets.sync.server import serve
    except ImportError:
      print(
        f"[{self.cfg.label}] 'websockets' not installed; browser bridge disabled."
      )
      return

    cfg = self.cfg

    def _handler(websocket) -> None:
      peer = websocket.remote_address
      print(f"[{cfg.label}] client connected from {peer}")
      with self._clients_lock:
        self._clients.add(websocket)
      try:
        for msg in websocket:
          try:
            data = json.loads(msg)
          except (TypeError, ValueError):
            continue
          if data.get("reset"):
            # Browser already reset its own state; don't echo back.
            self.reset_state(broadcast=False)
            continue
          self._apply_ws_update(data)
      finally:
        with self._clients_lock:
          self._clients.discard(websocket)
        self.reset_state(broadcast=False)
        print(f"[{cfg.label}] client {peer} disconnected; overlay released.")

    def _run() -> None:
      try:
        with serve(_handler, cfg.host, cfg.ws_port) as server:
          server.serve_forever()
      except OSError as exc:
        print(f"[{cfg.label}] WS bind {cfg.host}:{cfg.ws_port} failed: {exc}")

    threading.Thread(target=_run, name=f"{cfg.label}-ws", daemon=True).start()

  def _apply_ws_update(self, data: dict) -> None:
    """Browser is authoritative when connected; overwrite state each msg."""
    c = self.cfg
    try:
      vx = float(data.get("vx", 0.0))
      vy = float(data.get("vy", 0.0))
      yaw = float(data.get("yaw", 0.0))
      height = float(data.get("height", c.default_height))
      right = [float(x) for x in data.get("right_hand", self._default_right_offset())]
      left = [float(x) for x in data.get("left_hand", self._default_left_offset())]
    except (TypeError, ValueError):
      return

    hp = np.array(c.hand_pos_limit_xyz, dtype=np.float32)
    hn = np.array(c.hand_neg_limit_xyz, dtype=np.float32)
    lhp = np.array(c.left_hand_pos_limit_xyz, dtype=np.float32)
    lhn = np.array(c.left_hand_neg_limit_xyz, dtype=np.float32)

    with self.lock:
      self.state["vx"] = float(np.clip(vx, -c.max_vx, c.max_vx))
      self.state["vy"] = float(np.clip(vy, -c.max_vy, c.max_vy))
      self.state["yaw"] = float(np.clip(yaw, -c.max_yaw, c.max_yaw))
      self.state["height"] = float(np.clip(height, c.height_min, c.height_max))
      self.state["right_hand"] = np.clip(
        np.array(right, dtype=np.float32), hn, hp
      )
      self.state["left_hand"] = np.clip(
        np.array(left, dtype=np.float32), lhn, lhp
      )

  # ------------------------------------------------------------ HTML
  def _html_page(self) -> str:
    c = self.cfg
    hp, hn = c.hand_pos_limit_xyz, c.hand_neg_limit_xyz
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{c.label}</title>
<style>
  body {{ font-family: ui-monospace, Menlo, monospace; background:#1a1a1a; color:#eee; padding:20px; max-width:620px; margin:0 auto; }}
  h2 {{ margin-top:0; }}
  h3 {{ margin-bottom:4px; color:#8ef; }}
  .row {{ margin:6px 0; }}
  .label {{ display:inline-block; width:80px; }}
  .val {{ display:inline-block; width:80px; text-align:right; }}
  .track {{ position:relative; height:10px; background:#333; border-radius:5px; margin-top:4px; }}
  .needle {{ position:absolute; top:-2px; width:4px; height:14px; background:#4caf50; border-radius:2px; }}
  .mid {{ position:absolute; top:0; left:50%; width:1px; height:100%; background:#666; }}
  #status {{ padding:6px 10px; border-radius:4px; display:inline-block; }}
  .ok {{ background:#1b5e20; }}
  .err {{ background:#b71c1c; }}
  .pending {{ background:#616161; }}
  footer {{ margin-top:20px; font-size:12px; color:#888; }}
</style>
</head>
<body>
<h2>{c.label}</h2>
<div class="row">
  WebSocket: <span id="status" class="pending">connecting…</span>
</div>
<div class="row">
  Pad: <span id="pad">(press any button to wake)</span>
</div>

<h3>Locomotion</h3>
<div class="row"><span class="label">vx</span><span class="val" id="vx">+0.000</span>
  <div class="track"><div class="mid"></div><div class="needle" id="vx-bar"></div></div></div>
<div class="row"><span class="label">vy</span><span class="val" id="vy">+0.000</span>
  <div class="track"><div class="mid"></div><div class="needle" id="vy-bar"></div></div></div>
<div class="row"><span class="label">yaw</span><span class="val" id="yaw">+0.000</span>
  <div class="track"><div class="mid"></div><div class="needle" id="yaw-bar"></div></div></div>

<h3>Torso</h3>
<div class="row"><span class="label">height</span><span class="val" id="height">{c.default_height:.3f}</span></div>

<h3>Right hand offset (body frame)</h3>
<div class="row"><span class="label">r-x</span><span class="val" id="rx">{c.default_hand_x:+.3f}</span></div>
<div class="row"><span class="label">r-y</span><span class="val" id="ry">{c.default_hand_y:+.3f}</span></div>
<div class="row"><span class="label">r-z</span><span class="val" id="rz">{c.default_hand_z:+.3f}</span></div>

<h3>Left hand offset (mirrored from right on Y)</h3>
<div class="row"><span class="label">l-x</span><span class="val" id="lx">{c.default_hand_x:+.3f}</span></div>
<div class="row"><span class="label">l-y</span><span class="val" id="ly_val">{-c.default_hand_y:+.3f}</span></div>
<div class="row"><span class="label">l-z</span><span class="val" id="lz">{c.default_hand_z:+.3f}</span></div>

<h3>Raw gamepad (diagnostic)</h3>
<div class="row"><span class="label">btns</span><span id="raw_btn" style="font-size:12px;color:#cfc">—</span></div>
<div class="row"><span class="label">axes</span><span id="raw_axis" style="font-size:12px;color:#cfc">—</span></div>

<footer>
  Sticks = vx/vy/yaw · LT/RT = height - / + · D-pad = right-hand X/Y · LB/RB = right-hand Z · Start/Back = reset to nominal.<br/>
  Server clamps to limits and is authoritative. Close the tab or unplug the pad → overlay resets to nominal on the server.<br/>
  <br/>
  W3C standard: <code>buttons[0]=A, [3]=Y, [4]=LB, [5]=RB, [6]=LT, [7]=RT, [8]=Back, [9]=Start, [12..15]=dpad</code>
  <br/>Standard-mapping fallback: some xbox+Linux+Chrome combos still report triggers on <code>axes[4]=LT, axes[5]=RT</code> (idle ≈ -1) even when <code>mapping === 'standard'</code> — we OR them into the button reading once an axis is observed below -0.5.
  <br/>Xbox Linux classic (mapping !== 'standard'): triggers on <code>axes[4]=LT, axes[5]=RT</code>, dpad on <code>axes[6]=x, axes[7]=y</code>.
</footer>
<script>
  const WS_URL = `ws://${{location.hostname}}:{c.ws_port}`;
  const DEADZONE = {c.deadzone};
  const TICK_HZ = {c.tick_hz};
  const MAX_VX = {c.max_vx}, MAX_VY = {c.max_vy}, MAX_YAW = {c.max_yaw};
  const HEIGHT_MIN = {c.height_min}, HEIGHT_MAX = {c.height_max};
  const HAND_POS = [{hp[0]}, {hp[1]}, {hp[2]}];
  const HAND_NEG = [{hn[0]}, {hn[1]}, {hn[2]}];
  const HAND_SPEED = {c.gamepad_hand_speed};
  const HEIGHT_SPEED = {c.gamepad_height_speed};
  const DEFAULTS = {{
    height: {c.default_height},
    rx: {c.default_hand_x}, ry: {c.default_hand_y}, rz: {c.default_hand_z},
  }};
  // Reset glides integrated state back to defaults over ~RESET_TAU seconds
  // (frame-rate-independent EMA) instead of snapping. Avoids a discontinuity
  // in the published command. Triggered by Start/Back on the pad OR by an
  // inbound {{reset:true}} WS message from the server (viser reset button).
  const RESET_TAU = 0.3;
  const RESET_EPS = 1e-3;

  const statusEl = document.getElementById('status');
  const padEl = document.getElementById('pad');
  let ws = null;
  let state = {{
    height: DEFAULTS.height,
    rx: DEFAULTS.rx, ry: DEFAULTS.ry, rz: DEFAULTS.rz,
  }};
  let lastStart = 0;
  let resetActive = false;
  function requestReset() {{ resetActive = true; }}
  function lerpField(cur, tgt, alpha) {{ return cur + (tgt - cur) * alpha; }}
  // Some "standard" pads (notably xbox-on-Linux in Chrome/Firefox) report
  // analog triggers on axes[4]/[5] (idle ≈ -1, pressed = +1) instead of
  // populating buttons[6]/[7].value. We OR the two sources, but only after
  // we've observed the axis at < -0.5 — otherwise an uninitialized axis at 0
  // would map to 0.5 and constantly drive the height.
  let triggerAxisActive = {{ lt: false, rt: false }};

  function connect() {{
    ws = new WebSocket(WS_URL);
    ws.onopen = () => {{ statusEl.textContent = 'connected'; statusEl.className = 'ok'; }};
    ws.onclose = () => {{
      statusEl.textContent = 'disconnected – retrying…'; statusEl.className = 'err';
      setTimeout(connect, 1000);
    }};
    ws.onerror = () => {{ try {{ ws.close(); }} catch (e) {{}} }};
    ws.onmessage = (event) => {{
      // Server can push {{ reset: true }} to kick off a smooth reset (e.g.
      // when the user clicks "Reset xbox state" in the viser UI). The
      // browser integrates the EMA locally so the server sees a glide.
      try {{
        const data = JSON.parse(event.data);
        if (data && data.reset) requestReset();
      }} catch (e) {{}}
    }};
  }}
  connect();

  function dz(v) {{ return Math.abs(v) < DEADZONE ? 0 : v; }}
  function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}
  function setBar(id, v) {{
    const pct = Math.max(0, Math.min(1, (v + 1) / 2));
    document.getElementById(id).style.left = (pct * 100) + '%';
  }}
  function fmt(v) {{ return (v >= 0 ? '+' : '') + v.toFixed(3); }}

  const GOOD_ID = /xbox|xinput|microsoft|x-box|8bitdo|sony|playstation|dualshock|dualsense|nintendo|switch pro|gamepad|controller|joystick|pad/i;
  const BAD_ID  = /keyboard|kbd|mouse|trackpad|receiver/i;

  function pickPad(pads) {{
    const nonNull = [];
    for (const c of pads) {{ if (c) nonNull.push(c); }}
    if (nonNull.length === 0) return {{ pad: null, isUnknown: false, others: '' }};
    const standard = nonNull.filter(p => p.mapping === 'standard');
    const pad =
      standard.find(p => !BAD_ID.test(p.id)) ||
      standard.find(p =>  GOOD_ID.test(p.id)) ||
      standard[0] ||
      nonNull.find(p => GOOD_ID.test(p.id)) ||
      nonNull[0];
    const isUnknown = BAD_ID.test(pad.id) || (pad.mapping !== 'standard' && !GOOD_ID.test(pad.id));
    const others = nonNull.filter(p => p !== pad).map(p => p.id).join(', ');
    return {{ pad, isUnknown, others }};
  }}

  let lastTick = performance.now();
  function poll() {{
    const now = performance.now();
    const dt = Math.min(0.1, (now - lastTick) / 1000);
    lastTick = now;

    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    const {{ pad: p, isUnknown, others }} = pickPad(pads);
    if (p) {{
      const mapping = p.mapping || '(none)';
      const base = isUnknown
        ? `unrecognized device \u2014 ${{p.id}} (mapping: ${{mapping}})`
        : `${{p.id}} (mapping: ${{mapping}})`;
      padEl.textContent = others ? `${{base}}  |  +${{others.split(', ').length}} other: ${{others}}` : base;
      padEl.className = isUnknown ? 'err' : '';

      const lx = dz(p.axes[0] || 0);
      const ly = dz(p.axes[1] || 0);
      const rx = dz(p.axes[2] || 0);

      // Two layouts, auto-selected on p.mapping:
      //
      // (1) W3C Standard Gamepad (`p.mapping === 'standard'`) — Firefox/Chrome
      //     remap recognized pads (Xbox Wireless, DualShock/DualSense, etc.)
      //     to fixed indices per https://www.w3.org/TR/gamepad/#remapping :
      //       buttons[0]=A, [1]=B, [2]=X, [3]=Y, [4]=LB, [5]=RB,
      //               [6]=LT (analog value), [7]=RT (analog value),
      //               [8]=Back, [9]=Start, [10]=L3, [11]=R3,
      //               [12..15]=D-pad up/dn/lt/rt.
      //
      // (2) Classic Linux SDL xinput (anything not remapped; verified against
      //     8BitDo Ultimate 2C, `mapping: "(none)"`):
      //       buttons[0]=A, [1]=B, [3]=X, [4]=Y, [6]=LB, [7]=RB, [8]=LT, [9]=RT
      //               (LT/RT as pressed/value bools; axes[3]=RT, axes[4]=LT
      //                have ambiguous idle, so we skip them).
      //       axes[9] = POV hat: -1=up, -0.71=up-right, -0.43=right,
      //               -0.14=dn-right, 0.14=down, 0.43=dn-left, 0.71=left,
      //               1.0=up-left, |v|>1 released.
      let lt, rt, lb, rb, dUp, dDn, dLt, dRt, start;
      if (p.mapping === 'standard') {{
        const ax4 = p.axes[4], ax5 = p.axes[5];
        if (ax4 !== undefined && ax4 < -0.5) triggerAxisActive.lt = true;
        if (ax5 !== undefined && ax5 < -0.5) triggerAxisActive.rt = true;
        const ltAx = triggerAxisActive.lt ? Math.max(0, (ax4 + 1) / 2) : 0;
        const rtAx = triggerAxisActive.rt ? Math.max(0, (ax5 + 1) / 2) : 0;
        lt = Math.max((p.buttons[6] && p.buttons[6].value) || 0, ltAx);
        rt = Math.max((p.buttons[7] && p.buttons[7].value) || 0, rtAx);
        lb = (p.buttons[4] && p.buttons[4].pressed) ? 1 : 0;
        rb = (p.buttons[5] && p.buttons[5].pressed) ? 1 : 0;
        dUp = (p.buttons[12] && p.buttons[12].pressed) ? 1 : 0;
        dDn = (p.buttons[13] && p.buttons[13].pressed) ? 1 : 0;
        dLt = (p.buttons[14] && p.buttons[14].pressed) ? 1 : 0;
        dRt = (p.buttons[15] && p.buttons[15].pressed) ? 1 : 0;
        start =
          (p.buttons[9] && p.buttons[9].pressed) ||
          (p.buttons[8] && p.buttons[8].pressed);
      }} else {{
        lt = (p.buttons[8] && p.buttons[8].value) || 0;
        rt = (p.buttons[9] && p.buttons[9].value) || 0;
        lb = (p.buttons[6] && p.buttons[6].pressed) ? 1 : 0;
        rb = (p.buttons[7] && p.buttons[7].pressed) ? 1 : 0;

        dUp = 0; dDn = 0; dLt = 0; dRt = 0;
        const hat = p.axes[9];
        if (hat !== undefined && hat >= -1.05 && hat <= 1.05) {{
          if (hat < -0.85)       {{ dUp = 1; }}
          else if (hat < -0.57)  {{ dUp = 1; dRt = 1; }}
          else if (hat < -0.28)  {{ dRt = 1; }}
          else if (hat <  0.00)  {{ dDn = 1; dRt = 1; }}
          else if (hat <  0.28)  {{ dDn = 1; }}
          else if (hat <  0.57)  {{ dDn = 1; dLt = 1; }}
          else if (hat <  0.85)  {{ dLt = 1; }}
          else                   {{ dUp = 1; dLt = 1; }}
        }}

        // Start/Back tentative for classic layout; press Start once and check
        // the raw btns row if reset doesn't fire — then swap the index here.
        start =
          (p.buttons[10] && p.buttons[10].pressed) ||
          (p.buttons[11] && p.buttons[11].pressed);
      }}

      // Raw diagnostic panel.
      const btnToks = [];
      for (let i = 0; i < p.buttons.length; i++) {{
        const b = p.buttons[i];
        if (!b) continue;
        if (b.pressed || b.value > 0.05) {{
          btnToks.push(`${{i}}:${{b.value.toFixed(2)}}`);
        }}
      }}
      const axToks = [];
      for (let i = 0; i < p.axes.length; i++) {{
        const v = p.axes[i];
        if (v === undefined) continue;
        if (Math.abs(v) > 0.1) {{
          axToks.push(`${{i}}:${{v.toFixed(2)}}`);
        }}
      }}
      document.getElementById('raw_btn').textContent = btnToks.join(' ') || '—';
      document.getElementById('raw_axis').textContent = axToks.join(' ') || '—';

      let vx = -ly, vy = -lx, yaw = -rx;

      const scale = dt * TICK_HZ;
      if (resetActive) {{
        const a = 1 - Math.exp(-dt / RESET_TAU);
        state.height = lerpField(state.height, DEFAULTS.height, a);
        state.rx     = lerpField(state.rx,     DEFAULTS.rx,     a);
        state.ry     = lerpField(state.ry,     DEFAULTS.ry,     a);
        state.rz     = lerpField(state.rz,     DEFAULTS.rz,     a);
        if (Math.abs(state.height - DEFAULTS.height) < RESET_EPS &&
            Math.abs(state.rx     - DEFAULTS.rx)     < RESET_EPS &&
            Math.abs(state.ry     - DEFAULTS.ry)     < RESET_EPS &&
            Math.abs(state.rz     - DEFAULTS.rz)     < RESET_EPS) {{
          state.height = DEFAULTS.height;
          state.rx = DEFAULTS.rx; state.ry = DEFAULTS.ry; state.rz = DEFAULTS.rz;
          resetActive = false;
        }}
      }} else {{
        if (rt > 0.1) state.height = clamp(state.height + HEIGHT_SPEED * rt * scale, HEIGHT_MIN, HEIGHT_MAX);
        if (lt > 0.1) state.height = clamp(state.height - HEIGHT_SPEED * lt * scale, HEIGHT_MIN, HEIGHT_MAX);
        if (dUp) state.rx = clamp(state.rx + HAND_SPEED * scale, HAND_NEG[0], HAND_POS[0]);
        if (dDn) state.rx = clamp(state.rx - HAND_SPEED * scale, HAND_NEG[0], HAND_POS[0]);
        if (dLt) state.ry = clamp(state.ry + HAND_SPEED * scale, HAND_NEG[1], HAND_POS[1]);
        if (dRt) state.ry = clamp(state.ry - HAND_SPEED * scale, HAND_NEG[1], HAND_POS[1]);
        if (rb)  state.rz = clamp(state.rz + HAND_SPEED * scale, HAND_NEG[2], HAND_POS[2]);
        if (lb)  state.rz = clamp(state.rz - HAND_SPEED * scale, HAND_NEG[2], HAND_POS[2]);
      }}

      if (start) {{
        if (now - lastStart > 300) {{
          requestReset();
          vx = vy = yaw = 0;
          lastStart = now;
        }}
      }}

      document.getElementById('vx').textContent = fmt(vx);
      document.getElementById('vy').textContent = fmt(vy);
      document.getElementById('yaw').textContent = fmt(yaw);
      setBar('vx-bar', vx); setBar('vy-bar', vy); setBar('yaw-bar', yaw);
      document.getElementById('height').textContent = state.height.toFixed(3);
      document.getElementById('rx').textContent = fmt(state.rx);
      document.getElementById('ry').textContent = fmt(state.ry);
      document.getElementById('rz').textContent = fmt(state.rz);
      document.getElementById('lx').textContent = fmt(state.rx);
      document.getElementById('ly_val').textContent = fmt(-state.ry);
      document.getElementById('lz').textContent = fmt(state.rz);

      if (ws && ws.readyState === 1) {{
        ws.send(JSON.stringify({{
          vx: vx * MAX_VX, vy: vy * MAX_VY, yaw: yaw * MAX_YAW,
          height: state.height,
          right_hand: [state.rx, state.ry, state.rz],
          left_hand: [state.rx, -state.ry, state.rz],
          sticks_touched: (vx !== 0 || vy !== 0 || yaw !== 0),
          triggers_touched: (rt > 0.1 || lt > 0.1),
          hands_touched: !!(dUp || dDn || dLt || dRt || lb || rb),
          reset: !!start,
        }}));
      }}
    }}
    requestAnimationFrame(poll);
  }}
  requestAnimationFrame(poll);
</script>
</body>
</html>
"""


def _local_ip_hint() -> str:
  try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
      s.connect(("8.8.8.8", 80))
      return s.getsockname()[0]
  except OSError:
    return "<server-ip>"
