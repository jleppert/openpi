"""Interactive MuJoCo viewer with keyboard jog controls and camera editing for the JAKA Zu5.

Press ENTER to cycle modes: NAVIGATE -> JOG -> CAMERA -> NAVIGATE

Jog mode controls:
  LEFT / RIGHT  — J1 (base rotation)
  UP / DOWN     — J2 (shoulder)
  W / S         — J3 (elbow)
  A / D         — gripper open / close
  1-5           — set step size (1=tiny .. 5=large)
  R             — reset to home keyframe

Camera mode controls:
  Tab           — switch active camera (external_cam / wrist_cam)
  LEFT / RIGHT  — adjust X position
  UP / DOWN     — adjust Y position
  W / S         — adjust Z position
  Q / E         — adjust FOV (-/+ 1 degree)
  1-5           — set position step size

Camera preview windows (mouse controls, always active):
  Left-drag     — orbit (rotate around lookat point)
  Right-drag    — pan (translate lookat point)
  Scroll        — zoom (change distance)
  Middle-click  — undo mouse edits, restore original camera
"""

import enum
import pathlib
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_LINEAR,
    GL_QUADS,
    GL_RGB,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glClear,
    glEnable,
    glEnd,
    glGenTextures,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex2f,
)

_ASSETS_DIR = pathlib.Path(__file__).parent / "assets"
_MJCF_PATH = _ASSETS_DIR / "jaka_zu5.xml"

_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
_ARM_ACTUATOR_NAMES = [f"act_joint{i}" for i in range(1, 7)]
_GRIPPER_ACTUATOR_NAME = "act_gripper"
_GRIPPER_MAX_OPEN = 0.04

# Step sizes selectable with keys 1-5.
_STEP_SIZES = {1: 0.005, 2: 0.01, 3: 0.02, 4: 0.05, 5: 0.1}

_CAMERA_NAMES = ["external_cam", "wrist_cam"]
_RENDER_WIDTH = 640
_RENDER_HEIGHT = 480

# GLFW key codes (used by MuJoCo key callback).
_KEY_ENTER = 257
_KEY_TAB = 258
_KEY_RIGHT = 262
_KEY_LEFT = 263
_KEY_DOWN = 264
_KEY_UP = 265


class Mode(enum.Enum):
    NAVIGATE = "NAVIGATE"
    JOG = "JOG"
    CAMERA = "CAMERA"


_MODE_ORDER = [Mode.NAVIGATE, Mode.JOG, Mode.CAMERA]


def _next_mode(current: Mode) -> Mode:
    idx = _MODE_ORDER.index(current)
    return _MODE_ORDER[(idx + 1) % len(_MODE_ORDER)]


def _print_mode(mode: Mode):
    if mode == Mode.NAVIGATE:
        print("  [NAVIGATE] MuJoCo default controls.  ENTER=next mode")
    elif mode == Mode.JOG:
        print("  [JOG] Arrows=J1/J2  W/S=J3  A/D=gripper  1-5=step  R=reset  ENTER=next mode")
    elif mode == Mode.CAMERA:
        print("  [CAMERA] Tab=switch cam  Arrows=X/Y  W/S=Z  Q/E=FOV  1-5=step  ENTER=next mode")


def _camera_to_mjcf(model: mujoco.MjModel, cam_id: int, cam_name: str) -> str:
    """Format current camera params as MJCF XML for copy-paste."""
    pos = model.cam_pos[cam_id]
    fovy = model.cam_fovy[cam_id]
    mat = model.cam_mat0[cam_id].reshape(3, 3)
    x_axis = mat[:, 0]
    y_axis = mat[:, 1]
    xyaxes = np.concatenate([x_axis, y_axis])
    pos_str = " ".join(f"{v:.4f}" for v in pos)
    xy_str = " ".join(f"{v:.4f}" for v in xyaxes)
    return f'<camera name="{cam_name}" pos="{pos_str}" xyaxes="{xy_str}" fovy="{fovy:.0f}"/>'


# ---------------------------------------------------------------------------
# GLFW preview windows with mouse orbit / zoom / pan.
#
# Mouse edits write directly to model.cam_pos / model.cam_mat0 and
# data.cam_xpos / data.cam_xmat so the preview always renders in FIXED mode
# with the correct FOV.  This means what-you-see-is-what-you-get: the MJCF
# printed after orbiting will look identical when loaded back.
# ---------------------------------------------------------------------------

def _orbit_to_pos_mat(lookat, distance, azimuth_deg, elevation_deg):
    """Compute world-frame camera position and 3x3 orientation from orbit params."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    pos = np.array([
        lookat[0] + distance * np.cos(el) * np.cos(az),
        lookat[1] + distance * np.cos(el) * np.sin(az),
        lookat[2] + distance * np.sin(el),
    ])
    forward = lookat - pos
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    # Columns: x=right, y=up, z=-forward (OpenGL convention).
    cam_mat = np.column_stack([right, up, -forward])
    return pos, cam_mat


class _PreviewWindow:
    """GLFW window showing a MuJoCo camera render with mouse orbit/zoom/pan.

    Always renders in FIXED mode.  Mouse interactions update orbit state and
    write the resulting position/orientation back to the MuJoCo model so that
    the preview uses the camera's real FOV and the MJCF output is accurate.
    """

    def __init__(self, title: str, width: int, height: int, model: mujoco.MjModel, cam_id: int):
        glfw.window_hint(glfw.VISIBLE, True)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, title, None, None)
        if not self.win:
            raise RuntimeError(f"Failed to create GLFW window: {title}")

        # OpenGL texture setup.
        glfw.make_context_current(self.win)
        glEnable(GL_TEXTURE_2D)
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glfw.make_context_current(None)

        self._model = model
        self._cam_id = cam_id
        self._cam_name = title

        # Always FIXED mode — we write pos/orientation to model directly.
        self._fixed_cam = mujoco.MjvCamera()
        self._fixed_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._fixed_cam.fixedcamid = cam_id

        # Orbit state (lazily initialized on first mouse interaction).
        self._orbit_active = False
        self._orbit_init_pending = False
        self._lookat = np.zeros(3)
        self._distance = 1.0
        self._azimuth = 0.0
        self._elevation = 0.0

        # Original model values for middle-click restore.
        self._orig_pos = model.cam_pos[cam_id].copy()
        self._orig_mat = model.cam_mat0[cam_id].copy()

        # Camera changed — triggers MJCF print on next frame.
        self._dirty = False

        # Mouse state.
        self._button_left = False
        self._button_right = False
        self._last_x = 0.0
        self._last_y = 0.0

        # Register GLFW mouse callbacks.
        glfw.set_mouse_button_callback(self.win, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.win, self._on_cursor_pos)
        glfw.set_scroll_callback(self.win, self._on_scroll)

    # --- Mouse callbacks ---------------------------------------------------

    def _on_mouse_button(self, win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            was = self._button_left
            self._button_left = (action == glfw.PRESS)
            if was and not self._button_left:
                self._dirty = True
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            was = self._button_right
            self._button_right = (action == glfw.PRESS)
            if was and not self._button_right:
                self._dirty = True
        elif button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.PRESS:
            self._restore_original()
            print(f"    {self._cam_name}: restored original camera")
            return
        self._last_x, self._last_y = glfw.get_cursor_pos(win)
        if not self._orbit_active and (self._button_left or self._button_right):
            self._orbit_init_pending = True

    def _on_cursor_pos(self, win, x, y):
        dx = x - self._last_x
        dy = y - self._last_y
        self._last_x = x
        self._last_y = y
        if not self._orbit_active:
            return
        if self._button_left:  # Orbit.
            self._azimuth -= dx * 0.3
            self._elevation -= dy * 0.3
            self._elevation = max(-89, min(89, self._elevation))
        elif self._button_right:  # Pan.
            scale = self._distance * 0.002
            az = np.radians(self._azimuth)
            right = np.array([-np.sin(az), np.cos(az), 0.0])
            self._lookat -= right * dx * scale
            self._lookat[2] += dy * scale

    def _on_scroll(self, win, xoff, yoff):
        if not self._orbit_active:
            self._orbit_init_pending = True
            return
        self._distance *= 0.9 if yoff > 0 else 1.1
        self._distance = max(0.1, self._distance)
        self._dirty = True

    # --- Orbit management --------------------------------------------------

    def _init_orbit(self, data: mujoco.MjData):
        """Initialize orbit params from the camera's current world pose."""
        pos = data.cam_xpos[self._cam_id].copy()
        mat = data.cam_xmat[self._cam_id].reshape(3, 3)
        forward = -mat[:, 2]  # camera looks along local -z
        # Place lookat 1m ahead of camera.
        self._lookat = pos + forward * 1.0
        diff = pos - self._lookat
        self._distance = np.linalg.norm(diff)
        if self._distance < 0.01:
            self._distance = 1.0
            diff = np.array([1.0, 0.0, 0.0])
        self._azimuth = np.degrees(np.arctan2(diff[1], diff[0]))
        self._elevation = np.degrees(np.arcsin(np.clip(diff[2] / self._distance, -1, 1)))
        self._orbit_active = True
        self._orbit_init_pending = False

    def _apply_orbit_to_model(self, data: mujoco.MjData):
        """Write orbit-derived pos/orientation to model and data."""
        world_pos, cam_mat = _orbit_to_pos_mat(
            self._lookat, self._distance, self._azimuth, self._elevation,
        )
        body_id = self._model.cam_bodyid[self._cam_id]
        if body_id == 0:
            self._model.cam_pos[self._cam_id] = world_pos
            self._model.cam_mat0[self._cam_id] = cam_mat.flatten()
        else:
            body_pos = data.xpos[body_id]
            body_mat = data.xmat[body_id].reshape(3, 3)
            self._model.cam_pos[self._cam_id] = body_mat.T @ (world_pos - body_pos)
            self._model.cam_mat0[self._cam_id] = (body_mat.T @ cam_mat).flatten()
        # Write directly to data for immediate rendering (avoids 1-frame lag).
        data.cam_xpos[self._cam_id] = world_pos
        data.cam_xmat[self._cam_id] = cam_mat.flatten()

    def _restore_original(self):
        """Middle-click: restore model camera to its original XML values."""
        self._model.cam_pos[self._cam_id] = self._orig_pos.copy()
        self._model.cam_mat0[self._cam_id] = self._orig_mat.copy()
        self._orbit_active = False
        self._orbit_init_pending = False
        self._dirty = True

    def deactivate_orbit(self):
        """Deactivate orbit (call after keyboard edits to model.cam_pos)."""
        self._orbit_active = False
        self._orbit_init_pending = False

    # --- Render & state ----------------------------------------------------

    def consume_dirty(self) -> bool:
        if self._dirty:
            self._dirty = False
            return True
        return False

    def render_and_display(self, renderer: mujoco.Renderer, data: mujoco.MjData):
        """Render the camera view and blit to the GLFW window."""
        if self._orbit_init_pending:
            self._init_orbit(data)
        if self._orbit_active:
            self._apply_orbit_to_model(data)

        renderer.update_scene(data, camera=self._fixed_cam)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        rgb = renderer.render()

        glfw.make_context_current(self.win)
        h, w = rgb.shape[:2]
        glClear(GL_COLOR_BUFFER_BIT)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(-1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, 1)
        glTexCoord2f(0, 0); glVertex2f(-1, 1)
        glEnd()
        glfw.swap_buffers(self.win)
        glfw.make_context_current(None)

    def alive(self) -> bool:
        return not glfw.window_should_close(self.win)

    def destroy(self):
        glfw.destroy_window(self.win)


def main():
    model = mujoco.MjModel.from_xml_path(str(_MJCF_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    arm_actuator_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in _ARM_ACTUATOR_NAMES
    ]
    gripper_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, _GRIPPER_ACTUATOR_NAME)

    # Camera IDs for editing.
    cam_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) for name in _CAMERA_NAMES
    }

    step_size = _STEP_SIZES[3]  # default medium
    gripper_pos = data.ctrl[gripper_actuator_id]
    mode = Mode.NAVIGATE
    active_cam_idx = 0  # index into _CAMERA_NAMES

    # Offscreen renderer for camera previews.
    renderer = mujoco.Renderer(model, height=_RENDER_HEIGHT, width=_RENDER_WIDTH)

    # GLFW preview windows for each camera (with mouse controls).
    preview_wins = {}
    for cam_name in _CAMERA_NAMES:
        cid = cam_ids[cam_name]
        preview_wins[cam_name] = _PreviewWindow(cam_name, _RENDER_WIDTH, _RENDER_HEIGHT, model, cid)

    # Pending actions from key callback (thread-safe via list).
    pending = []

    def key_callback(key):
        pending.append(key)

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

    # Real-time sync: step physics to match wall-clock time.
    wall_start = time.monotonic()
    sim_start = data.time

    print("Press ENTER in the viewer window to cycle modes.")
    print("Camera preview windows: left-drag=orbit  right-drag=pan  scroll=zoom  middle-click=restore\n")
    _print_mode(mode)

    try:
        while viewer.is_running():
            # Process pending key events.
            while pending:
                key = pending.pop(0)

                # ENTER cycles mode.
                if key == _KEY_ENTER:
                    mode = _next_mode(mode)
                    _print_mode(mode)
                    continue

                # --- JOG mode keys ---
                if mode == Mode.JOG:
                    # Step size selection: keys '1'-'5' (ASCII 49-53).
                    if 49 <= key <= 53:
                        level = key - 48
                        step_size = _STEP_SIZES[level]
                        print(f"    step size: {step_size:.3f} rad  (level {level})")
                        continue

                    # R = reset (ASCII 82).
                    if key == 82:
                        mujoco.mj_resetDataKeyframe(model, data, 0)
                        mujoco.mj_forward(model, data)
                        gripper_pos = data.ctrl[gripper_actuator_id]
                        print("    reset to home")
                        continue

                    # J1: LEFT/RIGHT
                    if key == _KEY_LEFT:
                        data.ctrl[arm_actuator_ids[0]] -= step_size
                    elif key == _KEY_RIGHT:
                        data.ctrl[arm_actuator_ids[0]] += step_size
                    # J2: UP/DOWN
                    elif key == _KEY_UP:
                        data.ctrl[arm_actuator_ids[1]] -= step_size
                    elif key == _KEY_DOWN:
                        data.ctrl[arm_actuator_ids[1]] += step_size
                    # J3: W/S (ASCII 87, 83)
                    elif key == 87:  # W
                        data.ctrl[arm_actuator_ids[2]] -= step_size
                    elif key == 83:  # S
                        data.ctrl[arm_actuator_ids[2]] += step_size
                    # Gripper: A/D (ASCII 65, 68)
                    elif key == 65:  # A = open
                        gripper_pos = min(gripper_pos + 0.005, _GRIPPER_MAX_OPEN)
                        data.ctrl[gripper_actuator_id] = gripper_pos
                    elif key == 68:  # D = close
                        gripper_pos = max(gripper_pos - 0.005, 0.0)
                        data.ctrl[gripper_actuator_id] = gripper_pos

                    # Clamp actuator controls to their ranges.
                    for aid in arm_actuator_ids:
                        lo = model.actuator_ctrlrange[aid, 0]
                        hi = model.actuator_ctrlrange[aid, 1]
                        data.ctrl[aid] = max(lo, min(hi, data.ctrl[aid]))

                    # Keep J5 = -(J2+J3) for downward gripper.
                    j2_val = data.ctrl[arm_actuator_ids[1]]
                    j3_val = data.ctrl[arm_actuator_ids[2]]
                    data.ctrl[arm_actuator_ids[4]] = -(j2_val + j3_val)

                # --- CAMERA mode keys ---
                elif mode == Mode.CAMERA:
                    cam_name = _CAMERA_NAMES[active_cam_idx]
                    cid = cam_ids[cam_name]

                    # Tab switches active camera.
                    if key == _KEY_TAB:
                        active_cam_idx = (active_cam_idx + 1) % len(_CAMERA_NAMES)
                        cam_name = _CAMERA_NAMES[active_cam_idx]
                        cid = cam_ids[cam_name]
                        print(f"    active camera: {cam_name}")
                        continue

                    # Step size selection: keys '1'-'5' (ASCII 49-53).
                    if 49 <= key <= 53:
                        level = key - 48
                        step_size = _STEP_SIZES[level]
                        print(f"    step size: {step_size:.3f} m  (level {level})")
                        continue

                    # Position: arrows=X/Y, W/S=Z
                    if key == _KEY_RIGHT:
                        model.cam_pos[cid][0] += step_size
                    elif key == _KEY_LEFT:
                        model.cam_pos[cid][0] -= step_size
                    elif key == _KEY_UP:
                        model.cam_pos[cid][1] += step_size
                    elif key == _KEY_DOWN:
                        model.cam_pos[cid][1] -= step_size
                    elif key == 87:  # W
                        model.cam_pos[cid][2] += step_size
                    elif key == 83:  # S
                        model.cam_pos[cid][2] -= step_size
                    # FOV: Q/E (ASCII 81, 69)
                    elif key == 81:  # Q = decrease FOV
                        model.cam_fovy[cid] = max(1, model.cam_fovy[cid] - 1)
                    elif key == 69:  # E = increase FOV
                        model.cam_fovy[cid] = min(170, model.cam_fovy[cid] + 1)
                    else:
                        continue

                    # Deactivate orbit so it doesn't overwrite the keyboard edit.
                    preview_wins[cam_name].deactivate_orbit()
                    # Print updated MJCF.
                    print(f"    {_camera_to_mjcf(model, cid, cam_name)}")

            # Step physics to catch up with wall-clock time.
            target_time = sim_start + (time.monotonic() - wall_start)
            while data.time < target_time:
                mujoco.mj_step(model, data)
            viewer.sync()

            # Render camera previews into GLFW windows.
            for cam_name, pw in preview_wins.items():
                if pw.alive():
                    pw.render_and_display(renderer, data)
                    if pw.consume_dirty():
                        print(f"    {_camera_to_mjcf(model, cam_ids[cam_name], cam_name)}")
            glfw.poll_events()

    finally:
        for pw in preview_wins.values():
            pw.destroy()
        renderer.close()


if __name__ == "__main__":
    main()
