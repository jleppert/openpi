"""Interactive MuJoCo viewer with keyboard jog controls for the JAKA Zu5.

Press ENTER to toggle jog mode on/off.

Jog mode controls:
  LEFT / RIGHT  — J1 (base rotation)
  UP / DOWN     — J2 (shoulder)
  W / S         — J3 (elbow)
  A / D         — gripper open / close
  1-5           — set step size (1=tiny .. 5=large)
  R             — reset to home keyframe
  ENTER         — exit jog mode (return to MuJoCo defaults)
"""

import pathlib
import time

import mujoco
import mujoco.viewer

_ASSETS_DIR = pathlib.Path(__file__).parent / "assets"
_MJCF_PATH = _ASSETS_DIR / "jaka_zu5.xml"

_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
_ARM_ACTUATOR_NAMES = [f"act_joint{i}" for i in range(1, 7)]
_GRIPPER_ACTUATOR_NAME = "act_gripper"
_GRIPPER_MAX_OPEN = 0.04

# Step sizes selectable with keys 1-5.
_STEP_SIZES = {1: 0.005, 2: 0.01, 3: 0.02, 4: 0.05, 5: 0.1}

# GLFW key codes.
_KEY_ENTER = 257
_KEY_RIGHT = 262
_KEY_LEFT = 263
_KEY_DOWN = 264
_KEY_UP = 265


def main():
    model = mujoco.MjModel.from_xml_path(str(_MJCF_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    arm_actuator_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in _ARM_ACTUATOR_NAMES
    ]
    gripper_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, _GRIPPER_ACTUATOR_NAME)

    step_size = _STEP_SIZES[3]  # default medium
    gripper_pos = data.ctrl[gripper_actuator_id]
    jog_mode = False

    # Pending actions from key callback (thread-safe via list).
    pending = []

    def key_callback(key):
        pending.append(key)

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

    print("Press ENTER in the viewer window to toggle jog mode.\n")

    while viewer.is_running():
        # Process pending key events.
        while pending:
            key = pending.pop(0)

            # ENTER toggles jog mode.
            if key == _KEY_ENTER:
                jog_mode = not jog_mode
                if jog_mode:
                    print("  [JOG MODE ON]  Arrows=J1/J2  W/S=J3  A/D=gripper  1-5=step size  R=reset  ENTER=exit")
                else:
                    print("  [JOG MODE OFF] MuJoCo default controls active.  Press ENTER to jog.")
                continue

            # Ignore all other keys when not in jog mode.
            if not jog_mode:
                continue

            # Step size selection: keys '1'-'5' (ASCII 49-53).
            if 49 <= key <= 53:
                level = key - 48
                step_size = _STEP_SIZES[level]
                print(f"    step size: {step_size:.3f} rad  (level {level})")
                continue

            # R = reset (ASCII 82).
            if key == 82:
                mujoco.mj_resetDataKeyframe(model, data, 0)
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

        # Step physics and sync viewer.
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)


if __name__ == "__main__":
    main()
