"""Check the arm's reachable workspace at table height with the top-down constraint.

Sweeps J1/J2/J3, computes J5=-(J2+J3), and records gripper position.
Identifies which cube positions are physically reachable.
"""

import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

_ASSETS_DIR = pathlib.Path(__file__).parent / "assets"
_MJCF_PATH = _ASSETS_DIR / "jaka_zu5.xml"

_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
_TABLE_Z = 0.37
_CUBE_Z = 0.40  # cube center


def main():
    model = mujoco.MjModel.from_xml_path(str(_MJCF_PATH))
    data = mujoco.MjData(model)

    arm_joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in _ARM_JOINT_NAMES
    ]
    gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")

    # Get joint limits
    for i, jid in enumerate(arm_joint_ids):
        lo, hi = model.jnt_range[jid]
        print(f"  Joint{i+1} range: [{lo:.3f}, {hi:.3f}] rad = [{np.degrees(lo):.1f}, {np.degrees(hi):.1f}] deg")

    # Sweep J1, J2, J3 and record gripper position
    j1_range = np.linspace(-1.0, 1.0, 21)  # base rotation
    j2_range = np.linspace(-1.48, 1.5, 50)  # shoulder
    j3_range = np.linspace(-1.0, 3.0, 50)   # elbow

    # Collect all reachable positions at near-table-height
    reachable = []  # (x, y, z, j1, j2, j3)

    for j1 in j1_range:
        for j2 in j2_range:
            for j3 in j3_range:
                j5 = -(j2 + j3)

                # Set joint positions
                qpos_vals = [j1, j2, j3, 0.0, j5, 0.0]
                for i, (jid, val) in enumerate(zip(arm_joint_ids, qpos_vals)):
                    data.qpos[model.jnt_qposadr[jid]] = val

                mujoco.mj_forward(model, data)
                gx, gy, gz = data.xpos[gripper_body_id]

                # Check if gripper is near table height (within 10cm above table)
                if _TABLE_Z - 0.02 < gz < _TABLE_Z + 0.15:
                    reachable.append((gx, gy, gz, j1, j2, j3))

    reachable = np.array(reachable)
    print(f"\nTotal reachable configs near table height: {len(reachable)}")
    if len(reachable) == 0:
        print("No reachable positions found!")
        return

    print(f"  X range: [{reachable[:, 0].min():.3f}, {reachable[:, 0].max():.3f}]")
    print(f"  Y range: [{reachable[:, 1].min():.3f}, {reachable[:, 1].max():.3f}]")
    print(f"  Z range: [{reachable[:, 2].min():.3f}, {reachable[:, 2].max():.3f}]")

    # Check specific cube X positions
    cube_xs = [0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78]
    print(f"\n## Reachability at cube Y=0 for various X positions:")
    print(f"  {'Cube X':>8s}  {'Configs':>8s}  {'Min Z':>8s}  {'Max Z':>8s}  {'Can reach cube?':>16s}")

    for cx in cube_xs:
        # Find configs where gripper XY is within 3cm of target
        mask = (np.abs(reachable[:, 0] - cx) < 0.03) & (np.abs(reachable[:, 1]) < 0.03)
        nearby = reachable[mask]
        if len(nearby) > 0:
            min_z = nearby[:, 2].min()
            max_z = nearby[:, 2].max()
            can_reach = "YES" if min_z < _CUBE_Z + 0.03 else "NO (too high)"
            print(f"  {cx:8.2f}  {len(nearby):8d}  {min_z:8.3f}  {max_z:8.3f}  {can_reach:>16s}")
        else:
            print(f"  {cx:8.2f}  {0:8d}     ---      ---   NO (unreachable)")

    # Now check: for the "home" pose, where does the gripper end up?
    print(f"\n## Home pose check:")
    j2_home, j3_home = -1.48, 1.8
    home = [0.0, j2_home, j3_home, 0.0, -(j2_home + j3_home), 0.0]
    for i, (jid, val) in enumerate(zip(arm_joint_ids, home)):
        data.qpos[model.jnt_qposadr[jid]] = val
    mujoco.mj_forward(model, data)
    gx, gy, gz = data.xpos[gripper_body_id]
    print(f"  Gripper position at home: ({gx:.3f}, {gy:.3f}, {gz:.3f})")
    print(f"  Arm base position: {data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')]}")

    # Check gripper Z at specific (X, Y) targets by finding best J2/J3 combos
    print(f"\n## Detailed: gripper Z achievable at J1=0 (Y≈0) for each X:")
    j2_fine = np.linspace(-1.48, 2.0, 200)
    j3_fine = np.linspace(-1.0, 3.05, 200)

    for target_x in [0.62, 0.66, 0.70, 0.74, 0.78]:
        best_configs = []
        for j2 in j2_fine:
            for j3 in j3_fine:
                j5 = -(j2 + j3)
                qpos_vals = [0.0, j2, j3, 0.0, j5, 0.0]
                for i, (jid, val) in enumerate(zip(arm_joint_ids, qpos_vals)):
                    data.qpos[model.jnt_qposadr[jid]] = val
                mujoco.mj_forward(model, data)
                gx, gy, gz = data.xpos[gripper_body_id]
                if abs(gx - target_x) < 0.02 and abs(gy) < 0.02:
                    best_configs.append((gx, gy, gz, j2, j3))

        if best_configs:
            configs = np.array(best_configs)
            min_z_idx = configs[:, 2].argmin()
            max_z_idx = configs[:, 2].argmax()
            print(f"  X={target_x:.2f}: {len(configs)} configs, "
                  f"Z range=[{configs[:, 2].min():.3f}, {configs[:, 2].max():.3f}], "
                  f"best low: J2={configs[min_z_idx, 3]:.2f} J3={configs[min_z_idx, 4]:.2f} → Z={configs[min_z_idx, 2]:.3f}")
        else:
            print(f"  X={target_x:.2f}: NOT REACHABLE at J1=0")


if __name__ == "__main__":
    main()
