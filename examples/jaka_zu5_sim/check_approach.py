"""Quick check: simulate a few steps from home to see the gripper trajectory."""

import pathlib
import sys

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gym_env


def check_approach(model_path, target_x, target_y=0.0):
    """Run one episode with cube at a specific position and trace the gripper."""
    model = PPO.load(model_path)
    env = gym_env.JakaZu5PickCubeEnv()

    # Reset and manually place cube
    obs, _ = env.reset(seed=42)
    # Override cube position
    cube_qpos_adr = env._model.jnt_qposadr[env._cube_joint_id]
    env._data.qpos[cube_qpos_adr] = target_x
    env._data.qpos[cube_qpos_adr + 1] = target_y
    env._data.qpos[cube_qpos_adr + 2] = 0.40
    import mujoco
    mujoco.mj_forward(env._model, env._data)
    obs = env._get_obs()

    print(f"\n  Cube at ({target_x:.2f}, {target_y:.2f}, 0.40)")
    gripper_pos = env._data.xpos[env._gripper_body_id]
    cube_pos = env._data.xpos[env._cube_body_id]
    print(f"  Initial gripper: ({gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f})")
    print(f"  Initial cube:    ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")
    print(f"  Initial dist:    {np.linalg.norm(gripper_pos - cube_pos):.3f}")
    print(f"  Gripper X offset from cube: {gripper_pos[0] - target_x:+.3f}")

    print(f"\n  {'Step':>4s}  {'Grip X':>7s}  {'Grip Y':>7s}  {'Grip Z':>7s}  {'Cube X':>7s}  {'Cube Z':>7s}  {'Dist':>6s}  {'Grasp':>5s}  {'Act[0:3]':>20s}  {'GripCmd':>7s}")
    for step in range(40):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        gp = env._data.xpos[env._gripper_body_id]
        cp = env._data.xpos[env._cube_body_id]
        dist = info["dist"]
        grasp = info["has_grasp"]

        if step < 25 or step % 5 == 0 or terminated or truncated:
            print(f"  {step:4d}  {gp[0]:7.3f}  {gp[1]:7.3f}  {gp[2]:7.3f}  {cp[0]:7.3f}  {cp[2]:7.3f}  {dist:6.3f}  {'YES' if grasp else 'no':>5s}  [{action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f}]  {action[3]:+.2f}")

        if terminated:
            print(f"  → TERMINATED: {'SUCCESS' if info.get('success') else 'DROP'}")
            break
        if truncated:
            print(f"  → TRUNCATED")
            break

    env.close()


if __name__ == "__main__":
    model_path = "data/jaka_zu5_sim/rl/best_success_model/best_success_model"
    print("=" * 100)
    print("APPROACH TRAJECTORY ANALYSIS")
    print("=" * 100)

    for x in [0.62, 0.66, 0.70, 0.74, 0.78]:
        check_approach(model_path, x, 0.0)
    # Also check with Y offset
    print("\n--- With Y offset ---")
    for x in [0.62, 0.66, 0.70, 0.74]:
        check_approach(model_path, x, 0.05)
