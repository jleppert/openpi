"""Collect demonstrations from a trained RL policy and save as a LeRobot dataset.

Records DROID-format observations at 15 Hz throughout the entire episode,
including during the scripted grasp-and-lift sequence. Camera parameters
are customizable via JSON config for matching real camera setups.

Usage:
    # Collect 500 successful episodes (default)
    MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/collect_demos.py

    # Quick test (3 episodes)
    MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/collect_demos.py \
        --args.n-episodes 3

    # With custom camera config
    MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/collect_demos.py \
        --args.camera-config cameras.json

    # Export default camera config for editing
    MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/collect_demos.py \
        --args.save-camera-config cameras.json --args.n-episodes 0

    # Custom model path
    MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/collect_demos.py \
        --args.model-path data/jaka_zu5_sim/rl_test4/final_model.zip
"""

import dataclasses
import logging
import pathlib
import shutil
import sys

import mujoco
import mujoco.renderer
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from openpi_client import image_tools
from tqdm import tqdm
import tyro

# Ensure local imports work regardless of working directory.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import camera_config as cam_cfg

_ASSETS_DIR = pathlib.Path(__file__).parent / "assets"
_MJCF_PATH = _ASSETS_DIR / "jaka_zu5.xml"

# Arm/gripper names (must match MJCF).
_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
_ARM_ACTUATOR_NAMES = [f"act_joint{i}" for i in range(1, 7)]
_GRIPPER_JOINT = "finger_left"
_GRIPPER_ACTUATOR = "act_gripper"
_GRIPPER_MAX_OPEN = 0.04

# Physics/control.
_PHYSICS_STEPS = 33  # per control step (~15 Hz at 0.002s timestep)

# Workspace and task parameters.
_TABLE_Z = 0.37
_CUBE_X_RANGE = (0.55, 0.82)
_CUBE_Y_RANGE = (-0.35, 0.35)
_WS_X_RANGE = (0.45, 0.85)
_WS_Y_RANGE = (-0.40, 0.40)
_APPROACH_Z = 0.52
_GRASP_Z = 0.475
_LIFT_Z = 0.62
_LIFT_THRESHOLD = 0.50
_REACH_THRESHOLD = 0.02
_MAX_APPROACH_STEPS = 100

# IK solver.
_IK_MAX_ITER = 20
_IK_DAMPING = 0.01
_IK_TOL = 1e-3
_CONTROLLED_JOINTS = [0, 1, 2]  # J1, J2, J3

# Grasp phase timing (in 15 Hz control steps).
_DESCEND_STEPS = 9     # ~0.6s to descend to grasp height
_CLOSE_STEPS = 15      # ~1.0s to close gripper
_LIFT_INCREMENTS = 15  # number of Z increments during lift
_STEPS_PER_LIFT = 5    # control steps per lift increment


@dataclasses.dataclass
class Args:
    """Arguments for demonstration collection."""

    # Path to trained SAC model.
    model_path: str = "data/jaka_zu5_sim/rl_test4/best_success_model/best_success_model.zip"

    # Number of episodes to collect (0 = skip collection, useful with --save-camera-config).
    n_episodes: int = 500

    # Only save successful episodes.
    only_successful: bool = True

    # Language instruction for the task.
    prompt: str = "pick up the red cube"

    # LeRobot dataset repo ID.
    repo_id: str = "levelhq/jaka_zu5_pick_cube"

    # Camera config JSON file (overrides defaults if provided).
    camera_config: str | None = None

    # Save resolved camera config to JSON (for editing and reuse).
    save_camera_config: str | None = None

    # RL algorithm: "sac" or "ppo".
    algo: str = "sac"

    # Push dataset to HuggingFace Hub.
    push_to_hub: bool = False


class DemoCollector:
    """Collects demonstrations by replaying RL policy with full 15 Hz recording.

    Records DROID-format observations at every control step (33 physics steps),
    including during the scripted grasp-and-lift sequence that the RL policy
    triggers when the gripper is close enough to the cube.
    """

    def __init__(self, ext_cam: cam_cfg.CameraConfig, wrist_cam: cam_cfg.CameraConfig):
        self._model = mujoco.MjModel.from_xml_path(str(_MJCF_PATH))
        self._data = mujoco.MjData(self._model)

        # Apply camera configs to the MuJoCo model.
        self._ext_cfg = ext_cam
        self._wrist_cfg = wrist_cam
        self._ext_id = cam_cfg.apply_camera_config(self._model, ext_cam)
        self._wrist_id = cam_cfg.apply_camera_config(self._model, wrist_cam)

        # Per-camera renderers (may have different resolutions).
        self._ext_rend = mujoco.Renderer(
            self._model, height=ext_cam.render_height, width=ext_cam.render_width
        )
        self._wrist_rend = mujoco.Renderer(
            self._model, height=wrist_cam.render_height, width=wrist_cam.render_width
        )

        # Cache body/joint/actuator/geom IDs.
        self._jids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in _ARM_JOINT_NAMES
        ]
        self._aids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in _ARM_ACTUATOR_NAMES
        ]
        self._grip_jid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, _GRIPPER_JOINT
        )
        self._grip_aid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, _GRIPPER_ACTUATOR
        )
        self._cube_bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "red_cube"
        )
        self._cube_jid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint"
        )
        self._grip_bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base"
        )
        self._lfg = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_geom"
        )
        self._rfg = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_geom"
        )
        self._cg = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, "red_cube_geom"
        )

        self._rng = np.random.default_rng()
        self._dt = self._model.opt.timestep * _PHYSICS_STEPS  # ~0.066s

    def collect_episode(self, rl_model, prompt: str) -> tuple[list[dict], bool]:
        """Run one full episode. Returns (frames, success).

        frames: list of DROID-format dicts (one per 15 Hz control step).
        success: True if cube was lifted above threshold.
        """
        self._reset()
        frames: list[dict] = []

        # Phase 1: Approach — RL policy drives XY positioning.
        reached = False
        for _ in range(_MAX_APPROACH_STEPS):
            obs = self._rl_obs()
            action_xy, _ = rl_model.predict(obs, deterministic=True)
            action_xy = np.clip(
                action_xy,
                [_WS_X_RANGE[0], _WS_Y_RANGE[0]],
                [_WS_X_RANGE[1], _WS_Y_RANGE[1]],
            )
            target = np.array([action_xy[0], action_xy[1], _APPROACH_Z])
            jt = self._solve_ik(target)
            cur = self._read_joints()
            frames.append(self._record(self._droid_action(cur, jt, True), prompt))
            self._step(jt, gripper_open=True)

            if self._grip_cube_dist() <= _REACH_THRESHOLD:
                reached = True
                break

        if not reached:
            return frames, False

        cube_xy = self._data.xpos[self._cube_bid][:2].copy()

        # Phase 2: Descend to grasp height.
        desc_jt = self._solve_ik(np.array([cube_xy[0], cube_xy[1], _GRASP_Z]))
        for _ in range(_DESCEND_STEPS):
            cur = self._read_joints()
            frames.append(self._record(self._droid_action(cur, desc_jt, True), prompt))
            self._step(desc_jt, gripper_open=True)

        # Phase 3: Close gripper (keep descend IK targets, matching gym_env).
        hold_jt = desc_jt
        for _ in range(_CLOSE_STEPS):
            cur = self._read_joints()
            frames.append(self._record(self._droid_action(cur, hold_jt, False), prompt))
            self._step(hold_jt, gripper_open=False)

        if not self._check_grasp():
            return frames, False

        # Phase 4: Lift incrementally.
        for k in range(1, _LIFT_INCREMENTS + 1):
            z = _GRASP_Z + (_LIFT_Z - _GRASP_Z) * k / _LIFT_INCREMENTS
            lift_jt = self._solve_ik(np.array([cube_xy[0], cube_xy[1], z]))
            for _ in range(_STEPS_PER_LIFT):
                cur = self._read_joints()
                frames.append(
                    self._record(self._droid_action(cur, lift_jt, False), prompt)
                )
                self._step(lift_jt, gripper_open=False)

        success = bool(self._data.xpos[self._cube_bid][2] > _LIFT_THRESHOLD)
        return frames, success

    # ---- Simulation helpers ------------------------------------------------

    def _reset(self):
        """Reset simulation with randomized cube and arm start positions."""
        mujoco.mj_resetData(self._model, self._data)

        # Randomize cube on table.
        cx = self._rng.uniform(*_CUBE_X_RANGE)
        cy = self._rng.uniform(*_CUBE_Y_RANGE)
        adr = self._model.jnt_qposadr[self._cube_jid]
        self._data.qpos[adr : adr + 3] = [cx, cy, _TABLE_Z + 0.03]
        self._data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]

        # Nominal home pose for IK seed.
        j2, j3 = -1.48, 1.8
        home = [0.0, j2, j3, 0.0, -(j2 + j3), 0.0]
        for i, (jid, val) in enumerate(zip(self._jids, home)):
            self._data.qpos[self._model.jnt_qposadr[jid]] = val
            self._data.ctrl[self._aids[i]] = val
        mujoco.mj_forward(self._model, self._data)

        # IK to random start position.
        sx = self._rng.uniform(*_WS_X_RANGE)
        sy = self._rng.uniform(*_WS_Y_RANGE)
        start_jt = self._solve_ik(np.array([sx, sy, _APPROACH_Z]))
        for i, (jid, val) in enumerate(zip(self._jids, start_jt)):
            self._data.qpos[self._model.jnt_qposadr[jid]] = val
            self._data.ctrl[self._aids[i]] = val

        # Open gripper.
        self._data.qpos[self._model.jnt_qposadr[self._grip_jid]] = _GRIPPER_MAX_OPEN
        self._data.ctrl[self._grip_aid] = _GRIPPER_MAX_OPEN
        mujoco.mj_forward(self._model, self._data)

    def _step(self, joint_targets: np.ndarray, gripper_open: bool):
        """Set actuators and step physics one control cycle (33 steps)."""
        for i, aid in enumerate(self._aids):
            self._data.ctrl[aid] = joint_targets[i]
        self._data.ctrl[self._grip_aid] = (
            _GRIPPER_MAX_OPEN if gripper_open else 0.0
        )
        for _ in range(_PHYSICS_STEPS):
            mujoco.mj_step(self._model, self._data)

    def _rl_obs(self) -> np.ndarray:
        """4D RL observation: [gripper_x, gripper_y, cube_x, cube_y]."""
        gxy = self._data.xpos[self._grip_bid][:2].astype(np.float32)
        cxy = self._data.xpos[self._cube_bid][:2].astype(np.float32)
        return np.concatenate([gxy, cxy])

    def _read_joints(self) -> np.ndarray:
        """Read current 6 arm joint positions."""
        return np.array(
            [self._data.qpos[self._model.jnt_qposadr[j]] for j in self._jids],
            dtype=np.float64,
        )

    def _grip_cube_dist(self) -> float:
        """XY distance between gripper and cube."""
        return float(
            np.linalg.norm(
                self._data.xpos[self._grip_bid][:2]
                - self._data.xpos[self._cube_bid][:2]
            )
        )

    def _check_grasp(self) -> bool:
        """Check if both fingers are in contact with the cube."""
        left = right = False
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if {g1, g2} == {self._lfg, self._cg}:
                left = True
            if {g1, g2} == {self._rfg, self._cg}:
                right = True
        return left and right

    # ---- Action computation ------------------------------------------------

    def _droid_action(
        self, current: np.ndarray, target: np.ndarray, gripper_open: bool
    ) -> np.ndarray:
        """Compute 8D DROID action from joint position delta.

        The velocity is computed so that env.py's apply_action reproduces
        the same actuator targets: target = current + vel * dt.
        """
        vel = (target - current) / self._dt
        a = np.zeros(8, dtype=np.float32)
        a[:6] = vel.astype(np.float32)
        # a[6] = 0.0  (padded 7th DOF for Franka compatibility)
        a[7] = 1.0 if gripper_open else 0.0
        return a

    # ---- Observation recording ---------------------------------------------

    def _record(self, action: np.ndarray, prompt: str) -> dict:
        """Capture current observation and pair with the given action."""
        # External camera.
        self._ext_rend.update_scene(self._data, camera=self._ext_id)
        ext = self._ext_rend.render().copy()
        ext = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                ext, self._ext_cfg.output_size, self._ext_cfg.output_size
            )
        )

        # Wrist camera.
        self._wrist_rend.update_scene(self._data, camera=self._wrist_id)
        wrist = self._wrist_rend.render().copy()
        wrist = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                wrist, self._wrist_cfg.output_size, self._wrist_cfg.output_size
            )
        )

        # Joint positions (6D padded to 7D for DROID/Franka format).
        j6 = self._read_joints().astype(np.float32)
        j7 = np.append(j6, 0.0).astype(np.float32)

        # Gripper position normalized to [0, 1].
        g = self._data.qpos[self._model.jnt_qposadr[self._grip_jid]]
        gp = np.array([g / _GRIPPER_MAX_OPEN], dtype=np.float32)

        return {
            "exterior_image_1_left": ext,
            "wrist_image_left": wrist,
            "joint_position": j7,
            "gripper_position": gp,
            "actions": action,
            "task": prompt,
        }

    # ---- IK solver ---------------------------------------------------------

    def _solve_ik(self, target_pos: np.ndarray) -> np.ndarray:
        """Jacobian-based IK for J1-J3, with J4=0, J5=-(J2+J3), J6=0.

        Solves for the gripper_base body position. Returns 6 joint targets.
        State is saved/restored so the simulation is not affected.
        """
        qpos_save = self._data.qpos.copy()
        qvel_save = self._data.qvel.copy()

        nv = self._model.nv
        jacp = np.zeros((3, nv))

        for _ in range(_IK_MAX_ITER):
            mujoco.mj_forward(self._model, self._data)
            err = target_pos - self._data.xpos[self._grip_bid]
            if np.linalg.norm(err) < _IK_TOL:
                break

            jacp[:] = 0
            mujoco.mj_jacBody(
                self._model, self._data, jacp, None, self._grip_bid
            )
            dof_idx = [
                self._model.jnt_dofadr[self._jids[j]] for j in _CONTROLLED_JOINTS
            ]
            J = jacp[:, dof_idx]

            JJT = J @ J.T + (_IK_DAMPING**2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, err)

            for idx, cj in enumerate(_CONTROLLED_JOINTS):
                jid = self._jids[cj]
                adr = self._model.jnt_qposadr[jid]
                lo, hi = self._model.jnt_range[jid]
                self._data.qpos[adr] = np.clip(
                    self._data.qpos[adr] + dq[idx], lo, hi
                )

            # Wrist constraints: J4=0, J5=-(J2+J3), J6=0.
            j2 = self._data.qpos[self._model.jnt_qposadr[self._jids[1]]]
            j3 = self._data.qpos[self._model.jnt_qposadr[self._jids[2]]]
            j5_id = self._jids[4]
            j5_lo, j5_hi = self._model.jnt_range[j5_id]
            self._data.qpos[self._model.jnt_qposadr[self._jids[3]]] = 0.0
            self._data.qpos[self._model.jnt_qposadr[j5_id]] = np.clip(
                -(j2 + j3), j5_lo, j5_hi
            )
            self._data.qpos[self._model.jnt_qposadr[self._jids[5]]] = 0.0

        result = np.array(
            [self._data.qpos[self._model.jnt_qposadr[j]] for j in self._jids]
        )
        self._data.qpos[:] = qpos_save
        self._data.qvel[:] = qvel_save
        mujoco.mj_forward(self._model, self._data)
        return result

    def close(self):
        del self._ext_rend
        del self._wrist_rend


# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    # Build camera configs.
    if args.camera_config:
        cfgs = cam_cfg.load_configs(args.camera_config)
        ext_cam = cfgs.get("external", cam_cfg.default_external())
        wrist_cam = cfgs.get("wrist", cam_cfg.default_wrist())
    else:
        ext_cam = cam_cfg.default_external()
        wrist_cam = cam_cfg.default_wrist()

    if args.save_camera_config:
        cam_cfg.save_configs(
            {"external": ext_cam, "wrist": wrist_cam}, args.save_camera_config
        )
        logging.info("Camera config saved to %s", args.save_camera_config)

    logging.info(
        "External camera: pos=%s fov=%.1f render=%dx%d output=%d",
        ext_cam.pos, ext_cam.fov, ext_cam.render_width, ext_cam.render_height,
        ext_cam.output_size,
    )
    logging.info(
        "Wrist camera: pos=%s fov=%.1f render=%dx%d output=%d",
        wrist_cam.pos, wrist_cam.fov, wrist_cam.render_width, wrist_cam.render_height,
        wrist_cam.output_size,
    )

    if args.n_episodes == 0:
        logging.info("n_episodes=0, skipping collection.")
        return

    # Load RL model.
    if args.algo == "sac":
        from stable_baselines3 import SAC
        logging.info("Loading SAC model from %s", args.model_path)
        model = SAC.load(args.model_path, device="cpu")
    elif args.algo == "ppo":
        from stable_baselines3 import PPO
        logging.info("Loading PPO model from %s", args.model_path)
        model = PPO.load(args.model_path, device="cpu")
    else:
        raise ValueError(f"Unknown algo '{args.algo}', expected 'sac' or 'ppo'")

    collector = DemoCollector(ext_cam, wrist_cam)

    # Create LeRobot dataset (wipes any existing dataset with same repo_id).
    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="jaka_zu5",
        fps=15,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (ext_cam.output_size, ext_cam.output_size, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (wrist_cam.output_size, wrist_cam.output_size, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )

    saved = 0
    attempted = 0
    pbar = tqdm(total=args.n_episodes, desc="Collecting demos")

    while saved < args.n_episodes:
        attempted += 1
        frames, success = collector.collect_episode(model, args.prompt)

        if success or not args.only_successful:
            for f in frames:
                dataset.add_frame(f)
            dataset.save_episode()
            saved += 1
            pbar.update(1)

        pbar.set_postfix(
            attempted=attempted,
            success_rate=f"{saved}/{attempted}",
            ep_len=len(frames),
        )

    pbar.close()
    logging.info(
        "Collected %d/%d episodes (%.1f%% success rate)",
        saved, attempted, 100 * saved / attempted,
    )
    logging.info("Dataset saved to %s", output_path)

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["jaka_zu5", "pick_cube", "simulation"],
            private=False,
            push_videos=True,
        )

    collector.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
