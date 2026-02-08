"""JAKA Zu5 MuJoCo simulation client for OpenPI pi0_fast_droid inference."""

import dataclasses
import logging
import pathlib

import env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import saver as _saver
import tyro


@dataclasses.dataclass
class Args:
    out_dir: pathlib.Path = pathlib.Path("data/jaka_zu5_sim/videos")

    prompt: str = "pick up the red cube"

    # Action horizon: 8 steps at 15 Hz ≈ 0.5s, matching DROID default.
    action_horizon: int = 8

    host: str = "0.0.0.0"
    port: int = 8000

    # Show live MuJoCo viewer window (requires a display).
    display: bool = False


def main(args: Args) -> None:
    runtime = _runtime.Runtime(
        environment=_env.JakaZu5SimEnvironment(prompt=args.prompt, display=args.display),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[
            _saver.VideoSaver(args.out_dir),
        ],
        max_hz=15,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
