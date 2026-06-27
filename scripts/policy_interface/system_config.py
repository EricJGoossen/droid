import dataclasses
from typing import Optional

@dataclasses.dataclass
class Args:
    # Camera parameters
    scene_camera_id: str = "39668372"
    wrist_camera_id: str = "16744838"
    recording_fps: int = 10

    # Rollout parameters
    policy: str = "pi0"  # choose from ["pi0", "pi05", "molmoact", "groot"]
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "localhost"
    remote_port: int = 8000

    # Output parameters
    results_dir: str = ""
    config_file: str = ""
    default_results_dir: str = "./results"

 