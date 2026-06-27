import os
from enum import Enum
import numpy as np
import yaml
from moviepy.editor import ImageSequenceClip


EPISODE_CONFIG_KEYS = [
    "config_type",
    "task_name",
    "instructions",
    "max_timesteps",
    "num_rollouts",
    "max_step_score",
    "max_recall_score",
    "record_scene_video",
    "record_wrist_video",
]

EVALUATION_CONFIG_KEYS = [
    "config_type",
    "evaluation_name",
    "episode_paths",
]

EVAL_RESULT_KEYS = [
    "task_name",
    "policy_name",
    "policy_checkpoint",
    "instructions",
    "num_rollouts",
    "max_step_score",
    "max_recall_score",
    "max_timesteps",
    "folder_path",
    "expected_files",
]

ROLLOUT_RESULT_KEYS = [
    "instruction",
    "duration",
    "timesteps",
    "run_number",
    "timestamp",
    "data_files",
]

SCORE_RESULT_KEYS = [
    "success",
    "step_score",
    "recall_score",
    "comments",
]


class RolloutStatus(Enum):
    """Status of a rollout entry in the eval results file, based on which fields are filled in."""

    NOT_FOUND = 0     # rollout doesn't exist in file yet
    ROLLOUT_DATA = 1  # rollout data is written but not yet scored
    SCORE_DATA = 2    # rollout data and score data are written, but not yet all data files
    VIDEO_DATA = 3    # rollout data, score data, and all expected data files are present


def get_rollout_statuses(folder_path: str, filename: str) -> "list[RolloutStatus] | None":
    """Inspect an eval results file and report status for every rollout slot.

    Returns None if the eval file doesn't exist yet, is empty, or has eval
    fields missing (i.e. write_eval_results hasn't successfully run yet).

    Otherwise returns a list of RolloutStatus with length num_rollouts.

    Raises ValueError if any rollout key falls outside the valid range
    [0, num_rollouts).
    """
    path = os.path.join(folder_path, filename)

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not data:
        return None

    if any(key not in data for key in EVAL_RESULT_KEYS):
        return None

    num_rollouts = data["num_rollouts"]
    expected_files = data["expected_files"]
    folder_path = os.path.dirname(path) or "."

    rollouts = data.get("rollouts", {})

    out_of_range = [n for n in rollouts if n < 0 or n >= num_rollouts]
    if out_of_range:
        raise ValueError(
            f"Eval results file at '{path}' has rollout entries with "
            f"rollout_number outside the valid range [0, {num_rollouts}): {out_of_range}."
        )

    rollout_statuses = []
    for run_number in range(num_rollouts):
        rollout = rollouts.get(run_number)

        if rollout is None:
            rollout_statuses.append(RolloutStatus.NOT_FOUND)
            continue

        is_scored = all(key in rollout for key in SCORE_RESULT_KEYS)
        if not is_scored:
            rollout_statuses.append(RolloutStatus.ROLLOUT_DATA)
            continue

        has_all_files = _check_data_files(rollout["data_files"], expected_files, folder_path)
        if has_all_files:
            rollout_statuses.append(RolloutStatus.VIDEO_DATA)
        else:
            rollout_statuses.append(RolloutStatus.SCORE_DATA)

    return rollout_statuses


def is_episode_complete(rollout_statuses: "list[RolloutStatus] | None") -> bool:
    """An episode is complete iff its rollout file exists and every rollout
    slot has reached RolloutStatus.VIDEO_DATA.
    """
    if rollout_statuses is None:
        return False
    return len(rollout_statuses) > 0 and all(status == RolloutStatus.VIDEO_DATA for status in rollout_statuses)


def load_config(path: str) -> dict:
    """Load a config file and validate it against the schema matching its
    own config_type field (episode or evaluation).
    """
    config = _load_yaml_file(path, check_rollouts=False)

    if "config_type" not in config:
        raise ValueError(f"Config file at '{path}' is missing required key 'config_type'.")

    if config["config_type"] == "episode":
        for key in EPISODE_CONFIG_KEYS:
            if key not in config:
                raise ValueError(f"Missing required episode config key: '{key}'")
        if not isinstance(config["instructions"], list) or len(config["instructions"]) == 0:
            raise ValueError("Config key 'instructions' must be a non-empty list.")
    elif config["config_type"] == "evaluation":
        for key in EVALUATION_CONFIG_KEYS:
            if key not in config:
                raise ValueError(f"Missing required evaluation config key: '{key}'")
        if not isinstance(config["episode_paths"], list) or len(config["episode_paths"]) == 0:
            raise ValueError("Config key 'episode_paths' must be a non-empty list.")
    else:
        raise ValueError(
            f"Config file at '{path}' has unrecognized config_type "
            f"'{config['config_type']}' (expected 'episode' or 'evaluation')."
        )

    return config


def write_eval_results(folder_path: str, filename: str, results: dict) -> None:
    """Create the eval results file with config fields and an empty rollouts dict.

    Call this once at the start of an episode, before any rollouts are written.
    If the file already exists with at least one rollout already written
    (e.g. resuming after a crash), this does nothing rather than wiping out
    existing progress.
    """

    for key in EVAL_RESULT_KEYS:
        if key not in results:
            raise ValueError(f"Missing required result key: '{key}'")

    path = os.path.join(folder_path, filename)

    if os.path.exists(path):
        existing = _load_yaml_file(path)
        if existing["rollouts"]:
            return

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    data = dict(results)
    data["rollouts"] = {}

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def write_rollout_results(folder_path: str, filename: str, rollout_number: int, results: dict) -> None:
    """Write a rollout entry to the eval results file at the given rollout_number.

    If an entry already exists at rollout_number, it is fully overwritten --
    this is an upsert, not an append. Use write_score_results afterward to
    fill in score fields on this same entry.
    """

    for key in ROLLOUT_RESULT_KEYS:
        if key not in results:
            raise ValueError(f"Missing required rollout result key: '{key}'")

    path = os.path.join(folder_path, filename)

    if not isinstance(results["data_files"], list):
        raise ValueError("'data_files' must be a list of filenames.")

    data = _load_yaml_file(path)
    data["rollouts"][rollout_number] = dict(results)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def write_score_results(folder_path: str, filename: str, rollout_number: int, results: dict) -> None:
    """Update the rollout entry at the given rollout_number with score fields."""

    for key in SCORE_RESULT_KEYS:
        if key not in results:
            raise ValueError(f"Missing required score result key: '{key}'")

    path = os.path.join(folder_path, filename)

    data = _load_yaml_file(path)

    if rollout_number not in data["rollouts"]:
        raise ValueError(
            f"No rollout entry found for rollout_number={rollout_number} in '{path}'. "
            "Call write_rollout_results() for this rollout_number before write_score_results()."
        )

    data["rollouts"][rollout_number].update(results)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def save_rollout_video(folder_path: str, filename: str, video_frames: list, fps: int = 10) -> str:
    """Save a sequence of frames as an MP4.

    Returns the filename of the saved video (without folder path), which
    can be stored in the eval results file and looked up later for scoring.
    """

    os.makedirs(folder_path, exist_ok=True)

    video = np.stack(video_frames)
    save_path = os.path.join(folder_path, f"{filename}.mp4")
    ImageSequenceClip(list(video), fps=fps).write_videofile(save_path, codec="libx264")

    return os.path.basename(save_path)

def _load_yaml_file(path: str, check_rollouts: bool = True) -> dict:
    """Load an existing YAML file, raising if it's missing or malformed."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at '{path}'.")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"File at '{path}' exists but is empty or unreadable.")

    if check_rollouts and ("rollouts" not in data or not isinstance(data["rollouts"], dict)):
        raise ValueError(f"Eval results file at '{path}' is missing a valid 'rollouts' dict.")

    return data


def _check_data_files(data_files: list, expected_files: list, folder_path: str) -> bool:
    """Check that every expected_files entry matches exactly one data_files path,
    and that every matched file actually exists on disk.
    """

    for expected in expected_files:
        matches = [f for f in data_files if expected in f]
        if len(matches) != 1:
            return False

        full_path = os.path.join(folder_path, matches[0])
        if not os.path.exists(full_path):
            return False

    return True
