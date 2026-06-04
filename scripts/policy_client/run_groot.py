# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import functools
import os
import signal
import time
from moviepy.editor import ImageSequenceClip
import numpy as np
import msgpack
import msgpack_numpy as mnp
import zmq
from PIL import Image
import pandas as pd
from openpi_client import image_tools
from droid.robot_env import RobotEnv
import tqdm
import tyro
import yaml

faulthandler.enable()


DROID_CONTROL_FREQUENCY = 15

CONFIG_KEYS = [
    "task_name",
    "instructions",
    "folder_name",
    "max_timesteps",
    "num_rollouts",
]


class MsgSerializer:
    @staticmethod
    def to_bytes(data):
        default = functools.partial(MsgSerializer._safe_encode, chain=None)
        return msgpack.packb(data, default=default)

    @staticmethod
    def from_bytes(data):
        object_hook = functools.partial(MsgSerializer._safe_decode, chain=None)
        return msgpack.unpackb(data, object_hook=object_hook, raw=False)

    @staticmethod
    def _safe_encode(obj, chain=None):
        if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
            raise TypeError(f"Refusing to encode object-dtype ndarray")
        return mnp.encode(obj, chain=chain)

    @staticmethod
    def _safe_decode(obj, chain=None):
        if isinstance(obj, dict):
            nd_val = obj.get(b"nd", obj.get("nd"))
            kind_val = obj.get(b"kind", obj.get("kind"))
            if nd_val and kind_val in (b"O", "O"):
                raise ValueError("Refusing to decode object-dtype ndarray payload")
        return mnp.decode(obj, chain=chain)


class PolicyClient:
    def __init__(self, host="localhost", port=5555, timeout_ms=15000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def get_modality_config(self):
        request = {"endpoint": "get_modality_config"}
        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        response = MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def get_action(self, observation):
        request = {"endpoint": "get_action", "data": {"observation": observation, "options": None}}
        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        response = MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return tuple(response)

    def ping(self):
        request = {"endpoint": "ping"}
        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        return MsgSerializer.from_bytes(message)


@dataclasses.dataclass
class Args:
    left_camera_id: str = "39668372"
    right_camera_id: str = "33393346"
    wrist_camera_id: str = "16744838"
    external_camera: str = "left"
    max_timesteps: int = 600
    open_loop_horizon: int = 8
    remote_host: str = "localhost"
    remote_port: int = 5555
    results_dir: str = "/home/daphne/groot_results"


def prompt_yn(prompt: str) -> bool:
    while True:
        response = input(f"{prompt} (y/n): ").strip().lower()
        if response in ("y", "n"):
            return response == "y"
        print(f"  Invalid input '{response}' — please enter 'y' or 'n'.")


def prompt_int(prompt: str, min_val=None, max_val=None) -> int:
    while True:
        response = input(f"{prompt}: ").strip()
        try:
            value = int(response)
        except ValueError:
            print(f"  Invalid input '{response}' — please enter a whole number.")
            continue
        if min_val is not None and value < min_val:
            print(f"  Value must be at least {min_val}.")
            continue
        if max_val is not None and value > max_val:
            print(f"  Value must be at most {max_val}.")
            continue
        return value


def prompt_instructions() -> list:
    print("Enter instructions one per line. Press Enter on a blank line when done.")
    instructions = []
    i = 1
    while True:
        line = input(f"  Instruction {i}: ").strip()
        if not line:
            if not instructions:
                print("  Please enter at least one instruction.")
                continue
            break
        instructions.append(line)
        i += 1
    print(f"  Recorded {len(instructions)} instruction(s): {instructions}")
    return instructions


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def make_policy_client(host, port):
    return PolicyClient(host=host, port=port)


def rotvec_to_rot6d(rotvec):
    from scipy.spatial.transform import Rotation
    rot_matrix = Rotation.from_rotvec(rotvec).as_matrix()
    return rot_matrix[:, :2].T.flatten()


def get_modality_keys(modality_config):
    result = {}
    for modality in ["video", "state", "action", "language"]:
        cfg = modality_config[modality]
        if isinstance(cfg, dict):
            if "as_json" in cfg:
                result[modality] = cfg["as_json"]["modality_keys"]
            elif "modality_keys" in cfg:
                result[modality] = cfg["modality_keys"]
            else:
                raise ValueError(f"Cannot find modality_keys in config for '{modality}': {cfg}")
        else:
            result[modality] = cfg.modality_keys
    return result


def build_observation(curr_obs, modality_keys, external_camera, instruction, first_step=False):
    video_keys = modality_keys["video"]
    state_keys = modality_keys["state"]
    language_keys = modality_keys["language"]

    obs = {"video": {}, "state": {}, "language": {}}

    for key in video_keys:
        if "wrist" in key:
            img = curr_obs["wrist_image"]
        else:
            img = curr_obs[f"{external_camera}_image"]
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        if first_step:
            print(f"[DEBUG] video[{key}] shape: {img.shape} dtype: {img.dtype} min: {img.min()} max: {img.max()}")
        obs["video"][key] = img[None, None]

    for key in state_keys:
        if "gripper" in key:
            val = curr_obs["gripper_position"][None, None].astype(np.float32)
        elif "eef" in key:
            pos = curr_obs["cartesian_position"][:3]
            rot_input = curr_obs["cartesian_position"][3:]
            if first_step:
                print(f"[DEBUG] cartesian_position raw: {curr_obs['cartesian_position']}")
                print(f"[DEBUG] rotation component (last 3 values): {rot_input}")
            rot6d = rotvec_to_rot6d(rot_input)
            val = np.concatenate([pos, rot6d]).astype(np.float32)[None, None]
        else:
            val = curr_obs["joint_position"][None, None].astype(np.float32)
        if first_step:
            print(f"[DEBUG] state[{key}] shape: {val.shape} dtype: {val.dtype} values: {val.flatten()}")
        obs["state"][key] = val

    for key in language_keys:
        obs["language"][key] = [[instruction]]

    return obs


def run_rollout(env, policy_client, modality_keys, instruction, args):
    actions_from_chunk_completed = 0
    pred_action_chunk = None

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    video = []
    bar = tqdm.tqdm(range(args.max_timesteps))
    print("Running rollout... press Ctrl+C to stop early.")
    for t_step in bar:
        start_time = time.time()
        try:
            curr_obs = _extract_observation(
                args,
                env.get_observation(),
                save_to_disk=t_step == 0,
            )

            video.append(curr_obs[f"{args.external_camera}_image"])

            if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                actions_from_chunk_completed = 0

                with prevent_keyboard_interrupt():
                    obs = build_observation(
                        curr_obs, modality_keys, args.external_camera, instruction,
                        first_step=(t_step == 0),
                    )
                    action_chunk, _ = policy_client.get_action(obs)

                    pred_action_chunk = np.concatenate([
                        action_chunk["joint_position"][0],
                        action_chunk["gripper_position"][0],
                    ], axis=-1)

            action = pred_action_chunk[actions_from_chunk_completed]
            actions_from_chunk_completed += 1

            if action[-1].item() > 0.5:
                action = np.concatenate([action[:-1], np.ones((1,))])
            else:
                action = np.concatenate([action[:-1], np.zeros((1,))])

            env.step(action)

            elapsed_time = time.time() - start_time
            if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
        except KeyboardInterrupt:
            break

    return timestamp, t_step, video


def run_evaluation(env, policy_client, modality_keys, task_name, instructions, args, rollout_dir, benchmarking_mode=True):
    df = pd.DataFrame(columns=["task_name", "instruction", "run", "success", "completed_steps", "objective_recall", "duration", "video_filename"])
    save_time = datetime.datetime.now().strftime("%I-%M%p_%B_%d_%Y")
    yaml_filename = os.path.join(rollout_dir, f"eval_{save_time}.yaml")
    eval_target = prompt_int("Enter number of rollouts per instruction", min_val=1) if benchmarking_mode else None

    evals = 0
    try:
        while True:
            try:
                instruction = instructions[evals % len(instructions)]
                print(f"\nStarting rollout {evals + 1}" + (f" of {eval_target}" if eval_target is not None else "") + f" — instruction {(evals % len(instructions)) + 1}/{len(instructions)}: '{instruction}'")
                input("Press Enter to start rollout...")

                timestamp, t_step, video = run_rollout(
                    env, policy_client, modality_keys, instruction, args
                )

                if benchmarking_mode:
                    video = np.stack(video)
                    save_filename = os.path.join(rollout_dir, f"video_run{evals+1}_{timestamp}")
                    ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

                    success = 1.0 if prompt_yn("Did the rollout succeed?") else 0.0
                    completed_steps = prompt_int("How many steps were successfully completed?", min_val=0)
                    recall = prompt_int("How many of the objectives were completed?", min_val=0)

                    df = pd.concat([df, pd.DataFrame([{
                        "task_name": task_name,
                        "instruction": instruction,
                        "run": evals + 1,
                        "success": success,
                        "completed_steps": completed_steps,
                        "objective_recall": recall,
                        "duration": t_step,
                        "video_filename": save_filename + ".mp4",
                    }])], ignore_index=True)

                env.reset()
                evals += 1

                if eval_target is not None and evals >= eval_target:
                    print(f"Completed {evals} rollouts.")
                    break

            except KeyboardInterrupt:
                print("\nInterrupted — ending evaluation early.")
                env.reset()
                break

    finally:
        try:
            with open(yaml_filename, "w") as f:
                yaml.dump(
                    {"task_name": task_name, "instructions": instructions, "rollouts": df.to_dict(orient="records")},
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                )
            print(f"Results saved to {yaml_filename}")
        except Exception as e:
            print(f"WARNING: Failed to save YAML: {e}")

    return df


def main(args: Args):
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    print("Created the droid env!")

    policy_client = make_policy_client(args.remote_host, args.remote_port)
    print("Connecting to policy server...")

    ping = policy_client.ping()
    print(f"Server ping: {ping}")

    modality_config = policy_client.get_modality_config()
    modality_keys = get_modality_keys(modality_config)

    benchmarking_mode = prompt_yn("Are you benchmarking a task?")

    try:
        while True:
            try:
                task_name = input("Enter task name: ").strip()
                instructions = prompt_instructions()

                os.makedirs(args.results_dir, exist_ok=True)
                save_time = datetime.datetime.now().strftime("%I-%M%p_%B_%d_%Y")
                safe_task_name = task_name.replace(" ", "_").replace("/", "-")[:50]
                rollout_dir = os.path.join(args.results_dir, f"{safe_task_name}_{save_time}")
                os.makedirs(rollout_dir, exist_ok=True)

                run_evaluation(
                    env, policy_client, modality_keys, task_name, instructions, args, rollout_dir, benchmarking_mode
                )

                if not prompt_yn("Evaluate another set of instructions?"):
                    break

            except KeyboardInterrupt:
                print("\nInterrupted — returning to model selection.")
                try:
                    if prompt_yn("Swap to a different policy server?"):
                        policy_client = make_policy_client(args.remote_host, args.remote_port)
                        modality_config = policy_client.get_modality_config()
                        modality_keys = get_modality_keys(modality_config)
                        print("Connected to policy server!")
                    if not prompt_yn("Evaluate another task?"):
                        break
                except KeyboardInterrupt:
                    print("\nExiting.")
                    break

    except KeyboardInterrupt:
        print("\nExiting.")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    left_image = left_image[..., :3]
    wrist_image = wrist_image[..., :3]

    left_image = left_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)