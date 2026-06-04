# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import pandas as pd
from PIL import Image
import torch
from droid.robot_env import RobotEnv
import tqdm
import tyro
import yaml
import json_numpy
import requests
json_numpy.patch()

faulthandler.enable()


# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15

CONFIG_KEYS = [
    "task_name",
    "instructions",
    "folder_name",
    "max_timesteps",
    "num_rollouts",
]


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "39668372"
    right_camera_id: str = "33393346"
    wrist_camera_id: str = "16744838"


    # Policy parameters
    external_camera: str = "left"  # choose from ["left", "right"]


    # Rollout parameters
    max_timesteps: int = 600
    open_loop_horizon: int = 8


    # Remote server parameters
    remote_host: str = "localhost"
    remote_port: int = 8000


    # Output parameters
    results_dir: str = "/media/daphne/8563-0B16/MolmoAct/results"


    # Repo id
    repo_id: str = "allenai/MolmoAct2-DROID"



def prompt_yn(prompt: str) -> bool:
    """Prompt the user for a yes/no answer, retrying until valid input is given."""
    while True:
        response = input(f"{prompt} (y/n): ").strip().lower()
        if response in ("y", "n"):
            return response == "y"
        print(f"  Invalid input '{response}' — please enter 'y' or 'n'.")



def prompt_int(prompt: str, min_val=None, max_val=None) -> int:
    """Prompt the user for an integer, retrying until valid input is given."""
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
    """Prompt the user to enter a list of instructions one by one."""
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
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
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


def load_config(path: str) -> dict:
    """Load evaluation configuration from a YAML file."""

    with open(path, "r") as f: 
        config = yaml.safe_load(f)

    for key in CONFIG_KEYS:
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")
    if not isinstance(config["instructions"], list) or len(config["instructions"]) == 0:
        raise ValueError("Config key 'instructions' must be a non-empty list.")

    return config


def make_policy_client(host, port):
    return f"http://{host}:{port}"


def run_rollout(env, policy_client, instruction, args):
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
                    payload = json_numpy.dumps({
                        "external_cam": curr_obs[f"{args.external_camera}_image"],
                        "wrist_cam": curr_obs["wrist_image"],
                        "instruction": instruction,
                        "state": np.concatenate([curr_obs["joint_position"], curr_obs["gripper_position"]]),
                    })
                    response = requests.post(f"{policy_client}/act", data=payload, headers={"Content-Type": "application/json"})
                    result = json_numpy.loads(response.text)
                    pred_action_chunk = result["actions"]

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


def run_evaluation(env, policy_client, task_name, instructions, args, rollout_dir, benchmarking_mode=True):
    """One evaluation loop: N total rollouts, cycling through instructions round-robin."""
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
 
                timestamp, t_step, video = run_rollout(env, policy_client, instruction, args)
 
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
    print("Connected to policy server!")

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

                run_evaluation(env, policy_client, task_name, instructions, args, rollout_dir, benchmarking_mode)

                if not prompt_yn("Evaluate another set of instructions?"):
                    break

            except KeyboardInterrupt:
                print("\nInterrupted — returning to model selection.")
                try:  # Ctrl+C here exits
                    if prompt_yn("Swap to a different policy server?"):
                        policy_client = make_policy_client(args.remote_host, args.remote_port)
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



