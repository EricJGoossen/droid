import datetime
import faulthandler
import os
import tyro
from droid.robot_env import RobotEnv

from scripts.policy_interface.system_config import Args
from scripts.policy_interface.policy_clients import POLICY_CLIENTS
from scripts.policy_interface.eval_io import load_config
from scripts.policy_interface.eval_planning import build_plan
from scripts.policy_interface.eval_control import EvalControl


faulthandler.enable()


def _resolve_results_dir(args: Args, config: dict) -> str:
    """Return the results directory to use: args.results_dir if given,
    otherwise a timestamped fallback under args.default_results_dir.
    """
    if args.results_dir:
        return args.results_dir

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if config["config_type"] == "evaluation":
        name = config["evaluation_name"]
    else:
        name = f"{args.policy}_{config['task_name']}"

    return os.path.join(args.default_results_dir, f"{name}_{timestamp}")


def _run_plan(controller: EvalControl, plan):
    if plan.config_type == "episode":
        controller.run_eval_loop(plan.episodes[0])
    else:
        controller.run_evaluation_loop(plan)


def _check_cameras_exist(env: RobotEnv, args: Args) -> None:
    """Fail fast if either configured camera ID isn't present in the robot's
    observation, rather than letting this surface deep inside the first
    rollout's _extract_observation call.
    """
    image_keys = list(env.get_observation()["image"].keys())

    scene_found = any(args.scene_camera_id in key for key in image_keys)
    wrist_found = any(args.wrist_camera_id in key for key in image_keys)

    if not scene_found or not wrist_found:
        missing = []
        if not scene_found:
            missing.append(f"scene_camera_id={args.scene_camera_id}")
        if not wrist_found:
            missing.append(f"wrist_camera_id={args.wrist_camera_id}")
        raise RuntimeError(
            f"Camera(s) not found in observation: {', '.join(missing)}. "
            f"Available image keys: {image_keys}"
        )


def main(args: Args):
    policy_client = POLICY_CLIENTS[args.policy](args.remote_host, args.remote_port)

    env = RobotEnv(action_space=policy_client.action_space(), gripper_action_space=policy_client.gripper_space())
    print("Created the droid env!")

    _check_cameras_exist(env, args)

    policy_client.connect()
    print("Connected to policy server!")

    controller = EvalControl(args, env, policy_client)

    if not args.config_file:
        controller.run_test_loop()
        return

    config = load_config(args.config_file)
    results_dir = _resolve_results_dir(args, config)

    plan = build_plan(args.config_file, args.policy, results_dir)

    try:
        _run_plan(controller, plan)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)