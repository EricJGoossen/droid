import tqdm
import datetime
import time
import numpy as np
from droid.robot_env import RobotEnv

from scripts.policy_interface.policy_clients import PolicyClient
from scripts.policy_interface.eval_io import (
    save_rollout_video,
    write_eval_results,
    write_rollout_results,
    write_score_results,
    RolloutStatus,
)
from scripts.policy_interface.eval_planning import EpisodePlanEntry, EvaluationPlan
from scripts.policy_interface.eval_ui import update_status, start_rollout, get_score_input, get_test_instruction


# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


class EvalControl:
    def __init__(self, args: "Args", env: RobotEnv, policy_client: PolicyClient):
        self.args = args
        self.env = env
        self.policy_client = policy_client

    def run_rollout(
            self, 
            instruction: str, 
            run_number: int, 
            max_timesteps: int, 
            eval_results_dir: str = "", 
            max_step_score: int = None,
            max_recall_score: int = None,
            filename: str = "eval.yaml",
            save_data: bool = True,
            save_scene_video: bool = True,
            save_wrist_video: bool = True,
        ) -> None:
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        scene_video = []
        wrist_video = []

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        bar = tqdm.tqdm(range(max_timesteps))

        update_status("Running rollout...")
        t_step = 0
        
        interrupted = False
        try:
            for t_step in bar:
                start_time = time.time()

                curr_obs = self._extract_observation(
                    self.args,
                    self.env.get_observation(),
                )

                scene_video.append(curr_obs["scene_image"])
                wrist_video.append(curr_obs["wrist_image"])

                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= self.args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    pred_action_chunk = self.policy_client.infer(curr_obs, instruction)

                action = pred_action_chunk[actions_from_chunk_completed] if pred_action_chunk is not None else np.zeros((7,))
                actions_from_chunk_completed += 1

                if action[-1].item() > 0.5:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                self.env.step(action)

                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)

        except KeyboardInterrupt:
            interrupted = True

        finally:
            if not save_data:
                if interrupted:
                    raise KeyboardInterrupt
                return

            update_status("Rollout complete, saving results...")

            video_paths = []
            if save_scene_video:
                video_paths.append(save_rollout_video(eval_results_dir, f"scene_camera_rollout_{run_number}", scene_video))
            if save_wrist_video:
                video_paths.append(save_rollout_video(eval_results_dir, f"wrist_camera_rollout_{run_number}", wrist_video))

            rollout_results = {
                "run_number": run_number,
                "instruction": instruction,
                "duration": time.time() - start_time,
                "timesteps": t_step + 1,
                "data_files": video_paths,
                "timestamp": timestamp,
            }

            write_rollout_results(eval_results_dir, filename, run_number, rollout_results)

            score_results = get_score_input(max_step_score=max_step_score, max_recall_score=max_recall_score)
            write_score_results(eval_results_dir, filename, run_number, score_results)

            if interrupted:
                raise KeyboardInterrupt

    def run_eval_loop(self, episode: EpisodePlanEntry, filename: str = "eval.yaml", is_evaluation: bool = False) -> str:
        """Run one episode to completion: the initial not-yet-run pass, then
        an interactive retake loop until the user enters no rollout numbers.

        is_evaluation controls whether the retake-loop interrupt menu offers
        "advance" (skip remaining retakes, move to the next episode in an
        evaluation) -- not meaningful for a standalone episode, since there's
        nothing to advance to.

        Returns "advance" if the user chose to advance past this episode's
        retakes (only possible when is_evaluation=True), or "done" once the
        retake loop ends normally (user entered no numbers).
        """
        config_params = episode.episode_config
        eval_results_dir = episode.episode_dir

        expected_files = []
        if config_params.get("record_scene_video", False):
            expected_files.append("scene_camera")
        if config_params.get("record_wrist_video", False):
            expected_files.append("wrist_camera")

        eval_params = {
            "task_name": config_params["task_name"],
            "policy_name": self.args.policy,
            "policy_checkpoint": self.policy_client.get_policy_checkpoint(),
            "instructions": config_params["instructions"],
            "num_rollouts": config_params["num_rollouts"],
            "max_step_score": config_params.get("max_step_score", 0),
            "max_recall_score": config_params.get("max_recall_score", 0),
            "max_timesteps": config_params.get("max_timesteps", 0),
            "folder_path": eval_results_dir,
            "expected_files": expected_files,
        }
        write_eval_results(eval_results_dir, filename, eval_params)

        instructions = eval_params["instructions"]
        max_timesteps = eval_params["max_timesteps"]
        max_step_score = eval_params["max_step_score"]
        max_recall_score = eval_params["max_recall_score"]
        num_rollouts = eval_params["num_rollouts"]

        update_status("Starting episode for task: " + eval_params["task_name"])

        rollouts_to_run = []
        for run_number, status in enumerate(episode.rollout_statuses):
            if status == RolloutStatus.NOT_FOUND:
                rollouts_to_run.append(run_number)
            elif status == RolloutStatus.ROLLOUT_DATA:
                print(f"Rollout {run_number} is missing score data. Enter scores...")
                score_results = get_score_input(max_step_score=max_step_score, max_recall_score=max_recall_score)
                write_score_results(eval_results_dir, filename, run_number, score_results)
            elif status == RolloutStatus.SCORE_DATA:
                print(f"[WARNING] Rollout {run_number} is missing data files.")

        rollout_batch = rollouts_to_run

        while True:
            remaining_batch = list(rollout_batch)

            while remaining_batch:
                try:
                    self._run_rollout_batch(
                        remaining_batch, instructions, max_timesteps, eval_results_dir,
                        max_step_score, max_recall_score, filename, config_params,
                    )
                    remaining_batch = []
                except KeyboardInterrupt as e:
                    # run_rollout itself fully saves/scores the rollout it
                    # was interrupted on before re-raising, so that rollout
                    # counts as done too -- not just the ones before it.
                    completed_count = getattr(e, "completed_count", 0)
                    remaining_batch = remaining_batch[completed_count + 1:]

                    choice = self._handle_episode_interrupt(is_evaluation)
                    if choice == "advance":
                        return "advance"
                    if choice == "quit":
                        raise
                    # choice == "resume": loop continues with remaining_batch
                    # trimmed down to whatever this attempt hadn't reached yet.
                    self.env.reset()

            print("Enter rollout numbers to retake (whitespace-separated), or press Enter to finish:")
            retake_input = input("> ").strip()

            if not retake_input:
                return "done"

            try:
                retake_numbers = [int(token) for token in retake_input.split()]
            except ValueError:
                print(f"  Invalid input '{retake_input}' -- please enter whole numbers separated by whitespace.")
                continue

            out_of_range = [n for n in retake_numbers if n < 0 or n >= num_rollouts]
            if out_of_range:
                print(f"  Rollout number(s) {out_of_range} are out of range [0, {num_rollouts}).")
                continue

            rollout_batch = retake_numbers

    def _run_rollout_batch(
        self, rollout_numbers, instructions, max_timesteps, eval_results_dir,
        max_step_score, max_recall_score, filename, config_params,
    ) -> int:
        """Run a batch of rollout numbers (initial pass or a retake batch).

        Returns the count of rollout_numbers fully completed before a
        KeyboardInterrupt, if one occurs (not counting the one that was
        interrupted -- run_rollout itself saves/scores that one before
        re-raising, so the caller treats it as accounted for separately).
        On a clean run with no interrupt, the return value is unused.
        """
        completed = 0
        for rollout_num in rollout_numbers:
            instruction = instructions[rollout_num % len(instructions)]
            start_rollout(instruction, rollout_num)

            try:
                self.run_rollout(
                    instruction,
                    rollout_num,
                    max_timesteps,
                    eval_results_dir,
                    max_step_score=max_step_score,
                    max_recall_score=max_recall_score,
                    filename=filename,
                    save_data=True,
                    save_scene_video=config_params.get("record_scene_video", True),
                    save_wrist_video=config_params.get("record_wrist_video", True),
                )
            except KeyboardInterrupt as e:
                e.completed_count = completed
                raise

            completed += 1
            self.env.reset()

        return completed

    def _handle_episode_interrupt(self, is_evaluation: bool) -> str:
        """Prompt after an interrupt during the retake loop. Returns one of
        "resume", "advance", "test", or "quit". "advance" is only offered
        (and only returnable) when is_evaluation is True.
        """
        print("\nInterrupted.")

        if is_evaluation:
            prompt = "Resume, advance to the next episode, run a test rollout, or quit? (resume/advance/test/quit): "
            valid = {"resume", "advance", "test", "quit"}
        else:
            prompt = "Resume, run a test rollout, or quit? (resume/test/quit): "
            valid = {"resume", "test", "quit"}

        while True:
            choice = input(prompt).strip().lower()
            if choice in valid:
                break
            print(f"  Please enter one of: {', '.join(sorted(valid))}.")

        if choice == "test":
            try:
                self.run_test_loop()
            except KeyboardInterrupt:
                pass
            return self._handle_episode_interrupt(is_evaluation)

        return choice

    def run_evaluation_loop(self, plan: EvaluationPlan, filename: str = "eval.yaml") -> None:
        """Run every not-yet-complete episode in an evaluation plan, in list
        order. Episodes already complete (per the plan, computed up front)
        are left untouched -- this does not re-check or re-derive completion
        mid-run.
        """
        update_status(f"Starting evaluation: {plan.evaluation_name}")

        for episode in plan.episodes:
            if episode.is_complete:
                continue
            self.run_eval_loop(episode, filename=filename, is_evaluation=True)

    def run_test_loop(self) -> None:
        """A loop for testing rollout execution without saving any data or scores."""
        update_status("Running test rollout...")

        rollout_num = 1
        try:
            while True:
                instruction = get_test_instruction()
                start_rollout(instruction, rollout_num)
                self.run_rollout(instruction, run_number=0, max_timesteps=0, save_data=False)
                self.env.reset()

                rollout_num += 1
        except KeyboardInterrupt:
            pass

    def _extract_observation(self, args: "Args", obs_dict) -> dict:
        image_observations = obs_dict["image"]

        wrist_image, scene_image = None, None
        for key in image_observations:
            if args.scene_camera_id in key:
                scene_image = image_observations[key]
            elif args.wrist_camera_id in key:
                wrist_image = image_observations[key]

        if scene_image is None:
            raise ValueError(f"Scene camera image (id={args.scene_camera_id}) not found in observation keys: {list(image_observations.keys())}")
        if wrist_image is None:
            raise ValueError(f"Wrist camera image (id={args.wrist_camera_id}) not found in observation keys: {list(image_observations.keys())}")

        scene_image = scene_image[..., :3][..., ::-1]
        wrist_image = wrist_image[..., :3][..., ::-1]

        robot_state = obs_dict["robot_state"]
        cartesian_position = np.array(robot_state["cartesian_position"])
        joint_position = np.array(robot_state["joint_positions"])
        gripper_position = np.array([robot_state["gripper_position"]])

        return {
            "scene_image": scene_image,
            "wrist_image": wrist_image,
            "cartesian_position": cartesian_position,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }