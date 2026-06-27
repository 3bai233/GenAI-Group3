"""Run AgentS3 on OSWorld with Docker provider."""

import argparse
import datetime
import json
import logging
import os
import sys
import signal
import time
from multiprocessing import Process, Manager, current_process, Queue

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from desktop_env.desktop_env import DesktopEnv
from lib_results_logger import log_task_completion

if os.path.exists(os.path.join(os.path.dirname(__file__), "../../.env")):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

os.makedirs(os.path.join(os.path.dirname(__file__), "../../logs"), exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join(os.path.dirname(__file__), f"../../logs/agents3-{datetime_str}.log"),
    encoding="utf-8",
)
stdout_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.INFO)
stdout_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
stdout_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

logger = logging.getLogger("desktopenv.experiment")

active_environments = []
processes = []
is_terminating = False


def distribute_tasks(test_all_meta: dict) -> list:
    all_tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((domain, example_id))
    return all_tasks


def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    try:
        agent.reset()
    except Exception as e:
        logger.warning(f"agent.reset() failed: {e}")

    env.reset(task_config=example)
    time.sleep(60)
    obs = env._get_obs()

    with open(os.path.join(example_result_dir, "step_0.png"), "wb") as f:
        f.write(obs["screenshot"])
    with open(os.path.join(example_result_dir, "instruction.txt"), "w", encoding="utf-8") as f:
        f.write(instruction)

    done = False
    step_idx = 0
    while not done and step_idx < max_steps:
        response, actions = agent.predict(instruction, obs)
        for action in actions:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)
            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            with open(
                os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"), "wb"
            ) as f:
                f.write(obs["screenshot"])
            response.update({
                "step_num": step_idx + 1,
                "action_timestamp": action_timestamp,
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
                "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png",
            })
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(response, ensure_ascii=False))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    log_task_completion(
        task_id=example.get("id", os.path.basename(example_result_dir)),
        domain=example.get("snapshot", os.path.basename(os.path.dirname(example_result_dir))),
        score=result,
        result_dir=args.result_dir,
    )


def run_env_tasks(task_queue: Queue, args: argparse.Namespace, shared_scores: list, engine_params, engine_params_for_grounding):
    env = None
    try:
        from gui_agents.s3.agents.agent_s import AgentS3
        from gui_agents.s3.agents.grounding import OSWorldACI

        env = DesktopEnv(
            path_to_vm=args.path_to_vm,
            action_space=args.action_space,
            provider_name=args.provider_name,
            screen_size=(args.screen_width, args.screen_height),
            headless=args.headless,
            os_type="Ubuntu",
            require_a11y_tree=False,
            enable_proxy=True,
        )
        grounding_agent = OSWorldACI(
            env=env,
            platform="linux",
            engine_params_for_generation=engine_params,
            engine_params_for_grounding=engine_params_for_grounding,
            width=args.screen_width,
            height=args.screen_height,
        )
        agent = AgentS3(
            engine_params,
            grounding_agent,
            platform="linux",
            max_trajectory_length=args.max_trajectory_length,
        )

        logger.info(f"Process {current_process().name} started.")
        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break
            domain, example_id = item
            try:
                config_file = os.path.join(args.test_config_base_dir, f"examples/{domain}/{example_id}.json")
                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)
                instruction = example["instruction"]
                example_result_dir = os.path.join(
                    args.result_dir,
                    args.action_space,
                    args.observation_type,
                    args.model,
                    domain,
                    example_id,
                )
                os.makedirs(example_result_dir, exist_ok=True)
                logger.info(f"[{current_process().name}][Domain]: {domain}")
                logger.info(f"[{current_process().name}][Example ID]: {example_id}")
                logger.info(f"[{current_process().name}][Instruction]: {instruction}")
                try:
                    run_single_example(agent, env, example, args.max_steps, instruction, args, example_result_dir, shared_scores)
                except Exception as e:
                    import traceback
                    logger.error(f"Exception in {current_process().name} {domain}/{example_id}: {e}")
                    logger.error(traceback.format_exc())
                    with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                        f.write(json.dumps({"Error": f"{domain}/{example_id} - {e}"}))
                        f.write("\n")
            except Exception as e:
                logger.error(f"Task-level error in {current_process().name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Process-level error in {current_process().name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"{current_process().name} cleaning up...")
        if env:
            try:
                env.close()
            except Exception as e:
                logger.error(f"Error closing env: {e}")


def signal_handler(signum, frame):
    global is_terminating, processes
    if is_terminating:
        return
    is_terminating = True
    logger.info(f"Received signal {signum}. Shutting down...")
    for p in processes:
        if p.is_alive():
            p.terminate()
    time.sleep(1)
    sys.exit(0)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgentS3 on OSWorld")

    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--provider_name", type=str, default="docker")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--action_space", type=str, default="pyautogui")
    parser.add_argument("--observation_type", type=str, default="screenshot")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_trajectory_length", type=int, default=8)

    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--test_all_meta_path", type=str, default="evaluation_examples/test_nogdrive.json")
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples")
    parser.add_argument("--result_dir", type=str, default="./results_agents3")

    parser.add_argument("--model_provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--model_url", type=str, default="")
    parser.add_argument("--model_api_key", type=str, default="")
    parser.add_argument("--model_temperature", type=float, default=None)

    parser.add_argument("--ground_provider", type=str, default="openai")
    parser.add_argument("--ground_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--ground_api_key", type=str, default="EMPTY")
    parser.add_argument("--ground_model", type=str, default="/share/home/group3/agent/OSWorld/UI-TARS-1.5-7B")
    parser.add_argument("--grounding_width", type=int, default=1920)
    parser.add_argument("--grounding_height", type=int, default=1080)

    return parser.parse_args()


def get_unfinished(action_space, use_model, observation_type, result_dir, total_file_json):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path) and "result.txt" in os.listdir(example_path):
                    finished[domain].append(example_id)

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [x for x in total_file_json[domain] if x not in examples]

    return total_file_json


def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    global processes
    logger.info("Args: %s", args)
    all_tasks = distribute_tasks(test_all_meta)
    logger.info(f"Total tasks: {len(all_tasks)}")

    engine_params = {
        "engine_type": args.model_provider,
        "model": args.model,
        "base_url": args.model_url,
        "api_key": args.model_api_key,
        "temperature": args.model_temperature,
    }
    engine_params_for_grounding = {
        "engine_type": args.ground_provider,
        "model": args.ground_model,
        "base_url": args.ground_url,
        "api_key": args.ground_api_key,
        "grounding_width": args.grounding_width,
        "grounding_height": args.grounding_height,
    }

    with Manager() as manager:
        shared_scores = manager.list()
        task_queue = manager.Queue()
        for item in all_tasks:
            task_queue.put(item)

        num_envs = min(args.num_envs, len(all_tasks))
        processes = []
        for i in range(num_envs):
            p = Process(
                target=run_env_tasks,
                args=(task_queue, args, shared_scores, engine_params, engine_params_for_grounding),
                name=f"AgentS3-{i+1}",
            )
            p.daemon = True
            p.start()
            processes.append(p)
            logger.info(f"Started process {p.name} with PID {p.pid}")

        try:
            while True:
                alive = sum(1 for p in processes if p.is_alive())
                if task_queue.empty():
                    logger.info("All tasks finished.")
                    break
                if alive == 0:
                    logger.error("All processes died.")
                    break
                time.sleep(5)
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt. Shutting down...")
            raise

        scores = list(shared_scores)
    logger.info(f"Average score: {sum(scores) / len(scores) if scores else 0:.4f} ({len(scores)} tasks)")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = config()

    path_to_args = os.path.join(args.result_dir, args.action_space, args.observation_type, args.model, "args.json")
    os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
    with open(path_to_args, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    test_file_list = get_unfinished(args.action_space, args.model, args.observation_type, args.result_dir, test_all_meta)

    remaining = sum(len(v) for v in test_file_list.values())
    logger.info(f"Remaining tasks: {remaining}")

    test(args, test_file_list)
