from __future__ import annotations

import argparse
import datetime
import inspect
import json
import logging
import os
import signal
import sys
import time
from multiprocessing import Manager, Process
from multiprocessing import current_process
from typing import List

sys.path.insert(0, '/tmp')
sys.path.insert(0, '/share/home/group3/agent/OSWorld')

import lib_run_single
from desktop_env.desktop_env import DesktopEnv
from lib_results_logger import log_task_completion
from agent.OSWorld.uitars15_v2_bami import UITarsBamiAgent

active_environments = []
processes = []
is_terminating = False

if os.path.exists('.env'):
    from dotenv import load_dotenv
    load_dotenv()


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run end-to-end evaluation on the benchmark with UI-TARS + BAMI')
    parser.add_argument('--path_to_vm', type=str, default=None)
    parser.add_argument('--headless', action='store_true', help='Run in headless machine')
    parser.add_argument('--action_space', type=str, default='pyautogui', help='Action type')
    parser.add_argument('--observation_type', choices=['screenshot', 'a11y_tree', 'screenshot_a11y_tree', 'som'], default='screenshot', help='Observation type')
    parser.add_argument('--sleep_after_execution', type=float, default=3.0)
    parser.add_argument('--max_steps', type=int, default=15)
    parser.add_argument('--test_config_base_dir', type=str, default='evaluation_examples')
    parser.add_argument('--model', type=str, default='doubao-1-5-thinking-vision-pro-250428')
    parser.add_argument('--model_type', type=str, default='doubao', choices=['doubao', 'qwen25', 'qwen25vl'])
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--max_tokens', type=int, default=3000)
    parser.add_argument('--use_thinking', action='store_true', default=False)
    parser.add_argument('--max_trajectory_length', type=int, default=None)
    parser.add_argument('--max_image_history_length', type=int, default=5)
    parser.add_argument('--language', type=str, default='Chinese')
    parser.add_argument('--enable_bami', action='store_true', help='Enable BAMI refinement for UI-TARS actions')
    parser.add_argument('--bami_local_judge_model_path', type=str, default=None)
    parser.add_argument('--bami_local_judge_base_model_path', type=str, default=None)
    parser.add_argument('--bami_local_judge_gpu', type=str, default=None)
    parser.add_argument('--bami_mask_ratio', type=float, default=0.12)
    parser.add_argument('--bami_crop_expand_ratio', type=float, default=0.2)
    parser.add_argument('--domain', type=str, default='all')
    parser.add_argument('--test_all_meta_path', type=str, default='evaluation_examples/test_all.json')
    parser.add_argument('--result_dir', type=str, default='./results_uitars_bami')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of environments to run in parallel')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='Set the logging level')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region for the VM')
    parser.add_argument('--provider_name', type=str, default='aws', choices=['aws', 'virtualbox', 'vmware', 'docker', 'azure'], help='Provider name')
    parser.add_argument('--client_password', type=str, default='', help='Client password')
    parser.add_argument('--screen_width', type=int, default=1920, help='Screen width')
    parser.add_argument('--screen_height', type=int, default=1080, help='Screen height')
    return parser.parse_args()

args = config()
args.result_dir = os.path.abspath(args.result_dir)
args.result_dir = os.path.abspath(args.result_dir)
logger = logging.getLogger()
log_level = getattr(logging, args.log_level.upper())
logger.setLevel(log_level)
datetime_str = datetime.datetime.now().strftime('%Y%m%d@%H%M%S')
file_handler = logging.FileHandler(os.path.join('logs', f'normal-bami-{datetime_str}.log'), encoding='utf-8')
debug_handler = logging.FileHandler(os.path.join('logs', f'debug-bami-{datetime_str}.log'), encoding='utf-8')
stdout_handler = logging.StreamHandler(sys.stdout)
file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(log_level)
formatter = logging.Formatter(fmt='\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s')
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
stdout_handler.addFilter(logging.Filter('desktopenv'))
logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger = logging.getLogger('desktopenv.experiment')

def distribute_tasks(test_all_meta: dict) -> List[tuple]:
    all_tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((domain, example_id))
    return all_tasks

def _append_jsonl(file_path: str, payload: dict):
    with open(file_path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write('\n')

def run_single_example_with_bami_trace(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = lib_run_single.setup_logger(example, example_result_dir)
    env.reset(task_config=example)
    reset_params = inspect.signature(agent.reset).parameters
    reset_kwargs = {}
    if 'vm_ip' in reset_params:
        reset_kwargs['vm_ip'] = env.vm_ip
    if 'runtime_logger' in reset_params:
        reset_kwargs['runtime_logger'] = runtime_logger
    elif '_logger' in reset_params:
        reset_kwargs['_logger'] = runtime_logger
    elif 'logger' in reset_params:
        reset_kwargs['logger'] = runtime_logger
    agent.reset(**reset_kwargs) if reset_kwargs else agent.reset()
    time.sleep(60)
    obs = env._get_obs()
    done = False
    step_idx = 0
    env.controller.start_recording()
    bami_trace_path = os.path.join(example_result_dir, 'bami_traj.jsonl')
    while not done and step_idx < max_steps:
        response, actions = agent.predict(instruction, obs)
        for action in actions:
            action_timestamp = datetime.datetime.now().strftime('%Y%m%d@%H%M%S%f')
            logger.info('Step %d: %s', step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)
            logger.info('Reward: %.2f', reward)
            logger.info('Done: %s', done)
            screenshot_name = f'step_{step_idx + 1}_{action_timestamp}.png'
            screenshot_bytes = obs.get('screenshot') if isinstance(obs, dict) else None
            if screenshot_bytes is not None:
                os.makedirs(example_result_dir, exist_ok=True)
                with open(os.path.join(example_result_dir, screenshot_name), 'wb') as handle:
                    handle.write(screenshot_bytes)
            else:
                logger.warning('Step screenshot missing, recording step without screenshot file.')
                screenshot_name = None
            traj_payload = {'step_num': step_idx + 1, 'action_timestamp': action_timestamp, 'action': action, 'response': response, 'reward': reward, 'done': done, 'info': info, 'screenshot_file': screenshot_name}
            _append_jsonl(os.path.join(example_result_dir, 'traj.jsonl'), traj_payload)
            if isinstance(response, dict) and response.get('bami'):
                bami_payload = {'step_num': step_idx + 1, 'action_timestamp': action_timestamp, 'action': action, 'bami': response.get('bami'), 'raw_prediction': response.get('raw_prediction'), 'reground_prediction': response.get('reground_prediction'), 'screenshot_file': screenshot_name}
                _append_jsonl(bami_trace_path, bami_payload)
            if done:
                logger.info('The episode is done.')
                break
        step_idx += 1
    time.sleep(20)
    result = env.evaluate()
    logger.info('Result: %.2f', result)
    scores.append(result)
    with open(os.path.join(example_result_dir, 'result.txt'), 'w', encoding='utf-8') as handle:
        handle.write(f'{result}\n')
    log_task_completion(example, result, example_result_dir, args)
    env.controller.end_recording(os.path.join(example_result_dir, 'recording.mp4'))

def run_env_tasks(task_queue, args: argparse.Namespace, shared_scores: list):
    env = None
    try:
        screen_size = (args.screen_width, args.screen_height)
        env_kwargs = dict(path_to_vm=args.path_to_vm, action_space=args.action_space, provider_name=args.provider_name, screen_size=screen_size, headless=args.headless, os_type='Ubuntu', require_a11y_tree=args.observation_type in ['a11y_tree', 'screenshot_a11y_tree', 'som'], enable_proxy=True, client_password=args.client_password)
        env = DesktopEnv(**env_kwargs)
        agent = UITarsBamiAgent(model=args.model, model_type=args.model_type, max_tokens=args.max_tokens, top_p=args.top_p, temperature=args.temperature, max_trajectory_length=args.max_trajectory_length, max_image_history_length=args.max_image_history_length, use_thinking=args.use_thinking, language=args.language, enable_bami=args.enable_bami, bami_local_judge_model_path=args.bami_local_judge_model_path, bami_local_judge_base_model_path=args.bami_local_judge_base_model_path, bami_local_judge_gpu=args.bami_local_judge_gpu, bami_mask_ratio=args.bami_mask_ratio, bami_crop_expand_ratio=args.bami_crop_expand_ratio)
        logger.info('Process %s started.', current_process().name)
        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break
            domain, example_id = item
            try:
                config_file = os.path.join(args.test_config_base_dir, f'examples/{domain}/{example_id}.json')
                with open(config_file, 'r', encoding='utf-8') as handle:
                    example = json.load(handle)
                example_result_dir = os.path.join(args.result_dir, args.action_space, args.observation_type, args.model, domain, example_id)
                os.makedirs(example_result_dir, exist_ok=True)
                logger.info('[%s][Domain]: %s', current_process().name, domain)
                logger.info('[%s][Example ID]: %s', current_process().name, example_id)
                logger.info('[%s][Instruction]: %s', current_process().name, example['instruction'])
                run_single_example_with_bami_trace(agent, env, example, args.max_steps, example['instruction'], args, example_result_dir, shared_scores)
            except Exception as exc:
                import traceback
                logger.error('Exception in %s %s/%s: %s', current_process().name, domain, example_id, exc)
                logger.error(traceback.format_exc())

                err_text = str(exc)
                infra_error_markers = [
                    'NoneType',
                    'No such image',
                    'images/create',
                    'failed to resolve reference',
                    'lookup mirror.baidubce.com',
                    'Failed to establish a new connection',
                    'Connection refused',
                ]
                if any(marker in err_text for marker in infra_error_markers):
                    logger.critical('Infrastructure error detected, stopping run instead of consuming remaining queue: %s', err_text)
                    raise

                os.makedirs(example_result_dir, exist_ok=True)
                os.makedirs(example_result_dir, exist_ok=True)
                with open(os.path.join(example_result_dir, 'traj.jsonl'), 'a', encoding='utf-8') as handle:
                    handle.write(json.dumps({'Error': f'{domain}/{example_id} - {exc}'}, ensure_ascii=False))
                    handle.write('\n')
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                logger.error('Error closing environment: %s', exc)

def signal_handler(signum, frame):
    logger.info('Received signal %s. Shutting down...', signum)
    sys.exit(0)

def main():
    with open(args.test_all_meta_path, 'r', encoding='utf-8') as handle:
        test_all_meta = json.load(handle)
    if args.domain != 'all':
        test_all_meta = {args.domain: test_all_meta[args.domain]}
    all_tasks = distribute_tasks(test_all_meta)
    logger.info('Args: %s', args)
    logger.info('Total tasks: %d', len(all_tasks))
    manager = Manager()
    task_queue = manager.Queue()
    shared_scores = manager.list()
    for task in all_tasks:
        task_queue.put(task)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    process = Process(target=run_env_tasks, args=(task_queue, args, shared_scores), name='EnvProcess-1')
    process.start()
    processes.append(process)
    logger.info('Started process %s with PID %s', process.name, process.pid)
    process.join()
    logger.info('All processes completed. Finished tasks: %d', len(shared_scores))

if __name__ == '__main__':
    main()
