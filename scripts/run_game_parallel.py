import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from concurrent.futures import ProcessPoolExecutor, as_completed

NUM_BATTLES = 30
ROUNDS_PER_BATTLE = 10
INITIAL_STACK = 3000
SMALL_BLIND_AMOUNT = 5

VERBOSE_LEVEL = 1
EXPERIMENTS_DIR = "experiments"
LOG_LEVEL = "INFO"

from pypokerengine.api.game import setup_config, start_poker
from players.gpt_player import GPTPlayer
from players.llama_player import LlamaPlayer
from players.deepSeek_player import DeepSeekPlayer
from players.qwen_player import QwenPlayer
from players.grok_player import GrokPlayer
from players.gemini_player import GeminiPlayer
from players.loose_aggressive_monte_player import LooseAggressiveMontePlayer
from players.tight_passive_monte_player import TightPassiveMontePlayer
from players.tight_aggressive_monte_player import TightAggressiveMontePlayer
from players.loose_passive_monte_player import LoosePassiveMontePlayer
from players.maniac_monte_player import ManiacMontePlayer


import json
import os
import logging
import sys
from datetime import datetime, timedelta
import uuid
import shutil
import io
import time
from contextlib import redirect_stdout

def get_player_aliases():
    return {
        "gpt": GPTPlayer.PLAYER_ALIAS,
        "llama": LlamaPlayer.PLAYER_ALIAS,
        "deepseek": DeepSeekPlayer.PLAYER_ALIAS,
        "qwen": QwenPlayer.PLAYER_ALIAS,
        "grok": GrokPlayer.PLAYER_ALIAS,
        "gemini": GeminiPlayer.PLAYER_ALIAS,
        "lag": LooseAggressiveMontePlayer.PLAYER_ALIAS,
        "tp": TightPassiveMontePlayer.PLAYER_ALIAS,
        "ta":TightAggressiveMontePlayer.PLAYER_ALIAS,
        "lp":LoosePassiveMontePlayer.PLAYER_ALIAS,
        "Maniac":ManiacMontePlayer.PLAYER_ALIAS,
    }

class TeeOutput:
    """Writes to both the console and a file handle."""
    def __init__(self, console, file_handle):
        self.console = console
        self.file = file_handle

    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()

class ExperimentManager:
    """Manages poker experiments with logging and incremental saving."""

    def __init__(self, experiment_name=None):
        self.experiment_id = experiment_name or f"poker_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        self.exp_dir = os.path.join(EXPERIMENTS_DIR, self.experiment_id)
        self.results_file = os.path.join(self.exp_dir, "results.json")
        self.config_file = os.path.join(self.exp_dir, "config.json")
        self.log_file = os.path.join(self.exp_dir, "experiment.log")
        self.resume_file = os.path.join(self.exp_dir, "resume_state.json")

        os.makedirs(self.exp_dir, exist_ok=True)

        self.setup_logging()

        self.results = {"battles": [], "experiment_info": {}}
        self.resume_state = {"completed_battles": 0, "status": "running"}

        self.experiment_start_time = time.time()
        self.battle_times = []
        self.current_battle_start_time = None

        self.save_config()

        self.game_log_file = os.path.join(self.exp_dir, "game_output.log")
        self.original_stdout = sys.stdout

        self.logger.info(f"Experiment initialized: {self.experiment_id}")

    def setup_logging(self):
        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(getattr(logging, LOG_LEVEL))

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def save_config(self):
        config = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "settings": {
                "num_battles": NUM_BATTLES,
                "rounds_per_battle": ROUNDS_PER_BATTLE,
                "initial_stack": INITIAL_STACK,
                "small_blind_amount": SMALL_BLIND_AMOUNT,
                "verbose_level": VERBOSE_LEVEL
            },
            "players": get_player_aliases()
        }

        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def load_resume_state(self):
        if os.path.exists(self.resume_file):
            with open(self.resume_file, 'r') as f:
                self.resume_state = json.load(f)
            self.logger.info(f"Resuming from battle {self.resume_state['completed_battles'] + 1}")
            return True
        return False

    def save_resume_state(self):
        with open(self.resume_file, 'w') as f:
            json.dump(self.resume_state, f, indent=2)

    def load_existing_results(self):
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)

            # Reset start time on resume so remaining-time estimates reflect
            # only the current session.
            self.experiment_start_time = time.time()

    def save_battle_result(self, battle_num, battle_result):
        """Save a single battle result, merging with any existing results file."""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)

        while len(self.results["battles"]) <= battle_num:
            self.results["battles"].append(None)

        self.results["battles"][battle_num] = battle_result

        self.results["experiment_info"] = {
            "experiment_id": self.experiment_id,
            "total_battles": NUM_BATTLES,
            "completed_battles": battle_num + 1,
            "last_updated": datetime.now().isoformat(),
            "status": "completed" if battle_num + 1 >= NUM_BATTLES else "running"
        }

        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        self.resume_state["completed_battles"] = battle_num + 1
        self.resume_state["status"] = self.results["experiment_info"]["status"]
        self.save_resume_state()

        self.logger.info(f"Battle {battle_num + 1} results saved")

    def start_game_logging(self):
        self.game_log_handle = open(self.game_log_file, 'a', encoding='utf-8')
        self.tee_output = TeeOutput(self.original_stdout, self.game_log_handle)
        sys.stdout = self.tee_output

    def stop_game_logging(self):
        try:
            if hasattr(self, 'tee_output'):
                sys.stdout = self.original_stdout
            if hasattr(self, 'game_log_handle') and self.game_log_handle:
                self.game_log_handle.close()
                self.game_log_handle = None
        except Exception as e:
            # Restore stdout even if closing the log file raises.
            sys.stdout = self.original_stdout
            self.logger.warning(f"Error stopping game logging: {e}")

    def start_battle_timer(self):
        self.current_battle_start_time = time.time()

    def end_battle_timer(self):
        if self.current_battle_start_time is not None:
            battle_duration = time.time() - self.current_battle_start_time
            self.battle_times.append(battle_duration)
            self.current_battle_start_time = None
            return battle_duration
        return 0

    def format_time(self, seconds):
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def get_time_estimates(self, current_battle_num):
        if not self.battle_times:
            return None, None, None

        avg_battle_time = sum(self.battle_times) / len(self.battle_times)
        remaining_battles = NUM_BATTLES - (current_battle_num + 1)
        estimated_remaining_time = remaining_battles * avg_battle_time
        total_elapsed = time.time() - self.experiment_start_time
        estimated_total_time = total_elapsed + estimated_remaining_time

        return avg_battle_time, estimated_remaining_time, estimated_total_time

    def log_progress_stats(self, battle_num, battle_duration):
        completed_battles = battle_num + 1
        progress_percent = (completed_battles / NUM_BATTLES) * 100

        self.logger.info(f"Progress: {completed_battles}/{NUM_BATTLES} battles ({progress_percent:.1f}%)")
        self.logger.info(f"Battle {completed_battles} duration: {self.format_time(battle_duration)}")

        # Need at least 2 battles before the average is meaningful.
        if len(self.battle_times) >= 2:
            avg_time, remaining_time, total_time = self.get_time_estimates(battle_num)

            self.logger.info(f"Average battle time: {self.format_time(avg_time)}")
            self.logger.info(f"Estimated remaining time: {self.format_time(remaining_time)}")

            if remaining_time is not None:
                eta = datetime.now() + timedelta(seconds=remaining_time)
                self.logger.info(f"Estimated completion: {eta.strftime('%H:%M:%S')}")

        total_elapsed = time.time() - self.experiment_start_time
        self.logger.info(f"Total elapsed time: {self.format_time(total_elapsed)}")

    def get_starting_battle(self):
        if self.load_resume_state():
            self.load_existing_results()
            return self.resume_state["completed_battles"]
        return 0

def run_poker_game(rounds=ROUNDS_PER_BATTLE, initial_stack=INITIAL_STACK, small_blind=SMALL_BLIND_AMOUNT):
    """Run a single poker game and return detailed results."""
    gpt = GPTPlayer()
    llama = LlamaPlayer()
    deepseek = DeepSeekPlayer()
    qwen = QwenPlayer()
    grok = GrokPlayer()
    gemini = GeminiPlayer()
    lag = LooseAggressiveMontePlayer()
    tp = TightPassiveMontePlayer()
    ta =  TightAggressiveMontePlayer()
    lp = LoosePassiveMontePlayer()
    Maniac = ManiacMontePlayer()

    player_names = get_player_aliases()

    config = setup_config(max_round=rounds, initial_stack=initial_stack, small_blind_amount=small_blind)
    config.register_player(name=player_names["gpt"], algorithm=gpt)
    config.register_player(name=player_names["llama"], algorithm=llama)
    config.register_player(name=player_names["grok"], algorithm=grok)
    config.register_player(name=player_names["deepseek"], algorithm=deepseek)
    config.register_player(name=player_names["qwen"], algorithm=qwen)
    config.register_player(name=player_names["gemini"], algorithm=gemini)
    config.register_player(name=player_names["lag"], algorithm=lag)
    config.register_player(name=player_names["tp"], algorithm=tp)
    config.register_player(name=player_names["ta"], algorithm=ta)
    config.register_player(name=player_names["lp"], algorithm=lp)
    config.register_player(name=player_names["Maniac"], algorithm=Maniac)

    game_result = start_poker(config, verbose=VERBOSE_LEVEL)

    results = {
        "game_info": {
            "timestamp": datetime.now().isoformat(),
            "rounds": rounds,
            "initial_stack": initial_stack,
            "small_blind": small_blind
        },
        "players": {},
        "reasoning_history": {}
    }

    player_algorithms = [
        (player_names["llama"], llama),
        (player_names["grok"], grok),
        (player_names["gpt"], gpt),
        (player_names["deepseek"], deepseek),
        (player_names["qwen"], qwen),
        (player_names["gemini"], gemini),
        (player_names["lag"], lag),
        (player_names["tp"], tp),
        (player_names["ta"], ta),
        (player_names["lp"], lp),
        (player_names["Maniac"], Maniac),

    ]

    for player_name, algorithm in player_algorithms:
        stats = algorithm.get_performance_stats()
        results["players"][player_name] = stats

        if hasattr(algorithm, 'reasoning_history'):
            results["reasoning_history"][player_name] = algorithm.reasoning_history

    return results

def run_single_battle_worker(battle_idx, exp_dir, rounds, initial_stack, small_blind, verbose_level):
    """Run one battle in a child process, writing to a per-battle file to avoid
    concurrent writes to results.json."""
    # Each battle redirects stdout to its own log file so parallel workers
    # don't interleave on sys.stdout.
    game_log_path = os.path.join(exp_dir, f"game_output_battle_{battle_idx:04d}.log")

    with open(game_log_path, "a", encoding="utf-8") as f:
        with redirect_stdout(f):
            result = run_poker_game(
                rounds=rounds,
                initial_stack=initial_stack,
                small_blind=small_blind
            )

    battle_dir = os.path.join(exp_dir, "battle_results")
    os.makedirs(battle_dir, exist_ok=True)
    battle_path = os.path.join(battle_dir, f"battle_{battle_idx:04d}.json")

    with open(battle_path, "w", encoding="utf-8") as wf:
        json.dump(result, wf, indent=2)

    return {
        "battle_idx": battle_idx,
        "battle_path": battle_path
    }


def main():
    exp_manager = ExperimentManager()

    exp_manager.logger.info("="*60)
    exp_manager.logger.info(f"Starting poker experiment: {exp_manager.experiment_id}")
    exp_manager.logger.info(f"Configuration: {NUM_BATTLES} battles, {ROUNDS_PER_BATTLE} rounds each")
    exp_manager.logger.info(f"Initial stack: {INITIAL_STACK}, Small blind: {SMALL_BLIND_AMOUNT}")
    exp_manager.logger.info(f"Results will be saved to: {exp_manager.exp_dir}")
    exp_manager.logger.info("="*60)

    # Tune down if the LLM API starts rate-limiting.
    MAX_WORKERS = 5

    # Resume detection: any battle_XXXX.json already on disk counts as done,
    # so a killed run can be restarted by just invoking the script again.
    battle_dir = os.path.join(exp_manager.exp_dir, "battle_results")
    os.makedirs(battle_dir, exist_ok=True)

    done = set()
    for fn in os.listdir(battle_dir):
        if fn.startswith("battle_") and fn.endswith(".json"):
            idx = int(fn.split("_")[1].split(".")[0])
            done.add(idx)

    pending = [i for i in range(NUM_BATTLES) if i not in done]
    exp_manager.logger.info(f"Found {len(done)} completed battles, {len(pending)} pending.")

    futures = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i in pending:
            futures.append(ex.submit(
                run_single_battle_worker,
                i,
                exp_manager.exp_dir,
                ROUNDS_PER_BATTLE,
                INITIAL_STACK,
                SMALL_BLIND_AMOUNT,
                VERBOSE_LEVEL
            ))

        for fu in as_completed(futures):
            info = fu.result()
            exp_manager.logger.info(f"Battle {info['battle_idx']+1}/{NUM_BATTLES} finished -> {info['battle_path']}")

    # Merge per-battle files into results.json from the main process only.
    battles = [None] * NUM_BATTLES
    for i in range(NUM_BATTLES):
        p = os.path.join(battle_dir, f"battle_{i:04d}.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                battles[i] = json.load(f)

    final_results = {
        "battles": battles,
        "experiment_info": {
            "experiment_id": exp_manager.experiment_id,
            "total_battles": NUM_BATTLES,
            "completed_battles": sum(x is not None for x in battles),
            "last_updated": datetime.now().isoformat(),
            "status": "completed" if all(x is not None for x in battles) else "running"
        }
    }

    with open(exp_manager.results_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    exp_manager.logger.info("All battles merged into results.json")
    exp_manager.logger.info("="*60)
    exp_manager.logger.info("Experiment finished")
    exp_manager.logger.info("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run poker game experiments")
    parser.add_argument("--resume", type=str, help="Resume experiment with given ID")
    args = parser.parse_args()

    if args.resume:
        if not os.path.exists(os.path.join(EXPERIMENTS_DIR, args.resume)):
            print(f"Error: Experiment directory {args.resume} not found")
            sys.exit(1)

        exp_manager = ExperimentManager(args.resume)

        exp_manager.logger.info("="*60)
        exp_manager.logger.info(f"Resuming poker experiment: {exp_manager.experiment_id}")
        exp_manager.logger.info("="*60)

        start_battle = exp_manager.get_starting_battle()
        exp_manager.logger.info(f"Resuming from battle {start_battle + 1}")

        try:
            for i in range(start_battle, NUM_BATTLES):
                exp_manager.logger.info(f"Starting BATTLE {i+1}/{NUM_BATTLES}")
                exp_manager.logger.info("="*50)

                exp_manager.start_battle_timer()
                exp_manager.start_game_logging()

                try:
                    battle_result = run_poker_game(
                        rounds=ROUNDS_PER_BATTLE,
                        initial_stack=INITIAL_STACK,
                        small_blind=SMALL_BLIND_AMOUNT
                    )
                finally:
                    # Stop logging even if the game raises.
                    exp_manager.stop_game_logging()

                battle_duration = exp_manager.end_battle_timer()

                exp_manager.save_battle_result(i, battle_result)

                exp_manager.log_progress_stats(i, battle_duration)
                exp_manager.logger.info("-"*50)

            exp_manager.logger.info("="*60)
            exp_manager.logger.info("All battles completed successfully!")
            exp_manager.logger.info(f"Results saved in: {exp_manager.exp_dir}")
            exp_manager.logger.info("="*60)

            if os.path.exists(exp_manager.resume_file):
                os.remove(exp_manager.resume_file)

        except KeyboardInterrupt:
            exp_manager.logger.warning("Experiment interrupted by user")
            exp_manager.logger.info(f"Progress saved. Resume with: python run_game.py --resume {exp_manager.experiment_id}")
        except Exception as e:
            exp_manager.logger.error(f"Experiment failed with error: {str(e)}")
            exp_manager.logger.info(f"Progress saved. Resume with: python run_game.py --resume {exp_manager.experiment_id}")
            raise
    else:
        main()
