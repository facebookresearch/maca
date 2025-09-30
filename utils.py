"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""Utility functions for experiment setup, device management, and logging."""

import os
import gc
import json
import torch
import random
import numpy as np
from datetime import datetime
from transformers import TrainerCallback, set_seed
import time
import subprocess
import warnings
import logging as _logging
import inspect

verbose = False
# 1. Filter the warnings module (covers warnings.warn calls)
warnings.filterwarnings(
    "ignore",
    message=r".*incompatible with gradient checkpointing.*",
)

# 2. Attach a logging filter to drop any log records that still contain the text.
class _NoGradCkptCacheFilter(_logging.Filter):
    def filter(self, record):
        return "incompatible with gradient checkpointing" not in record.getMessage()

# Attach to *all* relevant loggers – root and the whole transformers subtree.
root_logger = _logging.getLogger()
root_logger.addFilter(_NoGradCkptCacheFilter())
_logging.getLogger("transformers").addFilter(_NoGradCkptCacheFilter())

# Tell HuggingFace Transformers to mute its own advisory helpers at the source.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# 3. Last resort – intercept writes to stderr/stdout that slip through the
#     logging/ warnings layers (the Qwen-2 code path prints directly).
import sys, io

# A smarter line-buffering wrapper: hold writes until a newline is seen so we
# can decide whether to pass or drop the *entire* line. This prevents the
# situation where a prefix (e.g. "Agent-1 | ") is written but the remainder of
# the advisory message is filtered out – which would leave stray blank or
# partial lines clogging the log.

class _LineFilterStream(io.TextIOBase):
    _TRIGGER_PHRASES = (
        "incompatible with gradient checkpointing",
        "`use_cache=True` is incompatible with gradient checkpointing",
    )

    def __init__(self, stream):
        self._stream = stream
        self._buffer = ""

    def write(self, msg):
        self._buffer += msg
        written = len(msg)

        # Process complete lines
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if any(phrase in line for phrase in self._TRIGGER_PHRASES):
                # Drop the entire line (including the newline)
                continue
            self._stream.write(line + "\n")

        return written

    def flush(self):
        # Flush whatever is left in the buffer (incomplete line)
        if self._buffer and not any(phrase in self._buffer for phrase in self._TRIGGER_PHRASES):
            self._stream.write(self._buffer)
        self._buffer = ""
        self._stream.flush()

    def isatty(self):
        return False

if not isinstance(sys.stderr, _LineFilterStream):
    sys.stderr = _LineFilterStream(sys.stderr)
if not isinstance(sys.stdout, _LineFilterStream):
    sys.stdout = _LineFilterStream(sys.stdout)

def pd(msg):
    if verbose:
        print(msg)
    return

def setup_environment():
    if torch.cuda.is_available():
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(torch.cuda.device_count())))
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
        os.environ['OMP_NUM_THREADS'] = '4'
        print(f'Using {torch.cuda.device_count()} GPUs')
        cleanup_gpu()
    else:
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        os.environ['OMP_NUM_THREADS'] = '4'
        print('Running on CPU')


def set_random_seed(seed):
    """Set the seed for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_available_devices():
    if torch.cuda.is_available():
        # Get the number of visible devices from CUDA_VISIBLE_DEVICES
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible_devices:
            # Convert the comma-separated string to a list of device indices
            device_indices = [int(x) for x in visible_devices.split(',')]
            return [f'cuda:{i}' for i in range(len(device_indices))]
        return [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    return ['cpu']


def get_agent_device_assignment(
    num_agents: int,
    max_agents_per_device: int,
    devices: list
):
    if len(devices) == 1 and devices[0] == 'cpu':
        return {0: {i:devices[0] for i in range(num_agents)}}
    
    max_debates = (len(devices) * max_agents_per_device) // num_agents
    current_device, current_capacity = 0, max_agents_per_device
    debta_map = {i:{} for i in range(max_debates)}
    for i in range(max_debates):
        for j in range(num_agents):
            if current_capacity > 0:
                debta_map[i].update({j: devices[current_device]})
                current_capacity -= 1
            else:
                current_device += 1
                current_capacity = max_agents_per_device - 1
                debta_map[i].update({j: devices[current_device]})
    return debta_map


def cleanup_gpu():
    """GPU cleanup to prevent memory leakage between training phases."""
    if not torch.cuda.is_available():
        return
        
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    # Get initial memory stats for all GPUs
    initial_memory = {}
    for gpu_id in range(torch.cuda.device_count()):
        try:
            initial_memory[gpu_id] = torch.cuda.memory_allocated(gpu_id) / (1024**3)
        except:
            initial_memory[gpu_id] = 0
    
    try:
        # Method 1: Standard PyTorch cleanup for all GPUs
        for gpu_id in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(gpu_id)
                    torch.cuda.reset_accumulated_memory_stats(gpu_id)
                    torch.cuda.synchronize(gpu_id)
            except Exception as e:
                print(f"Warning: Standard cleanup failed for GPU {gpu_id}: {e}")
        
        # Method 2: Force garbage collection again
        gc.collect()
        
        # Method 3: Try to reset CUDA context
        try:
            current_device = torch.cuda.current_device()
            for gpu_id in range(torch.cuda.device_count()):
                try:
                    torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
                    # Force context synchronization
                    torch.cuda.synchronize()
                except Exception as e:
                    print(f"Warning: Context reset failed for GPU {gpu_id}: {e}")
            
            # Restore original device
            torch.cuda.set_device(current_device)
        except Exception as e:
            print(f"Warning: CUDA context reset failed: {e}")
        
        # Method 4: Use nvidia-ml-py for process-level cleanup if available
        try:
            import pynvml
            pynvml.nvmlInit()
            for gpu_id in range(torch.cuda.device_count()):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    # Get process info to identify leaked processes
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    current_pid = os.getpid()
                    
                    # Check if our process is using too much memory
                    for proc in processes:
                        if proc.pid == current_pid and proc.usedGpuMemory > 1024**3:  # > 1GB
                            print(f"Warning: Current process using {proc.usedGpuMemory/(1024**3):.2f} GB on GPU {gpu_id}")
                except Exception as e:
                    print(f"Warning: nvidia-ml check failed for GPU {gpu_id}: {e}")
                    
            pynvml.nvmlShutdown()
        except (ImportError, Exception) as e:
            print(f"Warning: nvidia-ml-py not available or failed: {e}")
        
        # Method 5: Set environment variable for memory management
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Method 6: Final cleanup round
        for _ in range(2):
            gc.collect()
            torch.cuda.empty_cache()
        
        # Method 7: Try subprocess-based nvidia-smi memory reset
        try:
            for gpu_id in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                if allocated > 5.0:  # If using more than 5GB
                    # print(f"  GPU {gpu_id} using {allocated:.2f} GB - attempting subprocess cleanup")
                    try:
                        # Try to reset GPU memory via subprocess (last resort)
                        result = subprocess.run(
                            ["nvidia-smi", "--gpu-reset", "-i", str(gpu_id)], 
                            capture_output=True, 
                            text=True, 
                            timeout=10
                        )
                        if result.returncode != 0:
                            print(f"GPU reset command failed for GPU {gpu_id}")
                    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                        print(f"GPU reset via subprocess failed: {e}")
        except Exception as e:
            print(f"Warning: Subprocess cleanup failed: {e}")

        total_freed = 0
        for gpu_id in range(torch.cuda.device_count()):
            try:
                final_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                freed = initial_memory[gpu_id] - final_memory
                total_freed += freed
            except:
                print(f"   GPU {gpu_id}: Memory check failed")

    except Exception as e:
        print(f"  GPU cleanup encountered errors: {e}")

def setup_experiment_dir(config):
    """Create and setup experiment directory"""
    config['timestamp'] = str(datetime.now())
    experiment_id = f'{abs(hash(str(config)))}'
    experiment_dir = f"experiments/{experiment_id}"
    os.makedirs(experiment_dir, exist_ok=True)
    with open(f"{experiment_dir}/config.json", "w") as f:
        json.dump(config, f)
    return experiment_dir


def save_results(
    experiment_dir: str,
    train_accuracy: float,
    test_accuracy: float
):
    """Save training results"""
    results = {
        'train': train_accuracy,
        'test': test_accuracy,
        'timestamp': str(datetime.now())
    }
    with open(f"{experiment_dir}/results.json", "w") as f:
        json.dump(results, f)

def read_baselines(
    model: str,
    dataset: str,
    rounds: int = 2,
    agents: int = 3
):
    from glob import glob
    from debate import majority_vote
    from parser import parse_answer, grade_answer
    files = glob(f"data/debate/{model}_{dataset}_{agents}_{rounds}_*_False.json")
    # assert len(files) == 2, "Data for all 3 seeds is not available."

    single_train_acc, single_test_acc = [], []
    single_train_acc_std, single_test_acc_std = [], []
    majority_train_acc, majority_test_acc = [], []
    majority_train_acc_std, majority_test_acc_std = [], []
    debate_train_acc, debate_test_acc = [], []
    debate_train_acc_std, debate_test_acc_std = [], []
    for file in files[:2]:
        with open(file, 'r') as f:
            debate = {}
            for k, v in json.load(f).items():
                if k == 'metrics':
                    continue
                if v['split'] == 'train':
                    debate[k] = v
                elif v['split'] == 'test':
                    debate[k] = v

            tmp_debate_train, tmp_debate_test = [], []
            tmp_majority_train, tmp_majority_test = [], []
            tmp_single_train, tmp_single_test = [], []
            for k, v in debate.items():
                if k == 'metrics':
                    continue

                agent_answers = []
                for i in range(agents):
                    answer = parse_answer(v['context'][i][1]['content'], dataset)
                    if answer is not None:
                        agent_answers.append(answer)
                if len(agent_answers) > 0:
                    conses = int(grade_answer(majority_vote(agent_answers), v['ground_truth']))
                    single = np.mean([int(grade_answer(a, v['ground_truth'])) for a in agent_answers])
                else :
                    conses = None
                    single = None

                agent_answers = []
                for i in range(agents):
                    answer = parse_answer(v['context'][i][-1]['content'], dataset)
                    if answer is not None:
                        agent_answers.append(answer)
                if len(agent_answers) > 0:
                    deb = int(grade_answer(majority_vote(agent_answers), v['ground_truth']))
                else :
                    deb = None

                if v['split'] == 'train':
                    if deb is not None:
                        tmp_debate_train.append(deb)
                    if conses is not None:
                        tmp_majority_train.append(conses)
                    if single is not None:
                        tmp_single_train.append(single)
                elif v['split'] == 'test':
                    if deb is not None:
                        tmp_debate_test.append(deb)
                    if conses is not None:
                        tmp_majority_test.append(conses)
                    if single is not None:
                        tmp_single_test.append(single)
            
            single_train_acc.append(np.mean(tmp_single_train))
            single_train_acc_std.append(np.std(tmp_single_train) / np.sqrt(500))
            single_test_acc.append(np.mean(tmp_single_test))
            single_test_acc_std.append(np.std(tmp_single_test) / np.sqrt(500))
            majority_train_acc.append(np.mean(tmp_majority_train))
            majority_train_acc_std.append(np.std(tmp_majority_train) / np.sqrt(500))
            majority_test_acc.append(np.mean(tmp_majority_test))
            majority_test_acc_std.append(np.std(tmp_majority_test) / np.sqrt(500))
            debate_train_acc.append(np.mean(tmp_debate_train))
            debate_train_acc_std.append(np.std(tmp_debate_train) / np.sqrt(500))
            debate_test_acc.append(np.mean(tmp_debate_test))
            debate_test_acc_std.append(np.std(tmp_debate_test) / np.sqrt(500))

    print(model, dataset, rounds, agents)   
    print(f"Single Train Accuracy: {np.mean(single_train_acc) * 100:.2f} ± {np.mean(single_train_acc_std) * 100:.2f}")
    print(f"Single test Accuracy: {np.mean(single_test_acc) * 100:.2f} ± {np.mean(single_test_acc_std) * 100:.2f}")
    print(f"Majority Train Accuracy: {np.mean(majority_train_acc) * 100:.2f} ± {np.mean(majority_train_acc_std) * 100:.2f}")
    print(f"Majority test Accuracy: {np.mean(majority_test_acc) * 100:.2f} ± {np.mean(majority_test_acc_std) * 100:.2f}")
    print(f"Debate Train Accuracy: {np.mean(debate_train_acc) * 100:.2f} ± {np.mean(debate_train_acc_std) * 100:.2f}")
    print(f"Debate test Accuracy: {np.mean(debate_test_acc) * 100:.2f} ± {np.mean(debate_test_acc_std) * 100:.2f}")
    return debate

class WandbLoggingCallback(TrainerCallback):
    def __init__(self, agent_id, round, wandb_run, phase, log_interval=10):
        self.agent_id = agent_id
        self.wandb_run = wandb_run
        self.log_interval = log_interval
        self.phase = phase
        self.round=round
        self.last_logged_step = -1

        self.metrics_to_log = {
            'loss': 'Loss',
            'learning_rate': 'Learning Rate',
            'grad_norm': 'Gradient Norm',
            'mean_token_accuracy': 'Token Accuracy'
        }
        
        if self.wandb_run:
            self.wandb_run.define_metric("_step", hidden=True)

            for metric_key, metric_name in self.metrics_to_log.items():
                self.wandb_run.define_metric(
                    f"Agent_{self.agent_id}/{self.phase}_{self.round}/{metric_name}",
                    step_metric=f"_step"
                )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not self.wandb_run:
            return
        
        current_step = state.global_step

        if current_step > self.last_logged_step and current_step % self.log_interval == 0:
            log_dict = {
                "_step": current_step,
            }

            for key, value in logs.items():
                if key in self.metrics_to_log:
                    metric_name = f"Agent_{self.agent_id}/{self.phase}_{self.round}/{self.metrics_to_log[key]}"
                    log_dict[metric_name] = value
            
            if len(log_dict) > 1:
                self.wandb_run.log(log_dict)
                self.last_logged_step = current_step
    
    def on_train_begin(self, args, state, control, **kwargs):
        state.global_step = 0
        if hasattr(args, 'report_to'):
            args.report_to = []

    def on_evaluate(self, args, state, control, **kwargs):
        return control
    

# ---------------------------------------------------------------------------
# Compatibility patch: transformers >=4.42 changed Trainer._get_train_sampler to
# expect the *dataset* argument.  Older TRL releases (where KTOTrainer lives)
# still implement `_get_train_sampler(self)` and therefore crash with
# `TypeError: ... takes 1 positional argument but 2 were given`.
# If we detect that situation we patch the method at import-time so the whole
# code-base continues to work without downgrading either library.
# ---------------------------------------------------------------------------

try:
    from trl.trainer.kto_trainer import KTOTrainer  # noqa: E402
    from transformers.trainer import Trainer as _HFTrainer  # noqa: E402

    _sig = inspect.signature(KTOTrainer._get_train_sampler)
    if len(_sig.parameters) == 1:  # only "self" → old style, needs patch
        def _patched_get_train_sampler(self, dataset):  # type: ignore
            """Delegate to HF Trainer's implementation (new signature)."""
            return _HFTrainer._get_train_sampler(self, dataset)

        KTOTrainer._get_train_sampler = _patched_get_train_sampler  # type: ignore
        pd("Patched KTOTrainer._get_train_sampler for transformers ≥ 4.42 compatibility")
except Exception as _e:  # pragma: no cover – patch is best-effort
    # If TRL is not installed or the API already matches we silently continue.
    pass

# ------------------------------------------------------------
# Extra logging for TRL KTOTrainer: logs pos/neg reward, gap, KL
# ------------------------------------------------------------

class KTORewardLoggingCallback(TrainerCallback):
    """Forward KTO-specific statistics from `state.log_history` to WandB."""

    def __init__(self, agent_id: int, round: int, wandb_run):
        self.prefix = f"Agent_{agent_id}/KTO_{round}"
        self.wandb_run = wandb_run

    def on_log(self, args, state, control, **kwargs):
        if not self.wandb_run or not state.log_history:
            return

        last = state.log_history[-1]

        # Common key names in TRL >=0.8 for KTOTrainer
        metric_keys = [
            "pos_reward",      # desirable reward
            "neg_reward",      # undesirable reward
            "gap",             # pos - neg
            "kl",              # KL regulariser
            "loss"             # total loss already logged but keep for completeness
        ]

        payload = {}
        for k in metric_keys:
            if k in last:
                payload[f"{self.prefix}/{k}"] = last[k]

        if payload:
            self.wandb_run.log(payload, step=state.global_step)

# ------------------------------------------------------------
# Extra logging for TRL DPOTrainer: logs chosen/rejected rewards, accuracies, margins
# ------------------------------------------------------------

class DPORewardLoggingCallback(TrainerCallback):
    """Forward DPO-specific statistics from `state.log_history` to WandB."""

    def __init__(self, agent_id: int, round: int, wandb_run):
        self.prefix = f"Agent_{agent_id}/DPO_{round}"
        self.wandb_run = wandb_run

    def on_log(self, args, state, control, **kwargs):
        if not self.wandb_run or not state.log_history:
            return

        last = state.log_history[-1]

        # DPO-specific metric keys from HuggingFace TRL DPOTrainer
        metric_keys = [
            "rewards/chosen",      # mean difference between policy and reference model for chosen responses
            "rewards/rejected",    # mean difference between policy and reference model for rejected responses  
            "rewards/accuracies",  # mean of how often chosen rewards > rejected rewards
            "rewards/margins",     # mean difference between chosen and rejected rewards
            "loss"                 # total loss already logged but keep for completeness
        ]

        payload = {}
        for k in metric_keys:
            if k in last:
                # Convert rewards/chosen to chosen_reward for cleaner WandB naming
                clean_key = k.replace("rewards/", "").replace("/", "_")
                payload[f"{self.prefix}/{clean_key}"] = last[k]

        if payload:
            self.wandb_run.log(payload, step=state.global_step)

if __name__ == "__main__":

    for m in ["qwen2b", "llama3b", "phi4b"]:
        for d in ['gsm8k', 'math']:
            print(f"==================== {m} {d} ====================")
            read_baselines(m, d)
            