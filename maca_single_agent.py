#!/usr/bin/env python3
"""
Single Agent Training and Hyperparameter Tuning Script

Usage Examples:

HP Tuning:
    python maca_single_agent.py --lr_rl_range 1e-6 1e-5 --batch_rl_range 4 8

Normal Training + Test Evaluation:
    python maca_single_agent.py --use_full_test

Evaluation Only:
    python maca_single_agent.py --use_full_test --skip_training
"""

import os
import sys
import json
import argparse
import subprocess
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import wandb

def setup_gpu_environment(gpu_id: int) -> None:
    """Set up GPU environment for the specified GPU ID."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"HP Tuning | Set CUDA_VISIBLE_DEVICES={gpu_id}")
    
    if torch.cuda.is_available():
        print(f"HP Tuning | GPU {gpu_id} is available (mapped to cuda:0)")
        print(f"HP Tuning | GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        print(f"HP Tuning | Warning: CUDA not available after setting GPU {gpu_id}")

def build_debate_path(
    model: str,
    dataset: str,
    agents: int = 3,
    rounds: int = 2,
    seed: int = 0,
    diversity_prompt: bool = False,
    summarize: bool = False
) -> str:
    """Construct the OLD-format debate file path (no train/test sizes)."""
    filename = f"{model}_{dataset}_{agents}_{rounds}_{seed}_{diversity_prompt}_{summarize}.json"
    return os.path.join("data", "debate", filename)


def build_debate_path_new(
    model: str,
    dataset: str,
    train_size: int,
    test_size: int,
    agents: int,
    rounds: int,
    data_seed: int,
    diversity_prompt: bool,
    summarize: bool
) -> str:
    """Construct the NEW-format debate file path (includes train/test sizes).

    Mirrors orchestrator.py naming:
    data/debate/{model}_{dataset}_{train_size}_{test_size}_{agents}_{rounds}_{data_seed}_{diversity_prompt}_{summarize}.json
    """
    filename = (
        f"{model}_{dataset}_{train_size}_{test_size}_{agents}_{rounds}_{data_seed}_{diversity_prompt}_{summarize}.json"
    )
    return os.path.join("data", "debate", filename)


def resolve_debate_file(args) -> str:
    """Resolve debate file supporting both naming formats.

    Preference order: new-format (with sizes) → old-format (no sizes).
    If --debate_file is provided and exists, return it. If provided but missing,
    try to auto-resolve to one of the known formats.
    """
    # If user provided an explicit path and it exists, use it
    if args.debate_file and os.path.exists(args.debate_file):
        return args.debate_file

    # Compute candidate paths from current args, mirroring orchestrator.py
    candidate_new = build_debate_path_new(
        model=args.model,
        dataset=args.dataset,
        train_size=args.train_size,
        test_size=args.test_size,
        agents=args.agents,
        rounds=args.rounds,
        data_seed=args.data_seed,
        diversity_prompt=args.diversity_prompt,
        summarize=args.summarize,
    )
    candidate_old = build_debate_path(
        model=args.model,
        dataset=args.dataset,
        agents=args.agents,
        rounds=args.rounds,
        seed=args.data_seed,  # use data_seed to match orchestrator naming
        diversity_prompt=args.diversity_prompt,
        summarize=args.summarize,
    )

    # Prefer new-format if it exists
    if os.path.exists(candidate_new):
        return candidate_new
    if os.path.exists(candidate_old):
        return candidate_old

    # If a non-existing explicit path was given, try swapping formats once
    if args.debate_file:
        # If user gave old-format path, try new-format path; and vice-versa
        alt = candidate_new if args.debate_file.endswith(os.path.basename(candidate_old)) else candidate_old
        if os.path.exists(alt):
            return alt

    # Nothing resolved
    return ""

def resolve_debate_file_for_dataset(
    args,
    dataset_name: str
) -> str:
    """Resolve debate file for a specific dataset name using both naming formats."""
    # If user provided an explicit path and it exists, prefer it when it contains the dataset name
    if args.debate_file and os.path.exists(args.debate_file):
        if dataset_name in os.path.basename(args.debate_file):
            return args.debate_file
    candidate_new = build_debate_path_new(
        model=args.model,
        dataset=dataset_name,
        train_size=args.train_size,
        test_size=args.test_size,
        agents=args.agents,
        rounds=args.rounds,
        data_seed=args.data_seed,
        diversity_prompt=args.diversity_prompt,
        summarize=args.summarize,
    )
    candidate_old = build_debate_path(
        model=args.model,
        dataset=dataset_name,
        agents=args.agents,
        rounds=args.rounds,
        seed=args.data_seed,
        diversity_prompt=args.diversity_prompt,
        summarize=args.summarize,
    )
    if os.path.exists(candidate_new):
        return candidate_new
    if os.path.exists(candidate_old):
        return candidate_old
    return ""

def merge_balanced_training_sets(
    dataset_to_file: Dict[str, str],
    seed: int = 0
) -> Dict[str, Any]:
    """Merge multiple debate JSONs into a balanced training set via undersampling and interleaving.

    Only 'train' examples are included; evaluation is handled separately per dataset.
    """
    import random as _rnd
    _rnd.seed(seed)
    per_dataset_trains: Dict[str, List[tuple[str, Any]]] = {}
    min_count = None
    for dset, path in dataset_to_file.items():
        with open(path, 'r') as f:
            data = json.load(f)
        items = [(k, v) for k, v in data.items() if k != 'metrics' and v.get('split') == 'train']
        _rnd.shuffle(items)
        per_dataset_trains[dset] = items
        cnt = len(items)
        if min_count is None or cnt < min_count:
            min_count = cnt
    min_count = min_count or 0
    # Trim and interleave
    trimmed = {dset: items[:min_count] for dset, items in per_dataset_trains.items()}
    interleaved: List[tuple[str, Any]] = []
    for i in range(min_count):
        for dset in sorted(trimmed.keys()):
            interleaved.append((dset, trimmed[dset][i][1]))
    # Keep deterministic interleaving order to encourage balanced per-batch sampling downstream
    merged: Dict[str, Any] = {}
    for idx, (dset, ex) in enumerate(interleaved):
        merged[f"{dset}:{idx}"] = ex
    merged['metrics'] = {
        'merged_from_datasets': list(dataset_to_file.keys()),
        'per_dataset_min_train': min_count,
        'total_train_examples': len(interleaved),
        'balanced_sampling': 'undersample_to_min_round_robin'
    }
    return merged

def load_debate_data(
    debate_file: str
) -> Dict[str, Any]:
    """Load debate data from JSON file."""
    print(f"HP Tuning | Loading debate data from {debate_file}")
    with open(debate_file, 'r') as f:
        debate_data = json.load(f)
    
    # Count examples
    train_examples = sum(1 for k, v in debate_data.items() 
                        if k != 'metrics' and v.get('split') == 'train')
    test_examples = sum(1 for k, v in debate_data.items() 
                       if k != 'metrics' and v.get('split') == 'test')
    
    print(f"HP Tuning | Loaded debate data: {train_examples} train, {test_examples} test examples")
    return debate_data

def create_trial_folder_name(
    config: Dict[str, Any],
    hp_combination: Dict[str, Any]
) -> str:
    """Create a descriptive folder name based on hyperparameter values."""
    name_parts = []
    
    # Add model, dataset, and phase at the beginning
    model_name = config.get('model', 'unknown')
    
    # Handle multiple training datasets
    train_datasets = config.get('train_datasets', [])
    if train_datasets:
        # Use all training datasets in the folder name
        dataset_name = "_".join(train_datasets)
    else:
        # Fall back to single dataset name
        dataset_name = config.get('dataset_label', config.get('dataset', 'unknown'))
    
    phase = config.get('phase', 'unknown')
    
    # Add context suffixes based on flags
    context_suffix = ""
    if config.get('pref_debate_context', False):
        context_suffix = "_pdc"
    elif not config.get('include_debate_context', True):  # If include_debate_context is False
        context_suffix = "_nc"
    
    # Add initial responses suffix
    if config.get('use_initial_responses', False):
        context_suffix += "_ir"
    
    # Add ground truth suffix
    if config.get('use_ground_truth', False):
        context_suffix += "_gt"
    
    # If skip_training is enabled, avoid creating nested subfolders
    if config.get('skip_training', False) or config.get('test_only', False):
        # Trial will write directly into the provided output_dir
        return ""
    
    name_parts.append(f"{model_name}_{dataset_name}_{phase}{context_suffix}")
    
    # Common hyperparameters to include in folder name
    hp_mapping = {
        'lora_r': 'lorar',
        'lora_alpha': 'loraalpha', 
        'lr_sft': 'lrsft',
        'lr_rl': 'lrrl',
        'lr_kto': 'lrkto',
        'lr_dpo': 'lrdpo',
        'batch_size_sft': 'bsft',
        'batch_rl': 'brl',
        'batch_kto': 'bkto',
        'batch_dpo': 'bdpo',
        'epoch_sft': 'esft',
        'epoch_rl': 'erl',
        'epoch_kto': 'ekto',
        'epoch_dpo': 'edpo',
        'beta_kto': 'betakto',
        'beta_dpo': 'betadpo',
        'num_generations': 'ngen',
        'consensus_weight': 'cweight',
        'gradient_accumulation_steps': 'gas',
        'gradient_accumulation_steps_kto': 'gaskto',
        'gradient_accumulation_steps_dpo': 'gasdpo'
    }
    
    for param_name, param_value in hp_combination.items():
        if param_name in hp_mapping:
            # Format the value appropriately
            if isinstance(param_value, float):
                # For learning rates, use scientific notation
                if param_value < 0.01:
                    formatted_value = f"{param_value:.0e}".replace('e-0', 'e').replace('e+0', 'e')
                else:
                    formatted_value = str(param_value).replace('.', '')
            else:
                formatted_value = str(param_value)
            
            name_parts.append(f"{hp_mapping[param_name]}{formatted_value}")
    
    # If no hyperparameters were included, fall back to trial number
    if not name_parts:
        return f"{model_name}_{dataset_name}_{phase}{context_suffix}_trial_{config['hp_run_id']}"
    
    # Join parts and limit length
    folder_name = "_".join(name_parts)
    
    # If name is too long, truncate and add hash
    if len(folder_name) > 80:  # Increased limit since we have model+dataset prefix
        import hashlib
        hash_suffix = hashlib.md5(folder_name.encode()).hexdigest()[:8]
        folder_name = folder_name[:70] + "_" + hash_suffix
    
    return folder_name


def create_hp_configs(
    base_config: Dict[str, Any],
    hp_args: Dict[str, List]
) -> List[Dict[str, Any]]:
    """Create hyperparameter configurations from base config and HP arguments."""
    configs = []
    
    # Generate all combinations
    from itertools import product
    
    # Prepare parameter lists
    param_names = list(hp_args.keys())
    param_values = list(hp_args.values())
    
    # Generate all combinations
    combinations = list(product(*param_values))
    
    print(f"HP Tuning | Generating {len(combinations)} hyperparameter configurations")
    
    # Create configs
    for i, combination in enumerate(combinations):
        config = base_config.copy()
        
        # Apply hyperparameters
        for param_name, param_value in zip(param_names, combination):
            config[param_name] = param_value
        
        # Add run metadata
        config['hp_run_id'] = i
        config['hp_combination'] = dict(zip(param_names, combination))
        
        configs.append(config)
    
    return configs

def run_single_hp_trial(agent_idx: int, 
                       gpu_id: int,
                       phase: str,
                       iter_idx: int,
                       experiment_dir: str,
                       debate_data_file: str,
                       config: Dict[str, Any],
                       rewards_file: Optional[str] = None,
                       evaluate_after_training: bool = False,
                       evaluation_batch_size: int = 8,
                       evaluation_timeout: int = 7200,
                       training_timeout: int = -1,
                       skip_training: bool = False) -> tuple[bool, float]:
    """Run a single hyperparameter trial using train_agent_subprocess.py."""
    
    # Create config file for this trial
    config_file = os.path.join(experiment_dir, f"config_hp_{config['hp_run_id']}.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"HP Tuning | Running trial {config['hp_run_id']} on GPU {gpu_id}")
    print(f"HP Tuning | Hyperparameters: {config['hp_combination']}")

    if skip_training:
        print(f"HP Tuning | Skipping training, running evaluation only")
        # Prefer evaluating the trained checkpoint if it exists; otherwise fall back to base model
        default_ckpt = os.path.join(experiment_dir, "checkpoints", f"agent_{agent_idx}")
        has_ckpt = os.path.exists(config.get('checkpoint_path') or default_ckpt)
        use_base = not has_ckpt
        # For evaluation-only runs: evaluate only explicit test datasets (no implicit primary eval)
        primary_dataset = config.get('dataset')
        eval_success, eval_metrics = True, {}
        test_datasets = config.get('test_datasets', [primary_dataset])
        for dset in test_datasets:
            print(f"HP Tuning | Evaluating test dataset: {dset}")
            try:
                evaluate_trained_model(
                    agent_idx=agent_idx,
                    gpu_id=gpu_id,
                    experiment_dir=experiment_dir,
                    config=config,
                    debate_data_file=debate_data_file,
                    batch_size=evaluation_batch_size,
                    timeout=evaluation_timeout,
                    use_base_model=use_base,
                    use_test=config.get('use_test', False),
                    use_full_test=True if not config.get('use_test', False) else config.get('use_full_test', False),
                    fold_id=config.get('cv_fold_id', 0),
                    total_folds=config.get('cv_total_folds', 1),
                    dataset_override=dset,
                    results_suffix=dset if dset != primary_dataset else None
                )
            except Exception as e:
                print(f"HP Tuning | Evaluation failed for dataset {dset}: {e}")

        if eval_success:
            print(f"HP Tuning | Evaluation completed successfully")
            print(f"HP Tuning | Test accuracy: {eval_metrics.get('accuracy', 0):.4f}")
            return True, 0.0  # No training time
        else:
            print(f"HP Tuning | Evaluation failed")
            return False, 0.0
    
    # Build command for training
    cmd = [
        sys.executable, 'train_agent_subprocess.py',
        '--agent_idx', str(agent_idx),
        '--gpu_id', str(gpu_id),
        '--phase', phase,
        '--iter_idx', str(iter_idx),
        '--experiment_dir', experiment_dir,
        '--debate_data_file', debate_data_file,
        '--config_file', config_file
    ]
    
    if rewards_file and os.path.exists(rewards_file):
        cmd.extend(['--rewards_file', rewards_file])
    
    print(f"HP Tuning | Training timeout: {training_timeout}s ({'no timeout' if training_timeout == -1 else f'{training_timeout//3600}h {training_timeout%3600//60}m'})")
    
    # Run subprocess
    start_time = time.time()
    try:
        # Use configurable timeout (no timeout if -1)
        timeout = None if training_timeout == -1 else training_timeout
        result = subprocess.run(cmd, timeout=timeout)
        
        runtime = time.time() - start_time
        
        if result.returncode == 0:
            print(f"HP Tuning | Trial {config['hp_run_id']} completed successfully in {runtime:.1f}s")
            
            # Run evaluation if requested
            if evaluate_after_training:
                print(f"HP Tuning | Starting evaluation for trial {config['hp_run_id']}")
                eval_success, eval_metrics = evaluate_trained_model(
                    agent_idx=agent_idx,
                    gpu_id=gpu_id,
                    experiment_dir=experiment_dir,
                    config=config,
                    debate_data_file=debate_data_file,
                    batch_size=evaluation_batch_size,
                    timeout=evaluation_timeout,
                    use_test=config.get('use_test', False),
                    use_full_test=config.get('use_full_test', False),
                    fold_id=config.get('cv_fold_id', 0),
                    total_folds=config.get('cv_total_folds', 1)
                )
                
                if eval_success:
                    print(f"HP Tuning | Evaluation completed successfully")
                    print(f"HP Tuning | Test accuracy: {eval_metrics.get('accuracy', 0):.4f}")
                else:
                    print(f"HP Tuning | Evaluation failed")
                # Evaluate additional test datasets if requested
                primary_dataset = config.get('dataset')
                test_datasets = config.get('test_datasets', [primary_dataset])
                for dset in test_datasets:
                    if dset == primary_dataset:
                        continue
                    print(f"HP Tuning | Evaluating additional test dataset: {dset}")
                    try:
                        evaluate_trained_model(
                            agent_idx=agent_idx,
                            gpu_id=gpu_id,
                            experiment_dir=experiment_dir,
                            config=config,
                            debate_data_file=debate_data_file,
                            batch_size=evaluation_batch_size,
                            timeout=evaluation_timeout,
                            use_test=config.get('use_test', False),
                            use_full_test=True if not config.get('use_test', False) else config.get('use_full_test', False),
                            fold_id=config.get('cv_fold_id', 0),
                            total_folds=config.get('cv_total_folds', 1),
                            dataset_override=dset,
                            results_suffix=dset
                        )
                    except Exception as e:
                        print(f"HP Tuning | Evaluation failed for dataset {dset}: {e}")
            
            return True, runtime
        else:
            print(f"HP Tuning | Trial {config['hp_run_id']} failed after {runtime:.1f}s")
            return False, runtime
            
    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        print(f"HP Tuning | Trial {config['hp_run_id']} timed out after {training_timeout}s")
        return False, runtime
    except Exception as e:
        runtime = time.time() - start_time
        print(f"HP Tuning | Trial {config['hp_run_id']} failed with exception: {e}")
        return False, runtime

def get_available_gpus(exclude_gpus: List[int] = None) -> List[int]:
    """Get list of available GPU IDs."""
    if not torch.cuda.is_available():
        return []
    
    exclude_gpus = exclude_gpus or []
    available_gpus = [i for i in range(torch.cuda.device_count()) if i not in exclude_gpus]
    return available_gpus

def setup_wandb_hp_tuning(
    config: Dict[str, Any],
    hp_ranges: Dict[str, List],
    phase: str
):
    """Set up WandB for hyperparameter tuning session."""
    # Skip wandb for test cases
    if config.get('entity_name') == 'test-entity':
        print("Skipping wandb initialization for test case")
        return None

    # Create a unique group name for this HP tuning session
    group_name = f"hp_tuning_{int(time.time())}"

    # Initialize WandB
    wandb.init(
        project=config.get('project_name', 'llm-marl'),
        entity=config.get('entity_name', 'llm-marl'),
        name=f"HP_Tuning_{config['model']}_{config['dataset']}_{phase}",
        group=group_name,
        job_type="hyperparameter_tuning",
        config={
            **config,
            'hp_ranges': hp_ranges,
            'total_trials': len(list(__import__('itertools').product(*hp_ranges.values())))
        },
        settings=wandb.Settings(start_method="thread", console="off")
    )
    
    return wandb.run

def log_trial_result(
    wandb_run,
    trial_id: int,
    hyperparameters: Dict[str, Any],
    success: bool,
    experiment_dir: str,
    runtime: float = None
):
    """Log trial result to WandB."""
    log_data = {
        'trial_id': trial_id,
        'success': success,
        'experiment_dir': experiment_dir,
        'runtime_seconds': runtime or 0,
    }
    
    # Add hyperparameters to log
    for param_name, param_value in hyperparameters.items():
        log_data[f'hp_{param_name}'] = param_value
    
    # Check for evaluation results
    eval_results_file = os.path.join(experiment_dir, "evaluation_results.json")
    if os.path.exists(eval_results_file):
        try:
            with open(eval_results_file, 'r') as f:
                eval_metrics = json.load(f)
            
            # Add evaluation metrics to log
            log_data['eval_accuracy'] = eval_metrics.get('accuracy', 0.0)
            log_data['eval_correct'] = eval_metrics.get('correct', 0)
            log_data['eval_total'] = eval_metrics.get('total', 0)
            print(f"HP Tuning | Trial {trial_id} evaluation accuracy: {eval_metrics.get('accuracy', 0.0):.4f}")
        except Exception as e:
            print(f"HP Tuning | Failed to load evaluation results: {e}")
    
    wandb_run.log(log_data)

def evaluate_trained_model(agent_idx: int, 
                         gpu_id: int,
                         experiment_dir: str,
                         config: Dict[str, Any],
                         debate_data_file: str,
                         batch_size: int = 8,
                         timeout: int = 7200,
                         use_base_model: bool = False,
                         use_test: bool = False,
                         use_full_test: bool = False,
                         fold_id: int = 0,
                         total_folds: int = 1,
                         dataset_override: Optional[str] = None,
                         results_suffix: Optional[str] = None) -> tuple[bool, Dict[str, Any]]:
    """
    Load the trained checkpoint and evaluate on test set using debate parsing.
    
    Args:
        agent_idx: Agent index that was trained
        gpu_id: GPU ID to use for evaluation
        experiment_dir: Directory containing the trained checkpoint
        config: Configuration dictionary
        debate_data_file: Path to debate data file
        batch_size: Batch size for evaluation generation
        timeout: Timeout for evaluation in seconds
        use_base_model: If True, evaluate on the base model directly, not a trained agent.
        use_test: If True, use test set from debate data (subset of original test set).
        use_full_test: If True, use the complete original test set (examples 1500-1999).
                      If both use_test and use_full_test are False, use validation set (examples 2000-2499).
        
    Returns:
        Tuple of (success, evaluation_metrics)
    """
    import torch
    import asyncio
    from model import Agent
    from data import get_data
    from debate import construct_question_prompt
    from parser import parse_answer, grade_answer
    
    print(f"HP Evaluation | Loading trained model from {experiment_dir}")
    
    # Set up GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        # Load the trained agent or base model
        if use_base_model:
            print(f"HP Evaluation | Using base model (no checkpoint)")
            checkpoint_path = None
        else:
            # Use custom checkpoint path if provided, otherwise use default
            if config.get('checkpoint_path'):
                checkpoint_path = config['checkpoint_path']
                print(f"HP Evaluation | Using custom checkpoint: {checkpoint_path}")
            else:
                checkpoint_path = os.path.join(experiment_dir, "checkpoints", f"agent_{agent_idx}")
                print(f"HP Evaluation | Using default checkpoint: {checkpoint_path}")
            
            if not os.path.exists(checkpoint_path):
                print(f"HP Evaluation | Checkpoint not found at {checkpoint_path}")
                return False, {}
        
        # Initialize agent with trained checkpoint or base model
        dataset_name = dataset_override if dataset_override else config['dataset']
        agent = Agent(
            model_name=config['model'],
            agent_id=agent_idx,
            device_map=f'cuda:{gpu_id}',  # Explicitly use the specified GPU
            checkpoint_path=checkpoint_path,
            seed=config['data_seed'],  # Use data_seed for evaluation
            quantization=config.get('use_quantization', True),
            adapter_mode=config.get('use_adapter_mode', True),
            task=dataset_name
        )
        
        # Debug: Check what device is being used
        device = agent.get_device()
        print(f"HP Evaluation | Agent device: {device}")
        print(f"HP Evaluation | CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"HP Evaluation | CUDA device count: {torch.cuda.device_count()}")
            print(f"HP Evaluation | Current CUDA device: {torch.cuda.current_device()}")
        
        if use_full_test:
            # Use complete test set (1500-1999) for evaluation
            if dataset_name in ("math", "gsm8k", "mathqa", "csqa"):
                print(f"HP Evaluation | Using complete original test set (examples 1500-1999)")
                full_data = get_data(
                    dataset_name=dataset_name,
                    train_size=config['train_size'] + config['test_size'],
                    test_size=0,
                    seed=0
                )
                test_examples = full_data[config['train_size']:config['train_size'] + config['test_size']]
                print(f"HP Evaluation | Evaluating on {len(test_examples)} complete test examples (examples 1500-1999)")
            else:
                # For smaller/other datasets (e.g., svamp), use the entire native test split
                print(f"HP Evaluation | Using entire native test split for dataset '{dataset_name}'")
                try:
                    from datasets import load_dataset
                    if dataset_name == 'svamp':
                        raw = load_dataset(path='ChilleD/SVAMP', cache_dir='data/raw')
                        test_split = raw['test']
                        test_examples = [{
                            'question': ex['question_concat'],
                            'answer': str(ex['Answer'])
                        } for ex in test_split]
                    elif dataset_name == 'gpqa':
                        raw = load_dataset(path='Wanfq/gpqa', name='gpqa_main', cache_dir='data/raw')
                        # GPQA has only train; use tail as test
                        all_data = raw['train']
                        used = min(len(all_data), config['train_size'] + config.get('valid_size', config['test_size']))
                        remainder = list(range(used, len(all_data)))
                        # If nothing remains, fall back to a slice of train
                        if not remainder:
                            remainder = list(range(max(0, len(all_data) - config['test_size']), len(all_data)))
                        test_examples = []
                        choice_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
                        for idx in remainder[:config['test_size']]:
                            q = all_data['Question'][idx]
                            choices = [
                                all_data['Correct Answer'][idx],
                                all_data['Incorrect Answer 1'][idx],
                                all_data['Incorrect Answer 2'][idx],
                                all_data['Incorrect Answer 3'][idx]
                            ]
                            # Do not reshuffle at eval time; present as-is
                            question = (
                                f"{q}"
                                f" \n A) {choices[0]}"
                                f" \n B) {choices[1]}"
                                f" \n C) {choices[2]}"
                                f" \n D) {choices[3]}"
                            )
                            answer = choice_map[0]
                            test_examples.append({'question': question, 'answer': answer})
                    elif dataset_name == 'aime2024':
                        # AIME 2024 has only a 'train' split of 30 items; use all as test
                        raw = load_dataset(path='Maxwell-Jia/AIME_2024', cache_dir='data/raw')
                        test_split = raw['train']
                        test_examples = [{
                            'question': ex['Problem'],
                            'answer': str(ex['Answer'])
                        } for ex in test_split]
                    elif dataset_name == 'amc':
                        # AMC-23 has only a 'train' split of ~40 items; use all as test
                        raw = load_dataset(path='knoveleng/AMC-23', cache_dir='data/raw')
                        test_split = raw['train']
                        test_examples = [{
                            'question': ex['problem'],
                            'answer': str(ex['answer'])
                        } for ex in test_split]
                    else:
                        # Fallback: attempt to use HuggingFace dataset with 'test' split
                        raw = load_dataset(path=dataset_name, cache_dir='data/raw')
                        split = 'validation' if 'validation' in raw else 'test'
                        test_examples = []
                        for ex in raw[split]:
                            # Heuristic: expect 'question' and 'answer' fields
                            if 'question' in ex and 'answer' in ex:
                                test_examples.append({'question': ex['question'], 'answer': str(ex['answer'])})
                        if not test_examples:
                            raise RuntimeError(f"Unsupported dataset schema for '{dataset_name}'")
                    print(f"HP Evaluation | Evaluating on {len(test_examples)} native test examples")
                except Exception as e:
                    print(f"HP Evaluation | Failed to load native test split for '{dataset_name}': {e}")
                    # Graceful fallback: evaluate on debate-data test if available
                    with open(debate_data_file, 'r') as f:
                        debate_data = json.load(f)
                    test_examples = []
                    for key, value in debate_data.items():
                        if key != 'metrics' and value.get('split') == 'test':
                            question = value['context'][0][0]['content']
                            test_examples.append({'question': question, 'answer': value['ground_truth']})
                    print(f"HP Evaluation | Fallback evaluating on {len(test_examples)} test examples from debate data")
            
        elif use_test:
            # Use test set from debate data (subset of original test set)
            print(f"HP Evaluation | Using test set from debate data (subset of original test set)")
            with open(debate_data_file, 'r') as f:
                debate_data = json.load(f)
            
            # Extract test examples from debate data
            test_examples = []
            for key, value in debate_data.items():
                if key != 'metrics' and value.get('split') == 'test':
                    # Extract question from context[0][0]['content'] (first message from first agent)
                    question = value['context'][0][0]['content']
                    test_examples.append({
                        'question': question,
                        'answer': value['ground_truth']
                    })
            
            print(f"HP Evaluation | Evaluating on {len(test_examples)} test examples from debate data (WARNING: may be incomplete)")
            
        else:
            # Use validation set (2000-2499) for evaluation
            if dataset_name in ("math", "gsm8k", "mathqa", "csqa"):
                print(f"HP Evaluation | Using separate validation set (examples 2000-2499) for HP tuning")
                full_data = get_data(
                    dataset_name=dataset_name,
                    train_size=config['train_size'] + config['test_size'] + config.get('valid_size', config['test_size']),
                    test_size=0,
                    seed=0
                )
                start = config['train_size'] + config['test_size']
                end = start + config.get('valid_size', config['test_size'])
                val_examples = full_data[start:end]
                print(f"HP Evaluation | Validation pool size: {len(val_examples)} examples (separate from debate data)")
                if total_folds > 1:
                    fold_size = max(1, len(val_examples) // total_folds)
                    start_i = fold_id * fold_size
                    end_i = len(val_examples) if fold_id == total_folds - 1 else (start_i + fold_size)
                    test_examples = val_examples[start_i:end_i]
                    print(f"HP Evaluation | Using CV fold {fold_id+1}/{total_folds}: {len(test_examples)} examples")
                else:
                    test_examples = val_examples
                    print(f"HP Evaluation | Evaluating on {len(test_examples)} validation examples (no CV)")
            else:
                # For small datasets, construct validation from the tail of train; if insufficient, allow repeats
                print(f"HP Evaluation | Building validation set from native train split for dataset '{dataset_name}'")
                try:
                    from datasets import load_dataset
                    if dataset_name == 'svamp':
                        raw = load_dataset(path='ChilleD/SVAMP', cache_dir='data/raw')
                        train_split = raw['train']
                        need = config.get('valid_size', config['test_size'])
                        offset = min(len(train_split), config['train_size'])
                        avail = len(train_split) - offset
                        take = min(need, max(0, avail))
                        val_examples = [{
                            'question': ex['question_concat'],
                            'answer': str(ex['Answer'])
                        } for ex in train_split.select(range(offset, offset + take))]
                        # If not enough, repeat from start of train to fill
                        if len(val_examples) < need:
                            remaining = need - len(val_examples)
                            wrap = [{
                                'question': train_split[i]['question_concat'],
                                'answer': str(train_split[i]['Answer'])
                            } for i in range(min(remaining, len(train_split)))]
                            val_examples.extend(wrap)
                    else:
                        # Fallback generic validation creation (best-effort)
                        raw = load_dataset(path=dataset_name, cache_dir='data/raw')
                        split_name = 'train' if 'train' in raw else list(raw.keys())[0]
                        train_split = raw[split_name]
                        need = config.get('valid_size', config['test_size'])
                        offset = min(len(train_split), config['train_size'])
                        avail = len(train_split) - offset
                        take = min(need, max(0, avail))
                        val_examples = []
                        for idx in range(offset, offset + take):
                            ex = train_split[idx]
                            q = ex.get('question') or ex.get('Problem')
                            a = ex.get('answer') or ex.get('Answer') or ex.get('correct')
                            if q is not None and a is not None:
                                val_examples.append({'question': str(q), 'answer': str(a)})
                        if len(val_examples) < need and len(train_split) > 0:
                            remaining = need - len(val_examples)
                            for i in range(min(remaining, len(train_split))):
                                ex = train_split[i]
                                q = ex.get('question') or ex.get('Problem')
                                a = ex.get('answer') or ex.get('Answer') or ex.get('correct')
                                if q is not None and a is not None:
                                    val_examples.append({'question': str(q), 'answer': str(a)})
                    # CV split if requested
                    if total_folds > 1:
                        fold_size = max(1, len(val_examples) // total_folds)
                        start_i = fold_id * fold_size
                        end_i = len(val_examples) if fold_id == total_folds - 1 else (start_i + fold_size)
                        test_examples = val_examples[start_i:end_i]
                        print(f"HP Evaluation | Using CV fold {fold_id+1}/{total_folds}: {len(test_examples)} examples")
                    else:
                        test_examples = val_examples
                        print(f"HP Evaluation | Evaluating on {len(test_examples)} validation examples (no CV)")
                except Exception as e:
                    print(f"HP Evaluation | Failed to build validation set for '{dataset_name}': {e}")
                    test_examples = []
        
        # Evaluation metrics
        correct = 0
        total = 0
        results = []
        
        # Process in batches
        for i in range(0, len(test_examples), batch_size):
            batch = test_examples[i:i + batch_size]
            print(f"HP Evaluation | Processing batch {i//batch_size + 1}/{(len(test_examples) + batch_size - 1)//batch_size}")
            
            # Prepare all contexts for this batch
            batch_contexts = []
            batch_questions = []
            batch_ground_truths = []
            
            for j, example in enumerate(batch):
                # Construct prompt using same format as debate
                question = example['question']
                prompt = construct_question_prompt(question, dataset_name)
                
                # Create context
                context = [{"role": "user", "content": prompt}]
                batch_contexts.append(context)
                batch_questions.append(question)
                batch_ground_truths.append(example['answer'])
            
            try:
                print(f"HP Evaluation | Generating responses for batch of {len(batch_contexts)} examples...")
                # Use batch_generate for efficiency
                completions = asyncio.run(agent.batch_generate(
                    contexts_list=batch_contexts,
                    device=device,
                    temperature=config.get('temperature', 1.0),
                    top_p=config.get('top_p', 0.9)
                ))
                print(f"HP Evaluation | Generated {len(completions)} responses")
                
                # Process each completion
                for j, completion in enumerate(completions):
                    example_idx = i + j
                    response = completion["choices"][0]["message"]["content"]
                    question = batch_questions[j]
                    ground_truth = batch_ground_truths[j]
                    
                    # Parse and grade answer
                    parsed_answer = parse_answer(response, dataset_name)
                    if dataset_name in ("gpqa", "mathqa", "csqa"):
                        # Multiple-choice: compare case-insensitively on letters
                        is_correct = (parsed_answer or "").strip().upper() == (ground_truth or "").strip().upper()
                    else:
                        # Numeric/text: robust grading
                        is_correct = grade_answer(parsed_answer, ground_truth)
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    results.append({
                        'question': question,
                        'response': response,
                        'parsed_answer': parsed_answer,
                        'ground_truth': ground_truth,
                        'correct': is_correct
                    })
                    
                print(f"HP Evaluation | Batch {i//batch_size + 1} completed: {sum(1 for r in results[-len(completions):] if r['correct'])}/{len(completions)} correct")
                    
            except Exception as e:
                print(f"HP Evaluation | Error processing batch: {e}")
                # Handle batch errors by processing individually
                for j, example in enumerate(batch):
                    example_idx = i + j
                    question = example['question']
                    ground_truth = example['answer']
                    
                    try:
                        prompt = construct_question_prompt(question, dataset_name)
                        context = [{"role": "user", "content": prompt}]
                        
                        response = asyncio.run(agent.generate(
                            context=context,
                            device=device,
                            temperature=config.get('temperature', 1.0),
                            top_p=config.get('top_p', 0.9)
                        ))
                        
                        parsed_answer = parse_answer(response, dataset_name)
                        if dataset_name in ("gpqa", "mathqa", "csqa"):
                            is_correct = (parsed_answer or "").strip().upper() == (ground_truth or "").strip().upper()
                        else:
                            is_correct = grade_answer(parsed_answer, ground_truth)
                        if is_correct:
                            correct += 1
                        total += 1
                        
                        results.append({
                            'question': question,
                            'response': response,
                            'parsed_answer': parsed_answer,
                            'ground_truth': ground_truth,
                            'correct': is_correct
                        })
                        
                    except Exception as e2:
                        print(f"HP Evaluation | Error processing example {example_idx + 1}: {e2}")
                        results.append({
                            'question': question,
                            'response': 'ERROR',
                            'parsed_answer': '',
                            'ground_truth': ground_truth,
                            'correct': False
                        })
                        total += 1
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0.0
        metrics = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results
        }
        
        print(f"HP Evaluation | Final accuracy: {accuracy:.4f} ({correct}/{total})")
        
        # Save evaluation results (flat in experiment_dir). Primary dataset → evaluation_results.json; overrides → evaluation_results_<dataset>.json
        eval_results_file = os.path.join(
            experiment_dir,
            f"evaluation_results_{results_suffix}.json" if results_suffix else "evaluation_results.json"
        )
        with open(eval_results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"HP Evaluation | Results saved to {eval_results_file}")
        
        # Cleanup
        agent.cleanup()
        
        return True, metrics
        
    except Exception as e:
        print(f"HP Evaluation | Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def main():
    """Main function for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for single agent training")
    
    # Required arguments
    parser.add_argument('--debate_file', help="Path to 3-agent debate data JSON file (auto-constructed if not provided)")
    parser.add_argument('--output_dir', required=True, help="Output directory for HP runs")
    
    # Training configuration
    parser.add_argument('--agent_idx', type=int, default=0, help="Agent index to train (default: 0)")
    parser.add_argument('--phase', choices=['sft', 'rl', 'kto', 'dpo'], default='rl', 
                       help="Training phase (default: rl)")
    parser.add_argument('--iter_idx', type=int, default=0, help="Iteration index (default: 0)")
    parser.add_argument('--max_trials', type=int, default=None, help="Maximum number of trials to run")
    parser.add_argument('--gpu_id', type=int, default=None, help="Specific GPU ID to use (default: auto-select)")
    parser.add_argument('--exclude_gpus', type=int, nargs='*', default=[], help="GPU IDs to exclude")
    parser.add_argument('--rewards_file', help="Path to precomputed rewards file (RL only)")
    
    # Base configuration (all the standard args from args.py)
    parser.add_argument('--model', default='phi4b', help="Base model to use")
    parser.add_argument('--dataset', default='gsm8k', help="Dataset to use")
    parser.add_argument('--train_datasets', nargs='*', default=None,
                        help="Optional list of datasets to mix for training; overrides --dataset for training if provided")
    parser.add_argument('--test_datasets', nargs='*', default=None,
                        help="Optional list of datasets to evaluate on (each evaluated separately)")
    parser.add_argument('--train_size', type=int, default=1500, help="Number of training examples")
    parser.add_argument('--test_size', type=int, default=500, help="Number of test examples")
    parser.add_argument('--valid_size', type=int, default=500, help="Number of validation examples (when using validation)")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--data_seed', type=int, default=0, help="Data seed (default: 100 for HP tuning to avoid test set overlap)")
    parser.add_argument('--use_quantization', action='store_true', help="Enable quantization")
    parser.add_argument('--use_adapter_mode', action='store_true', help="Use adapter mode")
    parser.add_argument('--use_scheduler', action='store_true', help="Use scheduler")
    parser.add_argument('--max_concurrent_tasks', type=int, default=6, help="Max concurrent tasks")
    parser.add_argument('--agents', type=int, default=3, help="Number of agents")
    parser.add_argument('--rounds', type=int, default=2, help="Number of debate rounds")
    parser.add_argument('--batch_debate', type=int, default=8, help="Debate batch size")
    parser.add_argument('--temperature', type=float, default=1.0, help="Sampling temperature")
    parser.add_argument('--top_p', type=float, default=0.9, help="Top-p sampling")
    parser.add_argument('--use_vllm', action='store_true', help="Use vLLM")
    parser.add_argument('--use_async_debate', action='store_true', help="Use async debate")
    parser.add_argument('--summarize', action='store_true', help="Enable summarization")
    parser.add_argument('--diversity_prompt', action='store_true', help="Use diversity prompt")
    parser.add_argument('--use_majority_vote', action='store_true', help="Use majority vote")
    parser.add_argument('--use_ground_truth', action='store_true', help="Use ground truth answers instead of majority vote (overrides --use_majority_vote)")
    parser.add_argument('--include_debate_context', action='store_true', help="Include debate context")
    parser.add_argument('--pref_debate_context', action='store_true', help="Use debate context in preference data")
    parser.add_argument('--test_only', action='store_true', help="Test only")
    parser.add_argument('--train_only', action='store_true', help="Train only")
    parser.add_argument('--finetune', action='store_true', help="Enable SFT")
    parser.add_argument('--post_train', action='store_true', help="Enable RL training")
    parser.add_argument('--kto', action='store_true', help="Enable KTO")
    parser.add_argument('--dpo', action='store_true', help="Enable DPO")
    
    # SFT parameters
    parser.add_argument('--epoch_sft', type=int, default=1, help="SFT epochs")
    parser.add_argument('--batch_size_sft', type=int, default=1, help="SFT batch size")
    parser.add_argument('--lr_sft', type=float, default=1e-5, help="SFT learning rate")
    parser.add_argument('--weight_decay_sft', type=float, default=1e-2, help="SFT weight decay")
    
    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=8, help="LoRA r")
    parser.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha")
    parser.add_argument('--lora_dropout', type=float, default=0.1, help="LoRA dropout")
    parser.add_argument('--lora_target_modules', nargs='*', default=["q_proj", "k_proj", "v_proj", "o_proj"], help="LoRA target modules")
    
    # RL parameters
    parser.add_argument('--epoch_rl', type=int, default=1, help="RL epochs")
    parser.add_argument('--batch_rl', type=int, default=8, help="RL batch size")
    parser.add_argument('--lr_rl', type=float, default=1e-5, help="RL learning rate")
    parser.add_argument('--weight_decay_rl', type=float, default=1e-2, help="RL weight decay")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument('--num_generations', type=int, default=8, help="Number of generations")
    parser.add_argument('--consensus_batch_size', type=int, default=8, help="Consensus batch size")
    parser.add_argument('--entropy_coef', type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument('--consensus_weight', type=float, default=0.7, help="Consensus weight")
    
    # Judge model parameters
    parser.add_argument('--judge_model', default='qwen2b', help="Judge model")
    parser.add_argument('--judge_batch_size', type=int, default=4, help="Judge batch size")
    
    # Logging parameters
    parser.add_argument('--wandb', action='store_true', help="Enable Weights & Biases logging")
    parser.add_argument('--project_name', default='llm-marl', help="Project name")
    parser.add_argument('--entity_name', default='llm-marl', help="Entity name")
    parser.add_argument('--tmp', action='store_true', help="Temporary mode")
    parser.add_argument('--tmp_kaveh', action='store_true', help="Temporary Kaveh mode")
    
    # Transfer function parameters
    parser.add_argument('--use_consensus_reward', action='store_true', help="Use consensus reward")
    
    # KTO parameters
    parser.add_argument('--epoch_kto', type=int, default=3, help="KTO epochs")
    parser.add_argument('--batch_kto', type=int, default=8, help="KTO batch size")
    parser.add_argument('--lr_kto', type=float, default=1e-5, help="KTO learning rate")
    parser.add_argument('--beta_kto', type=float, default=0.1, help="KTO beta")
    parser.add_argument('--gradient_accumulation_steps_kto', type=int, default=4, help="KTO gradient accumulation steps")
    parser.add_argument('--train_size_kto', type=int, default=-1, help="KTO train size")
    parser.add_argument('--desirable_weight', type=float, default=1.0, help="KTO desirable weight")
    parser.add_argument('--undesirable_weight', type=float, default=1.0, help="KTO undesirable weight")
    parser.add_argument('--kto_weight_scale', type=float, default=1.0, help="Scaling factor for KTO weights")
    parser.add_argument('--flip_kto_weights', action='store_true', help="Flip desirable/undesirable KTO weights (risk-seeking)")
    parser.add_argument('--disable_kto_weight_cap', action='store_true', help="Disable the 4/3 cap on KTO weight ratios")
    
    # Evaluation parameters
    parser.add_argument('--evaluate_after_training', action='store_true', default=True,
                       help="After training, load the checkpoint and evaluate on test set using debate parsing (default: True)")
    parser.add_argument('--evaluation_batch_size', type=int, default=8, 
                       help="Batch size for evaluation generation")
    parser.add_argument('--evaluation_timeout', type=int, default=7200, 
                       help="Timeout for evaluation in seconds (default: 2 hours)")
    parser.add_argument('--cv_folds', type=int, default=1,
                        help="If >1, run K-fold cross-validation over the evaluation split and report averaged accuracy")
    parser.add_argument('--skip_training', action='store_true',
                       help="Skip training and just run evaluation on base model")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help="Path to custom checkpoint to load (overrides default checkpoint location)")
    
    # Context control parameters
    parser.add_argument('--no_context', action='store_true',
                       help="Disable debate context (sets include_debate_context=False)")
    
    # Evaluation data source parameters
    parser.add_argument('--use_test', action='store_true',
                       help="Use test set from debate data for evaluation instead of separate validation set")
    parser.add_argument('--use_full_test', action='store_true',
                       help="Use the complete original test set (examples 1500-1999) for evaluation")
    
    # Training data configuration parameters
    parser.add_argument('--use_initial_responses', action='store_true',
                       help="Use initial round responses instead of final round responses for post-training (regular majority vote instead of post-debate majority vote)")
    
    # Training timeout parameters
    parser.add_argument('--training_timeout', type=int, default=-1, 
                       help="Timeout for training in seconds (default: -1 = no timeout)")
    
    # DPO parameters
    parser.add_argument('--epoch_dpo', type=int, default=3, help="DPO epochs")
    parser.add_argument('--batch_dpo', type=int, default=8, help="DPO batch size")
    parser.add_argument('--lr_dpo', type=float, default=5e-6, help="DPO learning rate")
    parser.add_argument('--beta_dpo', type=float, default=0.1, help="DPO beta")
    parser.add_argument('--gradient_accumulation_steps_dpo', type=int, default=4, help="DPO gradient accumulation steps")
    parser.add_argument('--train_size_dpo', type=int, default=-1, help="DPO train size")
    
    # Reward model parameters
    parser.add_argument('--verifiable_reward', action='store_true', help="Use verifiable reward")
    parser.add_argument('--use_format_reward', action='store_true', help="Use format reward")
    
    # GPU parameters
    parser.add_argument('--max_agents_per_device', type=int, default=2, help="Max agents per device")
    parser.add_argument('--gpus_per_model', type=int, default=1, help="GPUs per model")
    parser.add_argument('--gpu_allocation_timeout', type=float, default=60.0, help="GPU allocation timeout")
    parser.add_argument('--use_parallel_training', action='store_true', help="Use parallel training")
    
    # Hyperparameter ranges (these will be used to generate combinations)
    # SFT hyperparameters
    parser.add_argument('--lr_sft_range', type=float, nargs='*', help="SFT learning rate range")
    parser.add_argument('--batch_size_sft_range', type=int, nargs='*', help="SFT batch size range")
    parser.add_argument('--epoch_sft_range', type=int, nargs='*', help="SFT epoch range")
    parser.add_argument('--weight_decay_sft_range', type=float, nargs='*', help="SFT weight decay range")
    
    # RL hyperparameters
    parser.add_argument('--lr_rl_range', type=float, nargs='*', help="RL learning rate range")
    parser.add_argument('--batch_rl_range', type=int, nargs='*', help="RL batch size range")
    parser.add_argument('--epoch_rl_range', type=int, nargs='*', help="RL epoch range")
    parser.add_argument('--lora_r_range', type=int, nargs='*', help="LoRA r range")
    parser.add_argument('--lora_alpha_range', type=int, nargs='*', help="LoRA alpha range")
    parser.add_argument('--num_generations_range', type=int, nargs='*', help="Number of generations range")
    parser.add_argument('--consensus_weight_range', type=float, nargs='*', help="Consensus weight range")
    parser.add_argument('--gradient_accumulation_steps_range', type=int, nargs='*', help="Gradient accumulation steps range")
    
    # KTO hyperparameters
    parser.add_argument('--lr_kto_range', type=float, nargs='*', help="KTO learning rate range")
    parser.add_argument('--beta_kto_range', type=float, nargs='*', help="KTO beta range")
    parser.add_argument('--batch_kto_range', type=int, nargs='*', help="KTO batch size range")
    parser.add_argument('--epoch_kto_range', type=int, nargs='*', help="KTO epoch range")
    parser.add_argument('--gradient_accumulation_steps_kto_range', type=int, nargs='*', help="KTO gradient accumulation steps range")
    parser.add_argument('--desirable_weight_range', type=float, nargs='*', help="KTO desirable weight range")
    parser.add_argument('--undesirable_weight_range', type=float, nargs='*', help="KTO undesirable weight range")
    parser.add_argument('--kto_weight_scale_range', type=float, nargs='*', help="KTO weight scaling factor range")
    
    # DPO hyperparameters
    parser.add_argument('--lr_dpo_range', type=float, nargs='*', help="DPO learning rate range")
    parser.add_argument('--beta_dpo_range', type=float, nargs='*', help="DPO beta range")
    parser.add_argument('--batch_dpo_range', type=int, nargs='*', help="DPO batch size range")
    parser.add_argument('--epoch_dpo_range', type=int, nargs='*', help="DPO epoch range")
    parser.add_argument('--gradient_accumulation_steps_dpo_range', type=int, nargs='*', help="DPO gradient accumulation steps range")
    
    
    args = parser.parse_args()
    
    # Handle use_ground_truth override
    if args.use_ground_truth:
        args.use_majority_vote = False
        print("Note: --use_ground_truth flag detected, setting use_majority_vote=False")
    
    # Handle no_context override
    if args.no_context:
        args.include_debate_context = False
        print("Note: --no_context flag detected, setting include_debate_context=False")

    # Handle wandb flag override
    if not args.wandb:
        args.entity_name = "test-entity"
        print("Note: --wandb flag not set, disabling wandb logging")

    # Resolve training datasets to debate files (supports single and multi)
    train_datasets = args.train_datasets if args.train_datasets else [args.dataset]
    train_dataset_to_file: Dict[str, str] = {}
    for dset in train_datasets:
        path = resolve_debate_file_for_dataset(args, dset)
        if not path:
            expected_new = build_debate_path_new(
                model=args.model,
                dataset=dset,
                train_size=args.train_size,
                test_size=args.test_size,
                agents=args.agents,
                rounds=args.rounds,
                data_seed=args.data_seed,
                diversity_prompt=args.diversity_prompt,
                summarize=args.summarize,
            )
            expected_old = build_debate_path(
                model=args.model,
                dataset=dset,
                agents=args.agents,
                rounds=args.rounds,
                seed=args.data_seed,
                diversity_prompt=args.diversity_prompt,
                summarize=args.summarize,
            )
            print(f"Error: Debate file not found for dataset '{dset}'.")
            print(f"HP Tuning | Checked (new format): {expected_new}")
            print(f"HP Tuning | Checked (old format): {expected_old}")
            return 1
        train_dataset_to_file[dset] = path
    if len(train_dataset_to_file) == 1:
        args.debate_file = next(iter(train_dataset_to_file.values()))
        print(f"HP Tuning | Using debate file: {args.debate_file}")
    else:
        print(f"HP Tuning | Using multiple training datasets: {list(train_dataset_to_file.keys())}")
    
    # Load/merge debate data for training
    if len(train_dataset_to_file) == 1:
        debate_data = load_debate_data(args.debate_file)
    else:
        debate_data = merge_balanced_training_sets(train_dataset_to_file, seed=args.data_seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save debate data to output directory
    debate_data_file = output_dir / ("debate_data.json" if len(train_dataset_to_file) == 1 else "debate_data_merged.json")
    with open(debate_data_file, 'w') as f:
        json.dump(debate_data, f, indent=2)
    if len(train_dataset_to_file) > 1:
        with open(output_dir / "training_sources.json", 'w') as f:
            json.dump(train_dataset_to_file, f, indent=2)

    # Resolve evaluation datasets (optional multi). We'll use them later for per-dataset tests.
    test_datasets = args.test_datasets if args.test_datasets else [args.dataset]
    
    # Create base config from args
    base_config = vars(args)
    
    # Add HP tuning specific overrides and only set defaults for missing values
    base_config.update({
        # Disable checkpoint saving for HP tuning
        'save_strategy': 'no',
        'save_steps': None,
        'save_total_limit': 0,
    })
    
    # Only set defaults for values that weren't provided via command line
    defaults_to_set = {
        'use_majority_vote': True,
        'verifiable_reward': True,
        'use_format_reward': True,
        # Note: include_debate_context and pref_debate_context are handled dynamically based on --no_context flag
        'summarize': False,
        'finetune': False,
        'post_train': False,
        'kto': False,
        'dpo': False,
        'peft': False,
        'use_vllm': False,
        'use_async_debate': True,
        'use_scheduler': True,
        'use_adapter_mode': True,
        'use_quantization': True,
        'use_parallel_training': True,
        'use_consensus_reward': True,
        'test_only': False,
        'train_only': False,
        'gpus_per_model': 1,
        'max_concurrent_tasks': 6,
        'agents': 3,
        'batch_debate': 24,
        'train_size': 1500,
        'test_size': 500,
        'iterations': 1,
        'batch_rl': 8,
        'consensus_batch_size': 8,
        'num_generations': 8,
        'lora_r': 32,
        'lora_alpha': 32,
        'epoch_rl': 1,
        'epoch_kto': 3,
        'lr_kto': 1e-5,
        'epoch_sft': 1,
        'entity_name': 'llm-marl',
        'batch_kto': 8,
        'gradient_accumulation_steps_kto': 4,
        'beta_kto': 0.1,
        'gradient_accumulation_steps': 4,
        'train_size_kto': -1,
        'desirable_weight': 1.0,
        'undesirable_weight': 1.0,
        'epoch_dpo': 3,
        'batch_dpo': 8,
        'lr_dpo': 5e-6,
        'beta_dpo': 0.1,
        'gradient_accumulation_steps_dpo': 4,
    }
    
    # Only set defaults for values that weren't provided via command line
    for key, default_value in defaults_to_set.items():
        if key not in base_config:
            base_config[key] = default_value
        elif base_config[key] is None:
            base_config[key] = default_value
        # For boolean flags, if they're False but should default to True, set them to True
        elif isinstance(default_value, bool) and isinstance(base_config[key], bool):
            if default_value and not base_config[key]:
                base_config[key] = default_value
    
    # Collect hyperparameter ranges
    hp_ranges = {}
    
    # RL hyperparameters
    if args.lr_rl_range:
        hp_ranges['lr_rl'] = args.lr_rl_range
    if args.batch_rl_range:
        hp_ranges['batch_rl'] = args.batch_rl_range
    if args.epoch_rl_range:
        hp_ranges['epoch_rl'] = args.epoch_rl_range
    if args.lora_r_range:
        hp_ranges['lora_r'] = args.lora_r_range
    if args.lora_alpha_range:
        hp_ranges['lora_alpha'] = args.lora_alpha_range
    if args.num_generations_range:
        hp_ranges['num_generations'] = args.num_generations_range
    if args.consensus_weight_range:
        hp_ranges['consensus_weight'] = args.consensus_weight_range
    if args.gradient_accumulation_steps_range:
        hp_ranges['gradient_accumulation_steps'] = args.gradient_accumulation_steps_range
    
    # KTO hyperparameters
    if args.lr_kto_range:
        hp_ranges['lr_kto'] = args.lr_kto_range
    if args.beta_kto_range:
        hp_ranges['beta_kto'] = args.beta_kto_range
    if args.batch_kto_range:
        hp_ranges['batch_kto'] = args.batch_kto_range
    if args.epoch_kto_range:
        hp_ranges['epoch_kto'] = args.epoch_kto_range
    if args.gradient_accumulation_steps_kto_range:
        hp_ranges['gradient_accumulation_steps_kto'] = args.gradient_accumulation_steps_kto_range
    if args.desirable_weight_range:
        hp_ranges['desirable_weight'] = args.desirable_weight_range
    if args.undesirable_weight_range:
        hp_ranges['undesirable_weight'] = args.undesirable_weight_range
    
    # DPO hyperparameters
    if args.lr_dpo_range:
        hp_ranges['lr_dpo'] = args.lr_dpo_range
    if args.beta_dpo_range:
        hp_ranges['beta_dpo'] = args.beta_dpo_range
    if args.batch_dpo_range:
        hp_ranges['batch_dpo'] = args.batch_dpo_range
    if args.epoch_dpo_range:
        hp_ranges['epoch_dpo'] = args.epoch_dpo_range
    if args.gradient_accumulation_steps_dpo_range:
        hp_ranges['gradient_accumulation_steps_dpo'] = args.gradient_accumulation_steps_dpo_range
    
    if not hp_ranges:
        # SINGLE TRIAL MODE: No HP ranges = single evaluation/training run
        print("HP Tuning | No hyperparameter ranges specified. Running single trial with provided configuration.")
        hp_configs = [base_config.copy()]
        # Add trial metadata
        hp_configs[0]['hp_run_id'] = 0
        hp_configs[0]['hp_combination'] = {}
        # Extract any hyperparameters that were explicitly set
        print(f"HP Tuning | Debug: base_config keys: {list(base_config.keys())}")
        for key, value in base_config.items():
            if key in ['lora_r', 'lora_alpha', 'lr_kto', 'beta_kto', 'batch_kto', 'epoch_kto', 
                      'lr_rl', 'batch_rl', 'epoch_rl', 'lr_dpo', 'beta_dpo', 'batch_dpo', 'epoch_dpo']:
                hp_configs[0]['hp_combination'][key] = value
                print(f"HP Tuning | Added {key}={value} to hp_combination")
        print(f"HP Tuning | Final hp_combination: {hp_configs[0]['hp_combination']}")
    else:
        # Create HP configurations from ranges
        hp_configs = create_hp_configs(base_config, hp_ranges)
    
    # Limit trials if specified
    if args.max_trials:
        hp_configs = hp_configs[:args.max_trials]
        print(f"HP Tuning | Limited to {len(hp_configs)} trials")
    
    # Determine GPU to use
    if args.gpu_id is not None:
        gpu_id = args.gpu_id
        print(f"HP Tuning | Using specified GPU {gpu_id}")
    else:
        available_gpus = get_available_gpus(args.exclude_gpus)
        if not available_gpus:
            print("Error: No available GPUs found")
            return 1
        gpu_id = available_gpus[0]
        print(f"HP Tuning | Auto-selected GPU {gpu_id}")
    
    # Initialize WandB (augment config for visibility)
    if len(train_datasets) > 1:
        base_config['train_datasets'] = train_datasets
        base_config['dataset_label'] = "+".join(train_datasets)
    else:
        base_config['train_datasets'] = train_datasets
        base_config['dataset_label'] = base_config['dataset']
    base_config['test_datasets'] = test_datasets
    if (args.test_datasets and not args.use_test and not args.use_full_test):
        # default to full test if multiple datasets were provided without source
        base_config['use_full_test'] = True
    wandb_run = setup_wandb_hp_tuning(base_config, hp_ranges, args.phase)
    
    # Run trials
    print(f"HP Tuning | Starting {len(hp_configs)} hyperparameter trials...")
    
    results = []
    successful_trials = 0
    
    for i, config in enumerate(hp_configs):
        print(f"\nHP Tuning | Trial {i+1}/{len(hp_configs)}")
        
        # Determine context settings based on --no_context flag BEFORE creating folder name
        if args.no_context:
            # If --no_context is set, disable both context flags
            config['include_debate_context'] = False
            config['pref_debate_context'] = False
            print(f"HP Tuning | Trial {config['hp_run_id']} | No context: include_debate_context=False, pref_debate_context=False")
        else:
            # Default behavior: include_debate_context=True, pref_debate_context=False
            # But allow override via command line flags
            config['include_debate_context'] = True  # Default to True
            
            if args.pref_debate_context:
                # When pref_debate_context is True, include_debate_context should also be True
                config['pref_debate_context'] = True
                config['include_debate_context'] = True
            else:
                config['pref_debate_context'] = False  # Default to False
            
            print(f"HP Tuning | Trial {config['hp_run_id']} | Context: include_debate_context={config['include_debate_context']}, pref_debate_context={config['pref_debate_context']}")
        
        # Create experiment directory for this trial with descriptive name (AFTER setting context flags)
        trial_folder_name = create_trial_folder_name(config, config['hp_combination'])
        trial_dir = output_dir / trial_folder_name if trial_folder_name else output_dir
        trial_dir.mkdir(exist_ok=True)
        
        # Update config with trial-specific paths
        config['experiment_dir'] = str(trial_dir)
        
        # Add custom checkpoint path if provided
        if args.checkpoint_path:
            config['checkpoint_path'] = args.checkpoint_path
        
        # Add evaluation mode flags
        config['use_test'] = args.use_test
        config['use_full_test'] = args.use_full_test
        config['train_datasets'] = base_config.get('train_datasets', [config['dataset']])
        config['test_datasets'] = base_config.get('test_datasets', [config['dataset']])
        config['dataset_label'] = base_config.get('dataset_label', config['dataset'])
        
        # Run trial (optionally with CV loops)
        if args.cv_folds and args.cv_folds > 1 and args.evaluate_after_training:
            print(f"HP Tuning | Running {args.cv_folds}-fold CV for evaluation")
            fold_metrics = []
            fold_success = True
            total_runtime = 0.0
            for fold_id in range(args.cv_folds):
                # For CV we evaluate after training on per-fold slice
                s, rtime = run_single_hp_trial(
                    agent_idx=args.agent_idx,
                    gpu_id=gpu_id,
                    phase=args.phase,
                    iter_idx=args.iter_idx,
                    experiment_dir=str(trial_dir),
                    debate_data_file=str(debate_data_file),
                    config={**config, 'cv_fold_id': fold_id, 'cv_total_folds': args.cv_folds},
                    rewards_file=args.rewards_file,
                    evaluate_after_training=True,
                    evaluation_batch_size=args.evaluation_batch_size,
                    evaluation_timeout=args.evaluation_timeout,
                    training_timeout=args.training_timeout,
                    skip_training=args.skip_training
                )
                fold_success = fold_success and s
                total_runtime += rtime

                # Load evaluation and capture accuracy per fold
                eval_results_file = os.path.join(trial_dir, "evaluation_results.json")
                try:
                    if os.path.exists(eval_results_file):
                        with open(eval_results_file, 'r') as f:
                            m = json.load(f)
                            fold_metrics.append(m.get('accuracy', 0.0))
                except Exception:
                    pass

            # Aggregate CV
            avg_acc = float(sum(fold_metrics) / len(fold_metrics)) if fold_metrics else 0.0
            print(f"HP Tuning | CV average accuracy over {len(fold_metrics)} folds: {avg_acc:.4f}")

            # Save CV summary
            cv_summary_path = os.path.join(trial_dir, "evaluation_results_cv.json")
            with open(cv_summary_path, 'w') as f:
                json.dump({
                    'cv_folds': args.cv_folds,
                    'fold_accuracies': fold_metrics,
                    'avg_accuracy': avg_acc
                }, f, indent=2)

            success, runtime = fold_success, total_runtime
        else:
            success, runtime = run_single_hp_trial(
                agent_idx=args.agent_idx,
                gpu_id=gpu_id,
                phase=args.phase,
                iter_idx=args.iter_idx,
                experiment_dir=str(trial_dir),
                debate_data_file=str(debate_data_file),
                config=config,
                rewards_file=args.rewards_file,
                evaluate_after_training=args.evaluate_after_training,
                evaluation_batch_size=args.evaluation_batch_size,
                evaluation_timeout=args.evaluation_timeout,
                training_timeout=args.training_timeout,
                skip_training=(args.skip_training or args.test_only)
            )
        
        # Record result
        result = {
            'trial_id': config['hp_run_id'],
            'trial_folder': trial_folder_name or '.',
            'hyperparameters': config['hp_combination'],
            'success': success,
            'experiment_dir': str(trial_dir),
            'runtime_seconds': runtime
        }
        
        # Add evaluation results if available (single file)
        eval_results_file = os.path.join(trial_dir, "evaluation_results.json")
        if os.path.exists(eval_results_file):
            try:
                with open(eval_results_file, 'r') as f:
                    eval_metrics = json.load(f)
                result['evaluation'] = {
                    'accuracy': eval_metrics.get('accuracy', 0.0),
                    'correct': eval_metrics.get('correct', 0),
                    'total': eval_metrics.get('total', 0)
                }
            except Exception as e:
                print(f"HP Tuning | Failed to load evaluation results for trial {config['hp_run_id']}: {e}")
        # Load any per-dataset evaluation files if present
        try:
            import glob as _glob
            per_files = _glob.glob(os.path.join(trial_dir, "evaluation_results_*.json"))
            if per_files:
                result['per_dataset_evaluation'] = {}
                for pth in per_files:
                    label = os.path.basename(pth)[len("evaluation_results_"):-5]
                    with open(pth, 'r') as f:
                        m = json.load(f)
                    result['per_dataset_evaluation'][label] = {
                        'accuracy': m.get('accuracy', 0.0),
                        'correct': m.get('correct', 0),
                        'total': m.get('total', 0)
                    }
        except Exception as e:
            print(f"HP Tuning | Failed to load per-dataset evaluation: {e}")
        
        results.append(result)
        
        if success:
            successful_trials += 1
        
        print(f"HP Tuning | Trial {config['hp_run_id']} {'SUCCESS' if success else 'FAILED'}")
        
        # Log trial result to WandB
        log_trial_result(wandb_run, config['hp_run_id'], config['hp_combination'], success, str(trial_dir), runtime)
    
    # Save results summary
    results_file = output_dir / "hp_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'total_trials': len(hp_configs),
            'successful_trials': successful_trials,
            'failed_trials': len(hp_configs) - successful_trials,
            'results': results
        }, f, indent=2)
    
    # Log final summary to WandB
    wandb_run.log({
        'hp_tuning_complete': True,
        'total_trials': len(hp_configs),
        'successful_trials': successful_trials,
        'failed_trials': len(hp_configs) - successful_trials,
        'success_rate': successful_trials / len(hp_configs) if hp_configs else 0
    })
    
    # Finish WandB run
    wandb_run.finish()
    
    print(f"\nHP Tuning | Completed {len(hp_configs)} trials")
    print(f"HP Tuning | Successful: {successful_trials}, Failed: {len(hp_configs) - successful_trials}")
    print(f"HP Tuning | Results saved to {results_file}")
    print(f"HP Tuning | WandB run: {wandb_run.url}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)