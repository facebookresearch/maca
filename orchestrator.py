"""Multi-agent debate orchestrator for coordinating training and evaluation."""

import os
import torch
import wandb
import shutil
import json
import torch.multiprocessing as mp
import asyncio
import queue
import threading
import time
from typing import List, Dict, Any, Union, Optional, Tuple, Set, Callable
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from model import Agent
from debate import distributed_multiagent_debate, scheduled_multiagent_debate
from data import get_data, prepare_finetune_data, prepare_post_train_data, prepare_kto_data, prepare_dpo_data
from utils import (
    setup_environment, 
    setup_experiment_dir,
    get_available_devices,
    cleanup_gpu,
    set_random_seed
)
from analyze_experiment_performance import (
    analyze_experiment_performance, 
    print_performance_comparison, 
    generate_summary_plots,
    create_structured_experiment_summary
)

from utils import pd

class MultiAgentOrchestrator:
    """Orchestrator for multi-agent reinforcement learning training and debate evaluation."""
    
    def __init__(
        self,
        config: Dict[str, Any]
    ) -> None:
        """Initialize the orchestrator with configuration and setup environment."""
        self.config = config
        
        # Setup environment and experiment directory
        setup_environment()
        if not (self.config['experiment_dir'] != '' and os.path.exists(self.config['experiment_dir'])):
            experiment_dir = setup_experiment_dir(config)
            self.config['experiment_dir'] = experiment_dir

        # Initialize GPU resource manager
        exclude_gpus = self.config.get('exclude_gpus', [])
        max_agents_per_gpu = self.config.get('max_agents_per_device', 1)
        self.gpu_manager = GPUResourceManager(
            exclude_gpus=exclude_gpus,
            max_concurrent_per_gpu=max_agents_per_gpu
        )
        
        # Update config with detected GPUs
        detected_gpu_ids = [gpu['id'] for gpu in self.gpu_manager.available_gpus]
        self.config['devices'] = [f'cuda:{gpu_id}' for gpu_id in detected_gpu_ids]
        
        # Get available devices (legacy compatibility)
        devices = get_available_devices()
        if not self.config['devices']:  # Fallback if GPU manager found no GPUs
            self.config['devices'] = devices
            
        # Setup wandb
        self._setup_wandb(
            project_name=self.config.get('project_name', "llm-marl"),
            entity_name=self.config.get('entity_name', "llm-marl")
        )
        
        print(f"\n=========== Experiment {self.config['experiment_dir'].split('/')[-1]} ===========", flush=True)
        print(self.config, flush=True)
        
        
        # Load or initialize dataset
        self.data = get_data(
            self.config['dataset'], 
            train_size=self.config['train_size'], 
            test_size=self.config['test_size'], 
            seed=self.config['data_seed']
        )

    def _setup_wandb(
        self,
        project_name: str = "llm-marl",
        entity_name: str = "llm-marl"
    ) -> None:
        """Initialize Weights & Biases logging for experiment tracking."""
        # Skip wandb for test cases
        if project_name == 'test-project' or entity_name == 'test-entity':
            print("Skipping wandb initialization for test case")
            return
            
        # ------------------------------------------------------------------
        # Use a persistent experiment-wide group ID so *all* orchestrator and
        # agent subprocess runs land in the same umbrella in WandB.
        # The first process that starts writes the file; subsequent ones read.
        # ------------------------------------------------------------------
        group_file = os.path.join(self.config['experiment_dir'], ".wandb_group")
        if os.path.exists(group_file):
            with open(group_file) as _gf:
                exp_group = _gf.read().strip()
        else:
            exp_group = os.path.basename(self.config['experiment_dir'].rstrip('/'))
            try:
                with open(group_file, "w") as _gf:
                    _gf.write(exp_group)
            except Exception as _e:
                print(f"Warning: could not write group file {group_file}: {_e}")
        
        wandb.init(
            project=project_name,
            entity=entity_name,
            dir=f"{self.config['experiment_dir']}/wandb",
            name=f"{self.config['dataset']}_{self.config['model']}_agents{self.config['agents']}_{exp_group}",
            group=exp_group,
            job_type="orchestrator",
            config=self.config,
            reinit=True,
            settings=wandb.Settings(start_method="thread")
        )
        
    def print_gpu_memory_summary(self) -> None:
        """Print a summary of GPU memory usage for all available GPUs."""
        if not torch.cuda.is_available():
            print("No CUDA GPUs available")
            return
            
        print("\n=== GPU MEMORY SUMMARY ===")
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
            reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)
            free_memory = total_memory - allocated_memory
            
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total memory: {total_memory:.2f} GB")
            print(f"  Allocated: {allocated_memory:.2f} GB")
            print(f"  Reserved: {reserved_memory:.2f} GB")
            print(f"  Free: {free_memory:.2f} GB")
        print("==========================\n")

    def train_agent(
        self,
        agent_idx: int,
        debate: Dict[str, Any],
        iter_idx: int,
        phase: str = 'sft',
        precomputed_rewards: Optional[List[float]] = None
    ) -> None:
        """
        Train a specific agent using either supervised fine-tuning or reinforcement learning.
        
        Args:
            agent_idx: Index of the agent to train
            debate: Debate data containing agent interactions
            iter_idx: Iteration index
            phase: Training phase ('sft' for supervised fine-tuning or 'rl' for reinforcement learning)
            precomputed_rewards: Optional precomputed reward values for RL training
        """
        checkpoint_path = f"{self.config['experiment_dir']}/checkpoints/agent_{agent_idx}"
        
        # Check if we're using adapter mode
        adapter_mode = self.config.get('use_adapter_mode', False) and self.config.get('use_quantization', False)
        
        # Initialize the agent
        agent = Agent(
            self.config['model'],
            agent_id=agent_idx,
            checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
            device_map='auto',
            wandb_run=wandb.run,
            use_cache=True if phase == 'sft' else False,
            seed=self.config['seed'],
            devices=self.config['devices'],
            quantization=self.config['use_quantization'],
            adapter_mode=adapter_mode,
            task=self.config['dataset'],
            for_training=True,  # This agent is being created for training
        )

        print(f"Agent {agent_idx} | Device: {agent.get_device()}", flush=True)
        set_random_seed(self.config['seed'])
        
        try:
            # Choose training method based on phase
            if phase == 'sft':
                train_data = prepare_finetune_data(debate, agent_idx)
                agent.finetune(train_data, self.config, iter_idx, checkpoint_path)
            elif phase == 'dpo':
                train_data = prepare_dpo_data(
                    debate,
                    include_debate_context=self.config.get('include_debate_context', False),
                    pref_debate_context=self.config.get('pref_debate_context', False),
                    dataset=self.config['dataset'],
                    diversity_prompt=self.config.get('diversity_prompt', False),
                    use_majority_vote=self.config.get('use_majority_vote', True),
                    use_initial_responses=self.config.get('use_initial_responses', False),
                )

                original_size = len(train_data) if train_data else 0
                limit = self.config.get('train_size_dpo', -1)
                if isinstance(limit, int) and original_size > 0 and limit > 0 and original_size > limit:
                    train_data = train_data.select(range(limit))
                    print(f"Agent {agent_idx} |  Using first {limit} / {original_size} preference pairs for DPO")
                else:
                    print(f"Agent {agent_idx} |  Using all {original_size} preference pairs for DPO")

                agent.dpo(train_data, self.config, iter_idx, checkpoint_path)

            elif phase == 'kto':
                # Prepare KTO preference data
                train_data = prepare_kto_data(
                    debate,
                    include_debate_context=self.config.get('include_debate_context', False),
                    pref_debate_context=self.config.get('pref_debate_context', False),
                    dataset=self.config['dataset'],
                    diversity_prompt=self.config.get('diversity_prompt', False),
                    use_majority_vote=self.config.get('use_majority_vote', True),
                    use_initial_responses=self.config.get('use_initial_responses', False),
                )

                original_size = len(train_data) if train_data else 0
                limit = self.config.get('train_size_kto', -1)
                if isinstance(limit, int) and original_size > 0 and limit > 0 and original_size > limit:
                    train_data = train_data.select(range(limit))
                    print(f"Agent {agent_idx} |  Using first {limit} / {original_size} preference records for KTO")
                else:
                    print(f"Agent {agent_idx} |  Using all {original_size} preference records for KTO")

                # ---------- dynamic class-balance weights ----------
                try:
                    labels = train_data["label"]
                    pos_cnt = int(sum(labels))
                    neg_cnt = int(len(labels) - pos_cnt)

                    if pos_cnt > 0 and neg_cnt > 0:
                        if pos_cnt > neg_cnt:
                            desirable_w = 1.0
                            undesirable_w = min(pos_cnt / neg_cnt, 4/3)
                        else:
                            undesirable_w = 1.0
                            desirable_w = min(neg_cnt / pos_cnt, 4/3)
                        cfg = dict(self.config)
                        cfg["desirable_weight"] = desirable_w
                        cfg["undesirable_weight"] = undesirable_w
                    else:
                        cfg = self.config
                except Exception as err:
                    print(f"Agent {agent_idx} |  Could not compute KTO weights: {err}")
                    cfg = self.config

                agent.kto(train_data, cfg, iter_idx, checkpoint_path)

            else:
                # Use the new prepare_post_train_data with optional parameters
                train_data = prepare_post_train_data(
                    debate, 
                    agent_idx,
                    include_debate_context=self.config.get('include_debate_context', False),
                    dataset=self.config['dataset'],
                    diversity_prompt=self.config.get('diversity_prompt', False),
                    use_initial_responses=self.config.get('use_initial_responses', False)
                )
                
                # Get original size of training data
                original_size = len(train_data) if train_data else 0
                print(f"Train data size: {original_size}")
                
                # Simply take the first train_size samples or less if not enough data
                max_samples = min(self.config['train_size'], original_size)
                if original_size > max_samples:
                    train_data = train_data.select(range(max_samples))
                    print(f"Using first {max_samples} samples for training")
                
                # Only pass precomputed rewards if available
                kwargs = {}
                if precomputed_rewards:  # Use precomputed logp-based rewards
                    kwargs['precomputed_rewards'] = precomputed_rewards
                
                agent.post_train(train_data, self.config, iter_idx, checkpoint_path, **kwargs)
        finally:
            # Proper cleanup
            agent.cleanup()
            cleanup_gpu()

    def _load_or_create_initial_debate(self) -> Dict[str, Any]:
        """
        Load existing debate data or create a new one for the initial round.
        Reuses existing debate results when possible.
        
        Returns:
            Dictionary containing debate data
        """
        os.makedirs('data/debate', exist_ok=True)
        
        # Debate file path (includes train_size and test_size)
        debate_path_global = f"data/debate/{self.config['model']}_{self.config['dataset']}_{self.config['train_size']}_{self.config['test_size']}_{self.config['agents']}_{self.config['rounds']}_{self.config['data_seed']}_{self.config['diversity_prompt']}_{self.config['summarize']}.json"
        
        debate_path_experiment = f"{self.config['experiment_dir']}/debate/debate_iteration_0.json"
        
        # Simple debate file handling: load existing or create new
        if os.path.exists(debate_path_global):
            # Load existing debate file
            debate = json.load(open(debate_path_global, 'r'))
            os.makedirs(f"{self.config['experiment_dir']}/debate", exist_ok=True)
            shutil.copy2(debate_path_global, debate_path_experiment)
            print(f"Loaded existing debate file: {debate_path_global}")
        else:
            # No exact match found, create new debate from scratch
            print(f"No existing debate file found, creating new debate from scratch...")
            
            # Determine if we should use the scheduler-based implementation
            use_scheduler = (
                self.config.get('use_adapter_mode', False) and 
                self.config.get('use_quantization', False) and
                self.config.get('use_scheduler', True)
            )
            
            # Create new debate
            if use_scheduler:
                pd("Using scheduled debate with adapter swapping for initial debate")
                debate = asyncio.run(scheduled_multiagent_debate(
                    data=self.data,
                    config=self.config,
                    iter_idx=0
                ))
            else:
                pd("Using distributed debate for initial debate")
                debate = asyncio.run(distributed_multiagent_debate(
                    data=self.data,
                    config=self.config,
                    iter_idx=0
                ))
            
            # Save the debate file
            os.makedirs(f"{self.config['experiment_dir']}/debate", exist_ok=True)
            with open(debate_path_experiment, 'w') as f:
                json.dump(debate, f)
            shutil.copy2(debate_path_experiment, debate_path_global)

        tmp = {}
        for k, v in debate.items():
            if k == 'metrics':
                continue
            if v['split'] == 'train':
                tmp[k] = v
            elif v['split'] == 'test':
                tmp[k] = v
        
        return tmp

    def run_debate(
        self,
        iter_idx: int
    ) -> Dict[str, Any]:
        """
        Run a debate round.

        Args:
            iter_idx: Current iteration index

        Returns:
            Dictionary containing debate data
        """
        if iter_idx == 0:
            debate = self._load_or_create_initial_debate()

            return debate

        # ------------------ iteration > 0 ------------------
        existing_debate_path = f"{self.config['experiment_dir']}/debate/debate_iteration_{iter_idx}.json"
        existing_debate: Optional[Dict[str, Any]] = None
        existing_splits: Set[str] = set()

        if os.path.exists(existing_debate_path):
            try:
                with open(existing_debate_path, 'r') as f:
                    existing_debate = json.load(f)
                for k, v in existing_debate.items():
                    if k != 'metrics' and isinstance(v, dict) and 'split' in v:
                        existing_splits.add(v['split'])
                print(f"Found existing debate results with splits: {existing_splits}")
            except Exception as e:
                print(f"Warning: Could not load existing debate results: {e}")
                existing_debate = None

        # Determine backend
        use_scheduler = (
            self.config.get('use_adapter_mode', False) and
            self.config.get('use_quantization', False) and
            self.config.get('use_scheduler', True)
        )

        # Requested splits
        data_for_debate = self.data
        current_run_splits: Set[str] = {'train', 'test'}  # default

        if self.config.get('test_only', False) and self.config.get('train_only', False):
            raise ValueError("Cannot set both test_only and train_only flags simultaneously")
        elif self.config.get('test_only', False):
            data_for_debate = [item for item in self.data if item.get('split') == 'test']
            current_run_splits = {'test'}
            print(f"Test-only mode: evaluating on {len(data_for_debate)} test examples (skipping {len(self.data) - len(data_for_debate)} training examples)")
        elif self.config.get('train_only', False):
            data_for_debate = [item for item in self.data if item.get('split') == 'train']
            current_run_splits = {'train'}
            print(f"Train-only mode: evaluating on {len(data_for_debate)} training examples (skipping {len(self.data) - len(data_for_debate)} test examples)")

        overwrite = self.config.get('overwrite_existing_splits', False)

        # ------------- Smart merging for train/test splits -------------
        if existing_debate and existing_splits:
            missing_splits = current_run_splits - existing_splits
            overlap_splits = current_run_splits & existing_splits

            # If anything is missing, run ONLY the missing split(s) and merge
            if missing_splits:
                print(
                    f"Merge: existing {existing_splits}; "
                    f"running only missing {missing_splits}"
                    + (f" (overlap present: {overlap_splits}, preserved)" if overlap_splits else "")
                )

                # Subset data to missing splits
                data_subset = [item for item in data_for_debate if item.get('split') in missing_splits]
                print(f"Running debate on {len(data_subset)} examples for splits {missing_splits}")

                if use_scheduler:
                    new_debate_results = asyncio.run(scheduled_multiagent_debate(
                        data=data_subset,
                        config=self.config,
                        iter_idx=iter_idx
                    ))
                else:
                    new_debate_results = asyncio.run(distributed_multiagent_debate(
                        data=data_subset,
                        config=self.config,
                        iter_idx=iter_idx
                    ))

                # Merge results
                merged_debate: Dict[str, Any] = {}
                for key, value in existing_debate.items():
                    if key != 'metrics':
                        merged_debate[key] = value

                for key, value in new_debate_results.items():
                    if key == 'metrics':
                        continue
                    if key in merged_debate:
                        # Protect against accidental id collision across different splits
                        old_split = merged_debate[key].get('split')
                        new_split = value.get('split')
                        if old_split != new_split:
                            raise ValueError(
                                f"Debate merge id collision for key {key}: "
                                f"existing split {old_split}, new split {new_split}"
                            )
                        # Same key & split: keep existing unless overwrite requested
                        if not overwrite:
                            continue
                    merged_debate[key] = value


                with open(existing_debate_path, 'w') as f:
                    json.dump(merged_debate, f)

                print(f"Successfully merged debate results. Combined dataset now includes: {existing_splits | current_run_splits}")
                debate = merged_debate

            else:
                # Nothing missing. Reuse or overwrite according to flag.
                if overlap_splits and not overwrite:
                    print(f"Reusing existing debate results for splits {overlap_splits}. "
                        f"Set overwrite_existing_splits=True to recompute.")
                    debate = existing_debate
                else:
                    if overlap_splits:
                        print(f"Warning: Re-running and overwriting existing splits {overlap_splits} (overwrite_existing_splits=True)")
                    else:
                        print(f"Unexpected case: running {current_run_splits}, existing {existing_splits}; proceeding to run requested splits.")

                    # Run debate on the requested data splits
                    print(f"Running debate on {len(data_for_debate)} examples for splits {current_run_splits}")

                    if use_scheduler:
                        debate = asyncio.run(scheduled_multiagent_debate(
                            data=data_for_debate,
                            config=self.config,
                            iter_idx=iter_idx
                        ))
                    else:
                        debate = asyncio.run(distributed_multiagent_debate(
                            data=data_for_debate,
                            config=self.config,
                            iter_idx=iter_idx
                        ))

                    # Save debate results
                    with open(existing_debate_path, 'w') as f:
                        json.dump(debate, f)

        else:
            # No existing debate or failed to read it → run normally
            print(f"Running debate on {len(data_for_debate)} examples for splits {current_run_splits}")
            
            if use_scheduler:
                debate = asyncio.run(scheduled_multiagent_debate(
                    data=data_for_debate,
                    config=self.config,
                    iter_idx=iter_idx
                ))
            else:
                debate = asyncio.run(distributed_multiagent_debate(
                    data=data_for_debate,
                    config=self.config,
                    iter_idx=iter_idx
                ))

            # Save fresh results
            os.makedirs(os.path.dirname(existing_debate_path), exist_ok=True)
            with open(existing_debate_path, 'w') as f:
                json.dump(debate, f)

        # ------------------ Return ------------------

        return debate

    def run_training(self) -> None:
        """
        Run the complete multi-agent training process through multiple iterations.
        This is the main entry point for the training pipeline.
        """
        # Check if parallel training is enabled
        use_parallel_training = self.config.get('use_parallel_training', True)  # Default to parallel
        
        # Initial debate
        print("\n=========== Debate 0 ===========", flush=True)
        debate = self.run_debate(iter_idx=0)
        
        for iter_idx in range(self.config['iterations']):
            # SFT Phase
            if self.config['finetune']:
                print(f"\n=========== SFT {iter_idx} ===========", flush=True)
                if use_parallel_training:
                    self.train_agents_parallel(debate, iter_idx+1, 'sft')
                else:
                    for agent_idx in range(self.config['agents']):
                        self.train_agent(agent_idx, debate, iter_idx+1, 'sft')
            
            # DPO Phase
            if self.config.get('dpo', False):
                print(f"\n=========== DPO {iter_idx} ===========", flush=True)
                if use_parallel_training:
                    self.train_agents_parallel(debate, iter_idx+1, 'dpo')
                else:
                    for agent_idx in range(self.config['agents']):
                        self.train_agent(agent_idx, debate, iter_idx+1, 'dpo')
            
            # KTO Phase
            if self.config.get('kto', False):
                print(f"\n=========== KTO {iter_idx} ===========", flush=True)
                if use_parallel_training:
                    self.train_agents_parallel(debate, iter_idx+1, 'kto')
                else:
                    for agent_idx in range(self.config['agents']):
                        self.train_agent(agent_idx, debate, iter_idx+1, 'kto')
            
            # RL Phase
            if self.config['post_train']:
                print(f"\n=========== RL {iter_idx} ===========", flush=True)
                if use_parallel_training:
                    self.train_agents_parallel(debate, iter_idx+1, 'rl')
                else:
                    for agent_idx in range(self.config['agents']):
                        self.train_agent(agent_idx, debate, iter_idx+1, 'rl')
            
            # Debate Phase
            print(f"\n=========== Debate {iter_idx + 1} ===========", flush=True)
            debate = self.run_debate(iter_idx+1)
        
        # Run experiment performance analysis at the end
        print(f"\n=========== EXPERIMENT PERFORMANCE ANALYSIS ===========", flush=True)
        try:
            # Check if debate files exist
            debate_dir = f"{self.config['experiment_dir']}/debate"
            if not os.path.exists(debate_dir):
                print(f"Warning: No debate directory found at {debate_dir}, skipping analysis")
                return
            
            # Check if we have at least 2 debate files for comparison
            debate_files = [f for f in os.listdir(debate_dir) if f.startswith("debate_iteration_") and f.endswith(".json")]
            if len(debate_files) < 2:
                print(f"Warning: Only {len(debate_files)} debate files found, need at least 2 for analysis")
                return
            
            # Analyze the experiment
            results = analyze_experiment_performance(
                self.config['experiment_dir'], 
                test_size=self.config['test_size'], 
                agents=self.config['agents']
            )
            
            if not results:
                print("Warning: No analysis results generated, skipping output")
                return
            
            # Print the analysis to terminal
            print_performance_comparison(results)
            
            # Create structured summary JSON with full-denominator metrics
            structured_summary = create_structured_experiment_summary(results)
            if structured_summary:
                summary_path = f"{self.config['experiment_dir']}/experiment_summary_structured.json"
                with open(summary_path, 'w') as f:
                    json.dump(structured_summary, f, indent=2)
                print(f"Structured experiment summary saved to {summary_path}")
                
                # Print full denominator metrics to terminal
                if 'full_denominator_metrics' in structured_summary:
                    print(f"\n=========== FULL DENOMINATOR METRICS (out of {structured_summary['full_denominator_metrics']['total_expected_examples']} examples) ===========")
                    full_metrics = structured_summary['full_denominator_metrics']
                    print(f"Initial Debate Train Accuracy: {full_metrics['initial_debate_train_accuracy']:.4f}")
                    print(f"Initial Debate Test Accuracy:  {full_metrics['initial_debate_test_accuracy']:.4f}")
                    print(f"Final Debate Train Accuracy:   {full_metrics['final_debate_train_accuracy']:.4f}")
                    print(f"Final Debate Test Accuracy:    {full_metrics['final_debate_test_accuracy']:.4f}")
                    
                    # Calculate improvements
                    train_improvement = full_metrics['final_debate_train_accuracy'] - full_metrics['initial_debate_train_accuracy']
                    test_improvement = full_metrics['final_debate_test_accuracy'] - full_metrics['initial_debate_test_accuracy']
                    print(f"\nImprovements:")
                    print(f"Train Accuracy: {train_improvement:+.4f}")
                    print(f"Test Accuracy:  {test_improvement:+.4f}")
                    print("=" * 70)
                    
                    # Print debate stage analysis
                    if 'iterations' in structured_summary:
                        print(f"\n=========== DEBATE STAGE ANALYSIS ===========")
                        print(f"Round 0 Avg | Round 0 MajVot | Round 1 Avg | Round 1 MajVot")
                        print("-" * 65)
                        
                        iterations = sorted(structured_summary['iterations'].keys())
                        for iteration in iterations:
                            iter_data = structured_summary['iterations'][iteration]
                            r0_avg = iter_data['round_0_avg_0shot_accuracy']
                            r0_maj = iter_data['round_0_majority_vote_accuracy']
                            r1_avg = iter_data['round_1_avg_0shot_accuracy']
                            r1_maj = iter_data['round_1_majority_vote_accuracy']
                            
                            print(f"Iteration {iteration}: {r0_avg:.4f} | {r0_maj:.4f} | {r1_avg:.4f} | {r1_maj:.4f}")
                        print("=" * 65)
            
            # Log analysis metrics to wandb
            if 'wandb' in globals() and getattr(wandb, 'run', None) is not None:
                iterations = sorted(results.keys())
                for iteration in iterations:
                    metrics = results[iteration]['metrics']
                    log = {"iter": int(iteration)}

                    # ---------- R1 consensus (present) ----------
                    test_blk = metrics.get("test", {})
                    r1_present_acc = float(test_blk.get("consensus_accuracy", 0.0))
                    r1_present_corr = int(test_blk.get("consensus_correct", 0))
                    r1_present_tot = int(test_blk.get("count", 0))
                    log.update({
                        "iter/r1_consensus_present_acc": r1_present_acc,
                        "iter/r1_consensus_present_correct": r1_present_corr,
                        "iter/r1_consensus_present_total": r1_present_tot,
                    })

                    # ---------- R1 consensus (full) ----------
                    cf = metrics.get("consensus_full")
                    if isinstance(cf, dict):
                        r1_full_acc = float(cf.get("acc_expected", 0.0))
                        r1_full_corr = int(cf.get("correct", 0))
                        r1_full_tot = int(cf.get("expected", 0))
                        log.update({
                            "iter/r1_consensus_full_acc": r1_full_acc,
                            "iter/r1_consensus_full_correct": r1_full_corr,
                            "iter/r1_consensus_full_total": r1_full_tot,
                        })

                    # ---------- R0 majority vote ----------
                    # present
                    if "r0_majority_accuracy" in metrics:
                        log["iter/r0_majority_present_acc"] = float(metrics["r0_majority_accuracy"])
                    # full
                    if "r0_majority_correct_full" in metrics and "r0_majority_total_full" in metrics:
                        r0_full_corr = int(metrics["r0_majority_correct_full"])
                        r0_full_tot = int(metrics["r0_majority_total_full"] or 1)
                        log.update({
                            "iter/r0_majority_full_acc": r0_full_corr / r0_full_tot,
                            "iter/r0_majority_full_correct": r0_full_corr,
                            "iter/r0_majority_full_total": r0_full_tot,
                        })

                    # ---------- Agent averages ----------
                    # present averages (computed from the script’s present fields)
                    ap = metrics.get("agent_performance", {})
                    r0_present_vals = []
                    r1_present_vals = []
                    for v in ap.values():
                        if isinstance(v, dict):
                            if "test_round0_accuracy" in v:
                                r0_present_vals.append(float(v["test_round0_accuracy"]))
                            if "test_round1_accuracy" in v or "test_accuracy" in v:
                                r1_present_vals.append(float(v.get("test_round1_accuracy", v.get("test_accuracy", 0.0))))
                    if r0_present_vals:
                        log["iter/avg_r0_present"] = float(np.mean(r0_present_vals))
                    # script already reports this as test.avg_agent_accuracy, but keep explicit:
                    if r1_present_vals:
                        log["iter/avg_r1_present"] = float(np.mean(r1_present_vals))
                    elif "avg_agent_accuracy" in test_blk:
                        log["iter/avg_r1_present"] = float(test_blk["avg_agent_accuracy"])

                    # full averages (init_all/final_all)
                    apf = metrics.get("agent_performance_full", {})
                    init_all_vals = [float(v.get("init_all", 0.0)) for v in apf.values() if isinstance(v, dict)]
                    final_all_vals = [float(v.get("final_all", 0.0)) for v in apf.values() if isinstance(v, dict)]
                    if init_all_vals:
                        log["iter/avg_r0_all"] = float(np.mean(init_all_vals))
                    if final_all_vals:
                        log["iter/avg_r1_all"] = float(np.mean(final_all_vals))

                    # ---------- Per-agent metrics (present + full) ----------
                    for agent_id, v in apf.items():
                        if not isinstance(v, dict):
                            continue
                        # R0
                        if "init_present" in v:
                            log[f"iter/{agent_id}/r0_present"] = float(v["init_present"])
                        if "init_all" in v:
                            log[f"iter/{agent_id}/r0_all"] = float(v["init_all"])
                        # R1
                        if "final_present" in v:
                            log[f"iter/{agent_id}/r1_present"] = float(v["final_present"])
                        if "final_all" in v:
                            log[f"iter/{agent_id}/r1_all"] = float(v["final_all"])
                        # trainer metrics (if attached)
                        for name in ["loss", "kl", "reward_chosen", "reward_rejected", "margin"]:
                            iv = v.get(f"initial_{name}")
                            fv = v.get(f"final_{name}")
                            if iv is not None:
                                log[f"iter/{agent_id}/initial_{name}"] = float(iv)
                            if fv is not None:
                                log[f"iter/{agent_id}/final_{name}"] = float(fv)

                    # Also push the script’s present summary fields for continuity
                    log["iter/consensus_accuracy"] = r1_present_acc
                    log["iter/consensus_correct"] = r1_present_corr
                    log["iter/consensus_total"] = r1_present_tot

                    wandb.log(log, step=int(iteration))

                print("Logged analysis metrics to wandb using iter/* namespace (present + full).")
            
            # Generate summary plot
            plot_path = f"{self.config['experiment_dir']}/experiment_summary.png"
            generate_summary_plots(results, plot_path)
            
            # Save the analysis results
            analysis_path = f"{self.config['experiment_dir']}/experiment_analysis.json"
            with open(analysis_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Experiment analysis saved to {analysis_path}")
            print(f"Summary plot saved to {plot_path}")
            
        except Exception as e:
            print(f"Warning: Could not run experiment performance analysis: {e}")
            import traceback
            traceback.print_exc()
        
        wandb.finish()
        self.cleanup()


    def cleanup(self) -> None:
        """
        Cleanup resources to prevent memory leaks.
        Should be called at the end of training.
        """
        if hasattr(self, 'gpu_manager'):
            self.gpu_manager.cleanup()
        cleanup_gpu()
        
        # Small delay to allow subprocess cleanup to complete
        time.sleep(2)
        
        # Force additional cleanup
        cleanup_gpu()
        
        # Print final memory status before continuing
        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                max_allocated = torch.cuda.max_memory_allocated(gpu_id) / (1024**3)
                props = torch.cuda.get_device_properties(gpu_id)
                total = props.total_memory / (1024**3)
                free = total - allocated
        


    def train_agents_parallel(
        self,
        debate: Dict[str, Any],
        iter_idx: int,
        phase: str = 'rl',
        precomputed_rewards: Optional[Dict[int, List[float]]] = None
    ) -> None:
        """Train multiple agents in parallel using subprocess isolation."""
        import subprocess
        import sys
        import threading
        import queue
        import time
        import tempfile
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        num_agents = self.config['agents']
        
        
        # Create temporary files for data serialization
        temp_dir = tempfile.mkdtemp(prefix=f"agent_training_iter_{iter_idx}_")
        
        try:
            # 1. Serialize configuration to file
            config_file = os.path.join(temp_dir, "config.json")
            with open(config_file, 'w') as f:
                # Create a serializable version of config
                serializable_config = {}
                for key, value in self.config.items():
                    # Skip non-serializable objects
                    if key in ['devices']:  # devices will be set per subprocess
                        continue
                    try:
                        json.dumps(value)  # Test if serializable
                        serializable_config[key] = value
                    except (TypeError, ValueError):
                        # Skip non-serializable values
                        continue
                json.dump(serializable_config, f, indent=2)
            
            # 2. Serialize debate data to file
            debate_file = os.path.join(temp_dir, f"debate_iter_{iter_idx}.json")
            with open(debate_file, 'w') as f:
                json.dump(debate, f)
            
            # 3. Serialize precomputed rewards if provided
            rewards_files = {}
            if precomputed_rewards:
                for agent_idx, rewards in precomputed_rewards.items():
                    rewards_file = os.path.join(temp_dir, f"rewards_agent_{agent_idx}.json")
                    with open(rewards_file, 'w') as f:
                        json.dump(rewards, f)
                    rewards_files[agent_idx] = rewards_file
                    print(f" Serialized precomputed rewards for {len(rewards_files)} agents")
            
            # Pre-allocate GPUs for all agents
            gpu_allocations = {}
            available_gpu_ids = [gpu['id'] for gpu in self.gpu_manager.available_gpus]
            
            for agent_idx in range(num_agents):
                # Simple round-robin allocation
                gpu_id = available_gpu_ids[agent_idx % len(available_gpu_ids)]
                gpu_allocations[agent_idx] = gpu_id
                # print(f" Pre-allocated GPU {gpu_id} for Agent {agent_idx}")
            
            # Prepare training commands for each agent
            training_commands = []
            for agent_idx in range(num_agents):
                allocated_gpu = gpu_allocations[agent_idx]
                
                # Build command to run train_agent_subprocess.py
                cmd = [
                    sys.executable,
                    'train_agent_subprocess.py',
                    '--agent_idx', str(agent_idx),
                    '--gpu_id', str(allocated_gpu),
                    '--phase', phase,
                    '--iter_idx', str(iter_idx),
                    '--experiment_dir', self.config['experiment_dir'],
                    '--debate_data_file', debate_file,
                    '--config_file', config_file,
                ]
                
                # Add rewards file if available for this agent
                if agent_idx in rewards_files:
                    cmd.extend(['--rewards_file', rewards_files[agent_idx]])
                
                training_commands.append((agent_idx, allocated_gpu, cmd))
            
            # Run training using ThreadPoolExecutor with subprocess calls
            def run_agent_subprocess(agent_info):
                agent_idx, gpu_id, cmd = agent_info
                agent_prefix = f"Agent-{agent_idx}"
                start_time = time.time()
                
                try:
                    # Use Popen for truly non-blocking execution
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,  # Merge stderr into stdout
                        text=True,
                        bufsize=1,  # Line buffered
                        universal_newlines=True
                    )
                    
                    # Stream output in real-time with agent prefix
                    output_lines = []
                    timeout_seconds = 7200  # 2 hour timeout
                    start_poll_time = time.time()
                    
                    while True:
                        line = process.stdout.readline()
                        if line:
                            # Add agent prefix to each line for identification
                            prefixed_line = f"{agent_prefix} | {line.rstrip()}"
                            print(prefixed_line)
                            output_lines.append(line)
                            start_poll_time = time.time()  # Reset timeout on activity
                        elif process.poll() is not None:
                            break
                        elif time.time() - start_poll_time > timeout_seconds:
                            print(f"{agent_prefix} |  Timeout after {timeout_seconds}s, terminating...")
                            process.terminate()
                            process.wait()
                            runtime = time.time() - start_time
                            return agent_idx, False, runtime, gpu_id
                    
                    # Wait for process to complete and get return code
                    return_code = process.wait()
                    
                    runtime = time.time() - start_time
                    
                    if return_code == 0:
                        # print(f"{agent_prefix} |  Completed on GPU {gpu_id} ({runtime:.2f}s)")
                        return agent_idx, True, runtime, gpu_id
                    else:
                        print(f"{agent_prefix} |  Failed on GPU {gpu_id} ({runtime:.2f}s)")
                        print(f"{agent_prefix} |  Exit code: {return_code}")
                        return agent_idx, False, runtime, gpu_id
                    
                except Exception as e:
                    runtime = time.time() - start_time
                    print(f"{agent_prefix} |  Unexpected error on GPU {gpu_id} ({runtime:.2f}s): {e}")
                    return agent_idx, False, runtime, gpu_id
            
            # Calculate optimal number of concurrent processes
            max_gpu_slots = len(self.gpu_manager.available_gpus) * self.gpu_manager.max_concurrent_per_gpu
            max_processes = min(num_agents, max_gpu_slots, 8)  # Cap at 8 processes for stability
            
            # Track training progress
            start_time = time.time()
            results = {}
            
            with ThreadPoolExecutor(max_workers=max_processes) as executor:
                # Submit all training jobs
                future_to_agent = {
                    executor.submit(run_agent_subprocess, cmd_info): cmd_info[0] 
                    for cmd_info in training_commands
                }
                
                # Monitor progress with periodic status updates
                completed = 0
                last_status_time = 0
                status_update_interval = 30  # Update every 30 seconds
                
                for future in as_completed(future_to_agent):
                    agent_idx, success, runtime, gpu_id = future.result()
                    results[agent_idx] = {
                        'success': success,
                        'runtime': runtime,
                        'gpu_id': gpu_id
                    }
                    completed += 1
                    
                    current_time = time.time()
                    
                    # Show completion update
                    agent_prefix = f" Agent-{agent_idx}"
                    status_icon = "" if success else ""
                    print(f"{agent_prefix} | {status_icon} Training completed on GPU {gpu_id} in {runtime:.1f}s")
                    
                    # Show overall progress update
                    if current_time - last_status_time >= status_update_interval or completed == num_agents:
                        elapsed = current_time - start_time
                        remaining = num_agents - completed
                        
                        if remaining > 0:
                            # Show which agents are still running
                            running_agents = []
                            for agent_idx, gpu_id, _ in training_commands:
                                if agent_idx not in results:
                                    running_agents.append(f"Agent-{agent_idx}(GPU{gpu_id})")
                            
                            print(f" PARALLEL STATUS | {completed}/{num_agents} completed | {remaining} still running: {', '.join(running_agents)} | {elapsed:.1f}s elapsed")
                        else:
                            print(f" ALL AGENTS COMPLETED | Total time: {elapsed:.1f}s")
                        
                        last_status_time = current_time
            
            # Analyze results
            successful_agents = sum(1 for r in results.values() if r['success'])
            failed_agents = num_agents - successful_agents
            total_time = time.time() - start_time
            
            print(f"\n{'='*60}")
            print(f" REAL PARALLEL TRAINING SUMMARY ({phase.upper()})")
            print(f"{'='*60}")
            print(f" Successful agents: {successful_agents}/{num_agents}")
            print(f" Total runtime: {total_time:.2f} seconds")
            
            if failed_agents > 0:
                print(f" Failed agents: {failed_agents}")
                for agent_idx, result in results.items():
                    if not result['success']:
                        agent_prefix = f" Agent-{agent_idx}"
                        print(f"   {agent_prefix}: Failed on GPU {result['gpu_id']}")
            
            # Final GPU status
            self.gpu_manager.print_status()
            
        finally:
            # Cleanup temporary files
            try:
                import shutil
                shutil.rmtree(temp_dir)
                # print(f" Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f" Warning: Could not clean up temp directory {temp_dir}: {e}")
            
            # Force cleanup after parallel training
            cleanup_gpu()

class GPUResourceManager:
    """GPU resource manager for dynamic allocation and monitoring."""
    
    def __init__(
        self,
        exclude_gpus: Optional[List[int]] = None,
        max_concurrent_per_gpu: int = 1
    ):
        """
        Initialize GPU resource manager.
        
        Args:
            exclude_gpus: List of GPU IDs to exclude from allocation
            max_concurrent_per_gpu: Maximum concurrent agents per GPU
        """
        self.exclude_gpus = exclude_gpus or []
        self.max_concurrent_per_gpu = max_concurrent_per_gpu
        self.available_gpus = self._detect_available_gpus()
        self.gpu_queue = queue.Queue()
        self.active_allocations = {}  # gpu_id -> list of agent_ids
        self.allocation_history = {}  # agent_id -> gpu_id
        self.lock = threading.Lock()
        
        # Initialize GPU queue
        for gpu_info in self.available_gpus:
            for _ in range(max_concurrent_per_gpu):
                self.gpu_queue.put(gpu_info['id'])
        
        pd(f" GPU Resource Manager initialized")
        pd(f"   Available GPUs: {[gpu['id'] for gpu in self.available_gpus]}")
        pd(f"   Total GPU slots: {len(self.available_gpus) * max_concurrent_per_gpu}")
    
    def _detect_available_gpus(self) -> List[Dict[str, Any]]:
        """Detect and validate available GPUs."""
        available_gpus = []
        
        if not torch.cuda.is_available():
            print(" CUDA not available")
            return available_gpus
        
        total_gpus = torch.cuda.device_count()
        print(f" Detected {total_gpus} total GPUs")
        
        for gpu_id in range(total_gpus):
            if gpu_id in self.exclude_gpus:
                print(f"  GPU {gpu_id}: Excluded (in use)")
                continue
            
            try:
                # GPU health check
                with torch.cuda.device(gpu_id):
                    # Test basic tensor operations
                    test_tensor = torch.tensor([1.0]).cuda()
                    test_result = test_tensor * 2.0
                    
                    # Get memory information
                    props = torch.cuda.get_device_properties(gpu_id)
                    memory_total = props.total_memory
                    memory_allocated = torch.cuda.memory_allocated(gpu_id)
                    memory_free = memory_total - memory_allocated
                    
                    # Test memory allocation (allocate 1GB to verify availability)
                    test_size = min(1024**3, memory_free // 2)  # 1GB or half of free memory
                    large_tensor = torch.zeros(test_size // 4, dtype=torch.float32).cuda()
                    
                    # Handle different PyTorch versions with different property names
                    multiprocessor_count = getattr(props, 'multiprocessor_count', getattr(props, 'multi_processor_count', 'unknown'))
                    max_threads_per_mp = getattr(props, 'max_threads_per_multiprocessor', getattr(props, 'max_threads_per_block', 'unknown'))
                    
                    gpu_info = {
                        'id': gpu_id,
                        'name': props.name,
                        'memory_total': memory_total / (1024**3),  # GB
                        'memory_free': memory_free / (1024**3),   # GB
                        'compute_capability': f"{props.major}.{props.minor}",
                        'multiprocessor_count': multiprocessor_count,
                        'max_threads_per_multiprocessor': max_threads_per_mp,
                    }
                    
                    available_gpus.append(gpu_info)
                    
                    # Clean up test tensors
                    del test_tensor, test_result, large_tensor
                    torch.cuda.empty_cache()
                          
            except Exception as e:
                print(f" GPU {gpu_id}: Health check failed ({e})")
        
        return available_gpus
    
    def allocate_gpu(
        self,
        agent_id: int,
        timeout: float = 30.0
    ) -> Optional[int]:
        """
        Allocate a GPU for an agent with timeout.
        
        Args:
            agent_id: Agent requesting GPU allocation
            timeout: Maximum time to wait for GPU availability
            
        Returns:
            GPU ID if allocated, None if timeout
        """
        try:
            gpu_id = self.gpu_queue.get(timeout=timeout)
            
            with self.lock:
                if gpu_id not in self.active_allocations:
                    self.active_allocations[gpu_id] = []
                self.active_allocations[gpu_id].append(agent_id)
                self.allocation_history[agent_id] = gpu_id
            
            return gpu_id
            
        except queue.Empty:
            pd(f" GPU allocation timeout for Agent {agent_id}")
            return None
    
    def release_gpu(
        self,
        agent_id: int
    ) -> None:
        """
        Release GPU allocation for an agent.
        
        Args:
            agent_id: Agent releasing GPU
        """
        with self.lock:
            if agent_id in self.allocation_history:
                gpu_id = self.allocation_history[agent_id]
                
                # Remove from active allocations
                if gpu_id in self.active_allocations:
                    if agent_id in self.active_allocations[gpu_id]:
                        self.active_allocations[gpu_id].remove(agent_id)
                
                # Return GPU to queue
                self.gpu_queue.put(gpu_id)
                del self.allocation_history[agent_id]
                
                pd(f" Released GPU {gpu_id} from Agent {agent_id}")
    
    def get_utilization_stats(self) -> Dict[str, Any]:
        """
        Get current GPU utilization statistics.
        
        Returns:
            Dictionary with utilization metrics
        """
        with self.lock:
            total_slots = len(self.available_gpus) * self.max_concurrent_per_gpu
            allocated_slots = sum(len(agents) for agents in self.active_allocations.values())
            available_slots = self.gpu_queue.qsize()
            
            gpu_usage = {}
            for gpu_info in self.available_gpus:
                gpu_id = gpu_info['id']
                active_agents = self.active_allocations.get(gpu_id, [])
                gpu_usage[gpu_id] = {
                    'active_agents': len(active_agents),
                    'agent_ids': active_agents,
                    'utilization': len(active_agents) / self.max_concurrent_per_gpu
                }
            
            return {
                'total_slots': total_slots,
                'allocated_slots': allocated_slots,
                'available_slots': available_slots,
                'utilization': allocated_slots / total_slots if total_slots > 0 else 0.0,
                'gpu_usage': gpu_usage
            }
    
    def print_status(self) -> None:
        """Print current resource allocation status."""
        stats = self.get_utilization_stats()
        pd(f" GPU Resource Status:")
        pd(f"   Total slots: {stats['total_slots']}")
        pd(f"   Allocated: {stats['allocated_slots']}")
        pd(f"   Available: {stats['available_slots']}")
        pd(f"   Utilization: {stats['utilization']:.1%}")
        
        for gpu_id, usage in stats['gpu_usage'].items():
            if usage['active_agents'] > 0:
                agent_list = ', '.join(f"Agent{aid}" for aid in usage['agent_ids'])
                pd(f"   GPU {gpu_id}: {usage['active_agents']}/{self.max_concurrent_per_gpu} slots ({agent_list})")
    
    def cleanup(self) -> None:
        """Cleanup all GPU allocations."""
        with self.lock:
            self.active_allocations.clear()
            self.allocation_history.clear()
            
            # Reinitialize GPU queue
            while not self.gpu_queue.empty():
                try:
                    self.gpu_queue.get_nowait()
                except queue.Empty:
                    break
            
            for gpu_info in self.available_gpus:
                for _ in range(self.max_concurrent_per_gpu):
                    self.gpu_queue.put(gpu_info['id'])
        
        print(" GPU Resource Manager cleaned up")