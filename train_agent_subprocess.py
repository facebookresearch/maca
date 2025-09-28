#!/usr/bin/env python3
"""
Subprocess wrapper for real agent training with complete GPU isolation.

This script recreates the orchestrator environment in a fresh process to ensure
complete GPU isolation while using the actual training logic.
"""
import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
from transformers import TrainerCallback
import time
from typing import Dict, Any, List, Optional

def setup_gpu_environment(
    gpu_id: int
) -> None:
    """Set up GPU environment for the specified GPU ID."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f" Agent subprocess | Set CUDA_VISIBLE_DEVICES={gpu_id}")
    
    # Verify GPU is available
    if torch.cuda.is_available():
        # After setting CUDA_VISIBLE_DEVICES, the specified GPU becomes cuda:0
        print(f" Agent subprocess | GPU {gpu_id} is available (mapped to cuda:0)")
        print(f" Agent subprocess | GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        print(f"  Agent subprocess | Warning: CUDA not available after setting GPU {gpu_id}")

class ProgressBarCallback(TrainerCallback):
    """Custom callback to show training progress with live-updating progress bars for parallel execution."""
    
    def __init__(
        self,
        agent_id: int,
        training_mode: str
    ):
        self.agent_id = agent_id
        self.training_mode = training_mode.upper()
        self.agent_prefix = f" Agent-{agent_id}"
        self.progress_bar = None
        
    def on_train_begin(
        self,
        args,
        state,
        control,
        **kwargs
    ):
        """Initialize progress bar when training starts."""
        total_steps = args.max_steps if args.max_steps > 0 else args.num_train_epochs * len(kwargs.get('train_dataloader', []))
        
        # Create a distinctive description with agent info
        desc = f"{self.agent_prefix} [{self.training_mode}]"
        
        # Use a unique position for each agent to prevent overlap
        # Position 0 = Agent 0, Position 1 = Agent 1, etc.
        self.progress_bar = tqdm(
            total=total_steps,
            desc=desc,
            unit="step",
            ncols=100,  # Consistent width
            position=self.agent_id,  # Each agent gets its own line
            leave=True,  # Keep the bar after completion
            dynamic_ncols=False,  # Fixed width to prevent jumping
            bar_format=f"{self.agent_prefix} {{l_bar}}{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
        )
        
        # Print initial message above the progress bar
        tqdm.write(f"{self.agent_prefix} |  Starting {self.training_mode} training ({total_steps} steps)")
        
    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs
    ):
        """Update progress bar after each training step."""
        if self.progress_bar:
            # Get current metrics
            current_step = state.global_step
            
            # Update progress bar
            self.progress_bar.n = current_step
            self.progress_bar.refresh()
            
    def on_train_end(
        self,
        args,
        state,
        control,
        **kwargs
    ):
        """Close progress bar when training ends."""
        if self.progress_bar:
            self.progress_bar.set_postfix_str(" DONE")
            self.progress_bar.close()
            
            # Print completion message
            total_time = time.time() - self.progress_bar.start_t if hasattr(self.progress_bar, 'start_t') else 0
            tqdm.write(f"{self.agent_prefix} |  {self.training_mode} training completed! Total time: {total_time:.1f}s")
            
    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs
    ):
        """Handle logging events to update progress bar with metrics."""
        if self.progress_bar and logs:
            # Update with the latest metrics
            postfix = {}
            if 'train_loss' in logs:
                postfix['loss'] = f'{logs["train_loss"]:.4f}'
            elif 'loss' in logs:
                postfix['loss'] = f'{logs["loss"]:.4f}'
            if 'learning_rate' in logs:
                postfix['lr'] = f'{logs["learning_rate"]:.2e}'
            if 'epoch' in logs:
                postfix['epoch'] = f'{logs["epoch"]:.2f}'
            if 'reward' in logs:
                postfix['reward'] = f'{logs["reward"]:.3f}'
                
            if postfix:
                self.progress_bar.set_postfix(postfix)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for subprocess agent training."""
    parser = argparse.ArgumentParser(description="Subprocess agent training with real training logic")
    
    # Agent configuration
    parser.add_argument('--agent_idx', type=int, required=True, help="Agent index")
    parser.add_argument('--gpu_id', type=int, required=True, help="GPU ID to use")
    parser.add_argument('--phase', choices=['sft', 'rl', 'kto', 'dpo'], required=True, help="Training phase")
    parser.add_argument('--iter_idx', type=int, required=True, help="Iteration index")
    
    # File paths
    parser.add_argument('--experiment_dir', required=True, help="Experiment directory")
    parser.add_argument('--debate_data_file', required=True, help="Path to debate data JSON file")
    parser.add_argument('--config_file', required=True, help="Path to config JSON file")
    parser.add_argument('--rewards_file', help="Path to precomputed rewards file (RL only)")
    
    return parser.parse_args()

class MinimalOrchestrator:
    """
    Minimal orchestrator for subprocess training.
    Only creates components needed for training a single agent.
    """
    
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        """Initialize minimal orchestrator with only training components."""
        self.config = config
        
    
    def train_agent_isolated(
        self,
        agent_idx: int,
        debate_data: Dict[str, Any],
        iter_idx: int,
        phase: str,
        precomputed_rewards: Optional[List[float]] = None
    ) -> None:
        """
        Train a single agent in isolation (adapted from orchestrator.train_agent).
        
        Args:
            agent_idx: Index of the agent to train
            debate_data: Debate data dictionary
            iter_idx: Iteration index
            phase: Training phase ('sft' or 'rl')
            precomputed_rewards: Optional precomputed rewards for RL
        """
        from model import Agent
        from data import prepare_finetune_data, prepare_post_train_data, prepare_kto_data, prepare_dpo_data
        from utils import set_random_seed, cleanup_gpu
        import wandb
        import os
        
        agent_prefix = f" Agent-{agent_idx}"
        checkpoint_path = f"{self.config['experiment_dir']}/checkpoints/agent_{agent_idx}"
        
        # Check if we're using adapter mode
        adapter_mode = self.config.get('use_adapter_mode', False) and self.config.get('use_quantization', False)
        
        print(f"{agent_prefix} |  Starting {phase.upper()} training in isolated subprocess")
        print(f"{agent_prefix} |  Checkpoint: {checkpoint_path}")
        print(f"{agent_prefix} |  Adapter mode: {adapter_mode}")
        
        # Initialize wandb for this subprocess (suppress output)
        if self.config.get('entity_name') != 'test-entity':  # Skip for tests
            os.environ['WANDB_SILENT'] = 'true'  # Suppress wandb output
            # Re-use the persisted group id so every agent/phase stays under
            # the same WandB umbrella created by the orchestrator.
            group_file = os.path.join(self.config['experiment_dir'], '.wandb_group')
            if os.path.exists(group_file):
                with open(group_file) as _gf:
                    exp_group = _gf.read().strip()
            else:
                exp_group = os.path.basename(self.config.get('experiment_dir', 'experiment').rstrip('/'))
            run_name = f"Agent_{agent_idx}_{phase}_iter_{iter_idx}"
            wandb.init(
                project=self.config.get('project_name', "llm-marl"),
                entity=self.config.get('entity_name', "llm-marl"),
                dir=f"{self.config['experiment_dir']}/wandb",
                name=run_name,
                group=exp_group,
                job_type=phase,
                config=self.config,
                resume="allow",
                reinit=True,
                settings=wandb.Settings(start_method="thread", console="off")
            )
        else:
            # Disable wandb entirely for HuggingFace auto-initialization
            os.environ['WANDB_DISABLED'] = 'true'
        
        # Initialize the agent
        agent = Agent(
            self.config['model'],
            agent_id=agent_idx,
            checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
            device_map='auto',
            wandb_run=wandb.run if 'wandb' in locals() else None,
            use_cache=True if phase == 'sft' else False,
            seed=self.config['seed'],
            devices=[f"cuda:{torch.cuda.current_device()}"],  # Use current GPU only
            quantization=self.config['use_quantization'],
            adapter_mode=adapter_mode,
            task=self.config['dataset'],
            for_training=True,
        )

        print(f"{agent_prefix} |  Device: {agent.get_device()}")
        set_random_seed(self.config['seed'])
        
        try:
            # Choose training method based on phase
            if phase == 'sft':
                print(f"{agent_prefix} |  Preparing SFT data ({len(prepare_finetune_data(debate_data, agent_idx)) if prepare_finetune_data(debate_data, agent_idx) else 0} samples)")
                train_data = prepare_finetune_data(debate_data, agent_idx)
                
                # Add progress bar callback to agent's finetune method
                original_finetune = agent.finetune
                def finetune_with_progress(*args, **kwargs):
                    # Add progress callback if not already present
                    if hasattr(agent, 'trainer') and agent.trainer:
                        progress_callback = ProgressBarCallback(agent_idx, 'sft')
                        if progress_callback not in agent.trainer.callback_handler.callbacks:
                            agent.trainer.add_callback(progress_callback)
                    return original_finetune(*args, **kwargs)
                
                agent.finetune = finetune_with_progress
                agent.finetune(train_data, self.config, iter_idx, checkpoint_path)
                
            elif phase == 'rl':
                train_data = prepare_post_train_data(
                    debate_data, 
                    agent_idx,
                    include_debate_context=self.config.get('include_debate_context', False),
                    dataset=self.config['dataset'],
                    diversity_prompt=self.config.get('diversity_prompt', False),
                    use_initial_responses=self.config.get('use_initial_responses', False)
                )
                
                # Get original size of training data
                original_size = len(train_data) if train_data else 0
                print(f"{agent_prefix} |  Preparing RL data ({original_size} samples)")
                
                # Simply take the first train_size samples or less if not enough data
                max_samples = min(self.config['train_size'], original_size)
                if original_size > max_samples:
                    train_data = train_data.select(range(max_samples))
                
                # Only pass precomputed rewards if available
                kwargs = {}
                if precomputed_rewards:  # Use precomputed logp-based rewards
                    kwargs['precomputed_rewards'] = precomputed_rewards
                
                # Add progress bar callback to agent's post_train method
                original_post_train = agent.post_train
                def post_train_with_progress(*args, **kwargs):
                    # Add progress callback if not already present
                    if hasattr(agent, 'trainer') and agent.trainer:
                        progress_callback = ProgressBarCallback(agent_idx, 'rl')
                        if progress_callback not in agent.trainer.callback_handler.callbacks:
                            agent.trainer.add_callback(progress_callback)
                    return original_post_train(*args, **kwargs)
                
                agent.post_train = post_train_with_progress
                agent.post_train(train_data, self.config, iter_idx, checkpoint_path, **kwargs)
                
            elif phase == 'dpo':
                train_data = prepare_dpo_data(
                    debate_data,
                    include_debate_context=self.config.get('include_debate_context', False),
                    pref_debate_context=self.config.get('pref_debate_context', False),
                    dataset=self.config['dataset'],
                    diversity_prompt=self.config.get('diversity_prompt', False),
                    use_majority_vote=self.config.get('use_majority_vote', True),
                    use_initial_responses=self.config.get('use_initial_responses', False)
                )

                original_size = len(train_data) if train_data else 0
                print(f"{agent_prefix} |  Preparing DPO data ({original_size} pairs)")

                limit = self.config.get('train_size_dpo', -1)
                if isinstance(limit, int) and limit > 0 and original_size > limit:
                    train_data = train_data.select(range(limit))
                    print(f"{agent_prefix} |  Using first {limit} / {original_size} preference pairs for DPO")
                else:
                    print(f"{agent_prefix} |  Using all {original_size} preference pairs for DPO")

                # Wrap agent.dpo with progress bar callback
                original_dpo = getattr(agent, 'dpo', None)
                if original_dpo is not None:
                    def dpo_with_progress(*args, **kwargs):
                        if hasattr(agent, 'trainer') and agent.trainer:
                            progress_callback = ProgressBarCallback(agent_idx, 'dpo')
                            if progress_callback not in agent.trainer.callback_handler.callbacks:
                                agent.trainer.add_callback(progress_callback)
                        return original_dpo(*args, **kwargs)

                    agent.dpo = dpo_with_progress  # type: ignore
                    agent.dpo(train_data, self.config, iter_idx, checkpoint_path)
                else:
                    raise AttributeError("Agent is missing dpo method; implementation required.")

            else:  # kto phase
                train_data = prepare_kto_data(
                    debate_data,
                    include_debate_context=self.config.get('include_debate_context', False),
                    pref_debate_context=self.config.get('pref_debate_context', False),
                    dataset=self.config['dataset'],
                    diversity_prompt=self.config.get('diversity_prompt', False),
                    use_majority_vote=self.config.get('use_majority_vote', True),
                    use_initial_responses=self.config.get('use_initial_responses', False)
                )

                original_size = len(train_data) if train_data else 0
                print(f"{agent_prefix} |  Preparing KTO data ({original_size} samples)")

                limit = self.config.get('train_size_kto', -1)
                if isinstance(limit, int) and limit > 0 and original_size > limit:
                    train_data = train_data.select(range(limit))
                    print(f"{agent_prefix} |  Using first {limit} / {original_size} preference records for KTO")
                else:
                    print(f"{agent_prefix} |  Using all {original_size} preference records for KTO")

                # ------------------------------------------------------------------
                # Dynamically set desirable/undesirable weights based on class balance
                # ------------------------------------------------------------------
                try:
                    labels = train_data["label"]
                    pos_cnt = int(sum(labels))
                    neg_cnt = int(len(labels) - pos_cnt)

                    # Only adjust when both classes present
                    if pos_cnt > 0 and neg_cnt > 0:
                        if pos_cnt > neg_cnt:
                            desirable_w = 1.0
                            if self.config.get('disable_kto_weight_cap', False):
                                undesirable_w = pos_cnt / neg_cnt
                            else:
                                undesirable_w = min(pos_cnt / neg_cnt, 4/3)
                        else:
                            undesirable_w = 1.0
                            if self.config.get('disable_kto_weight_cap', False):
                                desirable_w = neg_cnt / pos_cnt
                            else:
                                desirable_w = min(neg_cnt / pos_cnt, 4/3)

                        # Apply scaling factor if specified
                        weight_scale = self.config.get('kto_weight_scale', 1.0)
                        desirable_w *= weight_scale
                        undesirable_w *= weight_scale

                        # Optionally flip desirable/undesirable weights (risk-seeking)
                        if self.config.get('flip_kto_weights', False):
                            desirable_w, undesirable_w = undesirable_w, desirable_w

                        # Clone config so we don't mutate shared dict unexpectedly
                        kto_config = dict(self.config)
                        kto_config["desirable_weight"] = desirable_w
                        kto_config["undesirable_weight"] = undesirable_w
                    else:
                        kto_config = self.config  # leave defaults
                except Exception as weight_err:
                    print(f"{agent_prefix} |  Warning computing KTO class weights: {weight_err}")
                    kto_config = self.config

                # Save original method to avoid infinite recursion
                original_kto = agent.kto

                def kto_with_progress(*args, **kwargs):
                    """Wrapper that injects a progress bar into Agent.kto training."""
                    if hasattr(agent, 'trainer') and agent.trainer:
                        progress_callback = ProgressBarCallback(agent_idx, 'kto')
                        if progress_callback not in agent.trainer.callback_handler.callbacks:
                            agent.trainer.add_callback(progress_callback)
                    # Call the *original* method to avoid recursion
                    return original_kto(*args, **kwargs)

                agent.kto = kto_with_progress  # type: ignore
                agent.kto(train_data, kto_config, iter_idx, checkpoint_path)
                
        except Exception as e:
            print(f"{agent_prefix} |  Training failed: {e}")
            raise
        finally:
            # Comprehensive cleanup
            print(f"{agent_prefix} |  Starting comprehensive cleanup...")
            try:
                if 'agent' in locals():
                    agent.cleanup()
                cleanup_gpu()
                if 'wandb' in locals() and wandb.run:
                    wandb.finish()
                print(f"{agent_prefix} |  Cleanup completed successfully")
            except Exception as cleanup_error:
                print(f"{agent_prefix} |  Cleanup warning: {cleanup_error}")
            
            print(f"{agent_prefix} | Final subprocess cleanup...")

def main() -> int:
    """Main function for subprocess agent training."""
    try:
        args = parse_args()
    except Exception as e:
        print(f" Failed to parse args: {e}")
        return 1
    
    # Create agent prefix for consistent identification
    agent_prefix = f" Agent-{args.agent_idx}"
    
    # Set up GPU environment FIRST (before any CUDA operations)
    try:
        setup_gpu_environment(args.gpu_id)
    except Exception as e:
        print(f" GPU setup failed: {e}")
        return 1
    
    print(f"{agent_prefix} |  Starting {args.phase.upper()} training on GPU {args.gpu_id}")
    
    try:
        # Load configuration
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        
        # Load debate data
        with open(args.debate_data_file, 'r') as f:
            debate_data = json.load(f)
        
        # Load precomputed rewards if provided
        precomputed_rewards = None
        if args.rewards_file and os.path.exists(args.rewards_file):
            with open(args.rewards_file, 'r') as f:
                precomputed_rewards = json.load(f)
        
        # Create minimal orchestrator
        orchestrator = MinimalOrchestrator(config)
        
        # Run training
        orchestrator.train_agent_isolated(
            agent_idx=args.agent_idx,
            debate_data=debate_data,
            iter_idx=args.iter_idx,
            phase=args.phase,
            precomputed_rewards=precomputed_rewards
        )
        
        print(f"{agent_prefix} |  Training completed successfully")
        return 0
        
    except Exception as e:
        print(f"{agent_prefix} |  Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Final cleanup
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Final CUDA cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        except Exception as cleanup_error:
            print(f"{agent_prefix} |  Cleanup warning: {cleanup_error}")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 