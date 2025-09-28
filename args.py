"""Command line argument parsing for multi-agent debate framework."""

import argparse
from typing import Dict, Any

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for multi-agent debate training."""
    parser = argparse.ArgumentParser(description="Multi-agent reinforcement learning from human feedback framework")
    
    # ===== Core system configuration =====
    parser.add_argument('--model', action='store', default="phi4b", type=str, 
                       choices=["mistral7b", "llama1b", "llama3b", "llama8b", "phi4b", "qwen7b", "qwen2b", "gemma4b"],
                       help="Base model to use for agents")
    parser.add_argument('--dataset', action='store', default='gsm8k', type=str, dest="dataset", 
                       choices=['gsm8k', 'arithmatic', 'math', 'aime_amc', 'amc_aime', 'gpqa', 'svamp', 'mathqa', 'csqa'],
                       help="Dataset to use for training and evaluation")
    parser.add_argument('--train_size', action='store', default=500, type=int, dest="train_size",
                       help="Number of examples to use for training")
    parser.add_argument('--test_size', action='store', default=500, type=int, dest="test_size",
                       help="Number of examples to use for testing")
    parser.add_argument('--iterations', action='store', default=1, type=int, dest="iterations", 
                       help="Number of training iterations")
    parser.add_argument('--seed', action='store', type=int, dest='seed', default=0, 
                       help="Random seed for reproducibility - used for model training")
    parser.add_argument('--data_seed', action='store', type=int, dest='data_seed', default=0, 
                       help="Random seed for dataset sampling and debate generation")
    parser.add_argument('--use_quantization', action='store_true', dest="use_quantization",
                       help="Enable model quantization to reduce memory usage")
    parser.add_argument('--use_adapter_mode', action='store_true', dest="use_adapter_mode",
                       help="Use adapter swapping mode for more efficient memory usage (requires --use_quantization)")
    parser.add_argument('--use_scheduler', action='store_true', dest="use_scheduler",
                       help="Use dynamic job scheduler for adapter swapping (requires --use_adapter_mode and --use_quantization)")
    parser.add_argument('--max_concurrent_tasks', action='store', default=8, type=int, dest="max_concurrent_tasks",
                       help="Maximum number of concurrent generation tasks for scheduler")
    parser.add_argument('--experiment_dir', action='store', default='', type=str, dest="experiment_dir")

    # ===== Multi-agent configuration =====
    parser.add_argument('--agents', action='store', default=3, type=int, dest="agents", 
                       help="Number of agents for debate")
    parser.add_argument('--max_agents_per_device', action='store', default=2, type=int, dest="max_agents_per_device",
                       help="Maximum number of agents to allocate per GPU device")
    parser.add_argument('--gpus_per_model', action='store', default=1, type=int, dest="gpus_per_model",
                       help="Number of GPUs to allocate for each base model instance (for large models)")
    parser.add_argument('--exclude_gpus', type=int, nargs='*', default=[], dest="exclude_gpus",
                       help="GPU IDs to exclude from training (e.g., --exclude_gpus 0 3 to exclude GPUs 0 and 3)")
    parser.add_argument('--gpu_allocation_timeout', action='store', default=60.0, type=float, dest="gpu_allocation_timeout",
                       help="Timeout in seconds for GPU allocation requests")
    parser.add_argument('--use_parallel_training', action='store_true', dest="use_parallel_training",
                       help="Enable parallel agent training with process isolation (recommended for multi-GPU setups)")
    
    # ===== Text generation parameters =====
    parser.add_argument('--temperature', action='store', default=1, type=float, dest="temperature",
                       help="Sampling temperature for text generation")
    parser.add_argument('--top_p', action='store', default=0.9, type=float, dest="top_p",
                       help="Nucleus sampling parameter for text generation")
    parser.add_argument('--use_vllm', action='store_true', dest="use_vllm", 
                       help="Enable vLLM for faster generation in GRPO")
    
    # ===== Debate configuration =====
    parser.add_argument('--rounds', action='store', default=2, type=int, dest="rounds", 
                       help="Number of rounds for debate")
    parser.add_argument('--batch_debate', action='store', default=8, type=int, dest="batch_debate", 
                       help="Batch size for debate")
    parser.add_argument('--summarize', action='store_true', dest="summarize",
                       help="Enable summarization of other agents' responses")
    parser.add_argument('--diversity_prompt', action='store_true', dest="diversity_prompt", 
                       help="Add explicit diversity prompt to encourage different solutions")
    parser.add_argument('--use_majority_vote', action='store_true', dest="use_majority_vote",
                       help="Use majority voting to determine consensus answers")
    parser.add_argument('--use_ground_truth', action='store_true', dest="use_ground_truth",
                       help="Use ground truth answers instead of majority vote (overrides --use_majority_vote)")
    parser.add_argument('--include_debate_context', action='store_true', dest="include_debate_context",
                       help="Include debate context in RL training data")
    parser.add_argument('--no_context', action='store_true', dest="no_context",
                       help="Disable debate context (overrides --include_debate_context)")
    parser.add_argument('--use_async_debate', action='store_true', dest="use_async_debate",
                       help="Use asyncio-based debate implementation instead of distributed multiprocessing")
    parser.add_argument('--test_only', action='store_true', dest="test_only",
                       help="Skip training set evaluation in final debate rounds (only evaluate on test set after training)")
    parser.add_argument('--train_only', action='store_true', dest="train_only",
                       help="Skip test set evaluation in final debate rounds (only evaluate on training set after training)")
    
    # ===== Supervised fine-tuning (SFT) parameters =====
    parser.add_argument('--finetune', action='store_true', dest="finetune",
                       help="Enable supervised fine-tuning")
    parser.add_argument('--finetune_epoch', action='store', default=1, type=int, dest="epoch_sft",
                       help="Number of epochs for supervised fine-tuning")
    parser.add_argument('--finetune_batch_size', action='store', default=1, type=int, dest="batch_size_sft",
                       help="Batch size for supervised fine-tuning")
    parser.add_argument('--finetune_lr', action='store', default=5e-7, type=float, dest="lr_sft",
                       help="Learning rate for supervised fine-tuning")
    parser.add_argument('--finetune_weight_decay', action='store', default=1e-2, type=float, dest="weight_decay_sft",
                       help="Weight decay for supervised fine-tuning (1e-3 for Phi-3)")
    
    # ===== LoRA configuration parameters =====
    parser.add_argument('--lora_r', action='store', default=8, type=int, dest="lora_r",
                       help="LoRA attention dimension")
    parser.add_argument('--lora_alpha', action='store', default=32, type=int, dest="lora_alpha",
                       help="LoRA alpha parameter")
    parser.add_argument('--lora_dropout', action='store', default=0.1, type=float, dest="lora_dropout",
                       help="LoRA dropout probability")
    parser.add_argument('--lora_target_modules', action='store', default=["q_proj", "k_proj", "v_proj", "o_proj"], 
                       type=str, nargs='+', dest="lora_target_modules",
                       help="Target modules for LoRA adaptation")
    
    # ===== Reinforcement learning (RL) parameters =====
    parser.add_argument('--post_train', action='store_true', dest="post_train",
                       help="Enable reinforcement learning post-training")
    parser.add_argument('--post_train_lr', action='store', default=5e-6, type=float, dest="lr_rl",
                       help="Learning rate for reinforcement learning")
    parser.add_argument('--post_train_weight_decay', action='store', default=1e-2, type=float, dest="weight_decay_rl",
                       help="Weight decay for reinforcement learning")
    parser.add_argument('--epoch_rl', action='store', type=int, dest="epoch_rl",
                       help="Number of epochs for reinforcement learning")
    parser.add_argument('--batch_rl', action='store', default=16, type=int, dest="batch_rl",
                       help="Batch size for reinforcement learning")
    parser.add_argument('--consensus_batch_size', action='store', default=8, type=int, dest="consensus_batch_size",
                       help="Batch size for consensus reward computation (should be <= batch_rl for memory efficiency)")
    parser.add_argument('--entropy_coef', action='store', default=0.01, type=float, dest="entropy_coef",
                       help="Entropy coefficient for reinforcement learning")
    parser.add_argument('--gradient_accumulation_steps', action='store', default=4, type=int, dest="gradient_accumulation_steps", 
                       help="Number of gradient accumulation steps")
    parser.add_argument('--use_peft', action='store_true', dest="peft",
                       help="Enable Parameter-Efficient Fine-Tuning (PEFT)")
    
    # ===== Reward modeling =====
    parser.add_argument('--verifiable_reward', action='store_true', dest="verifiable_reward", 
                       help="Use 0.0 instead of -1.0 for incorrect answers")
    parser.add_argument('--use_format_reward', action='store_true', dest="use_format_reward", 
                       help="Use format reward functions")
    parser.add_argument('--num_generations', action='store', default=8, type=int, dest="num_generations",
                       help="Number of generations for reward training")
    
    # ===== Judge model parameters =====
    parser.add_argument('--judge_model', action='store', default="qwen2b", type=str, 
                       choices=["qwen7b", "qwen2b", "phi4b"], 
                       help="Model to use as judge for self-play")
    parser.add_argument('--judge_batch_size', action='store', default=4, type=int, dest="judge_batch_size", 
                       help="Batch size for judge model evaluations")
    
    # ===== Logging and experiment tracking =====
    parser.add_argument('--wandb', action='store_true', dest="wandb",
                       help="Enable Weights & Biases logging")
    parser.add_argument('--project_name', action='store', default="llm-marl", type=str, dest="project_name",
                       help="Project name for Weights & Biases logging")
    parser.add_argument('--entity_name', action='store', default="llm-marl", type=str, dest="entity_name",
                       help="Entity/team name for Weights & Biases logging")
    
    # ===== Transfer function parameters =====
    parser.add_argument('--use_consensus_reward', action='store_true', dest="use_consensus_reward",
                       help="Use consensus reward function")
    
    # ===== Kahneman-Tversky Optimization (KTO) parameters =====
    parser.add_argument('--kto', action='store_true', dest='kto',
                       help='Enable KTO preference fine-tuning phase')
    parser.add_argument('--epoch_kto', action='store', default=3, type=int, dest='epoch_kto',
                       help='Number of training epochs for KTO')
    parser.add_argument('--batch_kto', action='store', default=8, type=int, dest='batch_kto',
                       help='Per-device batch size for KTO')
    parser.add_argument('--lr_kto', action='store', default=5e-6, type=float, dest='lr_kto',
                       help='Learning rate for KTO')
    parser.add_argument('--beta_kto', action='store', default=0.1, type=float, dest='beta_kto',
                       help='Inverse-temperature β for KTO loss')
    parser.add_argument('--gradient_accumulation_steps_kto', action='store', default=1, type=int, dest='gradient_accumulation_steps_kto',
                       help='Gradient accumulation steps for KTO')
    parser.add_argument('--train_size_kto', action='store', default=-1, type=int, dest='train_size_kto',
                       help='Optional cap on preference records per agent for KTO (<=0 ⇒ use all)')
    parser.add_argument('--desirable_weight', action='store', default=1.0, type=float, dest='desirable_weight',
                       help='Loss weight on desirable (positive) examples in KTO')
    parser.add_argument('--undesirable_weight', action='store', default=1.0, type=float, dest='undesirable_weight',
                       help='Loss weight on undesirable (negative) examples in KTO')
    parser.add_argument('--pref_debate_context', action='store_true', dest='pref_debate_context',
                       help='Use full second-to-last round context for preference datasets (overrides --include_debate_context)')
    
    # ===== Direct Preference Optimization (DPO) parameters =====
    parser.add_argument('--dpo', action='store_true', dest='dpo',
                       help='Enable DPO preference fine-tuning phase')
    parser.add_argument('--epoch_dpo', action='store', default=3, type=int, dest='epoch_dpo',
                       help='Number of training epochs for DPO')
    parser.add_argument('--batch_dpo', action='store', default=8, type=int, dest='batch_dpo',
                       help='Per-device batch size for DPO')
    parser.add_argument('--lr_dpo', action='store', default=5e-6, type=float, dest='lr_dpo',
                       help='Learning rate for DPO')
    parser.add_argument('--beta_dpo', action='store', default=0.1, type=float, dest='beta_dpo',
                       help='Inverse-temperature β for DPO loss')
    parser.add_argument('--gradient_accumulation_steps_dpo', action='store', default=1, type=int, dest='gradient_accumulation_steps_dpo',
                       help='Gradient accumulation steps for DPO')
    parser.add_argument('--train_size_dpo', action='store', default=-1, type=int, dest='train_size_dpo',
                       help='Optional cap on preference records per agent for DPO (<=0 ⇒ use all)')
    
    # ===== Set all default configurations in a single call =====
    parser.set_defaults(
        # Active defaults
        use_majority_vote=True, # set to False to use ground truth labels
        verifiable_reward=True,  # use 0.0 instead of -1.0 for incorrect answers
        use_format_reward=True, # set to False to disable format reward function
        include_debate_context=True, # set to True to include debate context in RL training
        pref_debate_context=False,  # Use full second-to-last round context in preference data
        
        # Training Method
        summarize=False, # set to True to enable summarization of other agents' responses
        finetune=False, # set to True to enable supervised fine-tuning
        post_train=False, # set to True to enable reinforcement learning post-training
        kto=False,
        dpo=False,
        
        # Performance
        use_vllm=False, # set to True to use vLLM for faster generation in GRPO
        use_async_debate=True, # set to True to use asyncio-based debate implementation
        use_scheduler=True, # set to True to use scheduler-based adapter swapping
        # max_concurrent_tasks=6, # maximum concurrent generation tasks for scheduler
        use_adapter_mode=True, # set to True to use adapter swapping mode (requires quantization)
        use_quantization=True,
        use_parallel_training=True, # set to True to enable parallel agent training with process isolation
        
        use_consensus_reward=True,
        test_only=False,
        train_only=False,        

    )
    
    # Handle use_ground_truth override
    args = parser.parse_args()
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

    return args
