
"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""Multi-agent debate system with distributed processing and automatic GPU allocation."""

import os
import sys
import json
import random
import numpy as np
import torch.multiprocessing as mp
from typing import List, Dict, Any, Union, Optional, Tuple, Set, Callable
import torch
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from model import Agent
from parser import grade_answer, parse_answer
from utils import get_agent_device_assignment, cleanup_gpu, set_random_seed
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import wandb
import time
from datetime import datetime
from scheduler import AdapterJobScheduler

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


def construct_assistant_message(
    completion: Dict[str, Any]
) -> Dict[str, str]:
    """Convert model completion to assistant message format."""
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}

def construct_question_prompt(
    question: str,
    dataset: str
) -> str:
    """Construct question prompt with dataset-specific formatting instructions."""
    # Default format instructions
    format_instructions = (
        "\n Provide a bullet point summary of your step-by-step reasoning."
        "\n {answer_type}"
    )
    
    if dataset in ['gpqa', 'mathqa', 'csqa']:
        answer_type = "Your final answer should be a single choice letter in the form \\boxed{{answer}}, at the end of your response."
        instruction = f"Answer the following multiple choice question as accurately as possible. {question} "
    else:
        answer_type = "Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."
        instruction = f"Solve the following math problem. {question} "
    # Build the complete prompt
    return instruction + format_instructions.format(answer_type=answer_type)


def construct_message(
    agents: List[List[Dict[str, str]]],
    question: str,
    dataset: str,
    summary: Optional[str],
    idx: int,
    diversity_prompt: bool
) -> Dict[str, str]:
    """
    Construct a message for an agent with feedback from other agents.
    
    Args:
        agents: List of agent response contexts
        question: Original question
        dataset: Dataset identifier
        summary: Optional summary of other agents' responses
        idx: Index of the response to use from each agent
        diversity_prompt: Whether to include a diversity prompt
        
    Returns:
        Dictionary with role and content for the message
    """
    if summary:
        content = f"Here is a summary of solutions by other agents: {summary}"
        content += f"\n Using this summary as additional advice, give an updated bullet point summary of your step-by-step reasoning to the question: {question}"
    else:
        content = "Here are solutions from other agents: "
        for agent in agents:
            content += f"\n\n One agent responsed as follows: {agent[idx]['content']}"
        content += f"\n Using each response as additional advice, give an updated bullet point summary of your step-by-step reasoning to the question: {question}"

    if diversity_prompt:
        content += "\n Your solution should arrive to the correct answer using a different method compared to other agents."
    if dataset in ['gpqa', 'mathqa', 'csqa']:
        content += "\n Make sure your final answer is a single choice letter in the form \\boxed{{answer}}, at the end of your response."
    else:
        content += "\n Make sure your final answer is in the form \\boxed{{answer}}, at the end of your response."
    return {"role": "user", "content": content}


def majority_vote(
    answers: List[str]
) -> str:
    """Return most common answer using grade_answer for equivalence checks."""
    answer_set, counts = [], []
    for answer in answers:
        is_match = False
        for i, candidate_answer in enumerate(answer_set):
            if grade_answer(candidate_answer, answer):
                is_match = True
                counts[i] = counts[i] + 1
                break
        if not is_match:
            answer_set.append(answer)
            counts.append(1)
    responses = sorted(zip(counts, answer_set))
    return responses[-1][1]


    """
    Compute accuracy metrics from debate results.
    
    Args:
        debate_data: Dictionary of debate results
        config: Configuration dictionary
        
    Returns:
        Dictionary of metrics including train and test accuracy
    """
    # --------- consensus-level metrics (conditional on presence) ---------
    train_correct_present, test_correct_present = 0, 0
    train_present_cnt, test_present_cnt = 0, 0

    # --------- individual-agent tallies ---------
    num_agents = config.get('agents', 3)
    agent_stats = {
        f"agent_{i}": {
            'train_correct_present': 0,
            'test_correct_present': 0,
            'train_responses': 0,
            'test_responses': 0
        } for i in range(num_agents)
    }

    # helper to grade a single agent answer given raw structures
    def _extract_agent_answer(example: Dict[str, Any], agent_idx: int) -> Tuple[bool, bool]:
        """Return (response_present, correct_flag)"""
        # binary format fast path
        if 'agent_answers' in example and isinstance(example['agent_answers'], list):
            ans_list = example['agent_answers']
            if agent_idx < len(ans_list):
                correct_flag = bool(ans_list[agent_idx])
                present_flag = True  # binary list implies response attempted
                return present_flag, correct_flag
            return False, False
        # old format: inspect context
        if 'context' in example and len(example['context']) > agent_idx:
            ctx = example['context'][agent_idx]
            if isinstance(ctx, list) and len(ctx) > 1:
                # assume final assistant message contains answer
                final_msg = ctx[-1]
                if final_msg.get('role') == 'assistant' and final_msg.get('content'):
                    present_flag = True
                    parsed = parse_answer(final_msg['content'], dataset=config.get('dataset', 'gsm8k'))
                    correct_flag = grade_answer(parsed, example['ground_truth'])
                    return present_flag, correct_flag
        return False, False

    # iterate examples
    for k, v in debate_data.items():
        if k == 'metrics' or not isinstance(v, dict):
            continue

        split = v.get('split', 'train')
        is_correct_consensus = grade_answer(v['consensus_answer'], v['ground_truth'])

        if split == 'train':
            train_present_cnt += 1
            if is_correct_consensus:
                train_correct_present += 1
        else:
            test_present_cnt += 1
            if is_correct_consensus:
                test_correct_present += 1

        # per-agent loop
        for i in range(num_agents):
            present, correct = _extract_agent_answer(v, i)
            stats = agent_stats[f"agent_{i}"]
            if split == 'train':
                if present:
                    stats['train_responses'] += 1
                if correct:
                    stats['train_correct_present'] += 1
            else:
                if present:
                    stats['test_responses'] += 1
                if correct:
                    stats['test_correct_present'] += 1

    # denominators from config (full counts)
    total_train_expected = config.get('train_size', train_present_cnt)
    total_test_expected = config.get('test_size', test_present_cnt)

    # conditional means
    train_mean_present = train_correct_present / train_present_cnt if train_present_cnt else 0.0
    test_mean_present = test_correct_present / test_present_cnt if test_present_cnt else 0.0

    # full means (penalise missing)
    train_mean_full = train_correct_present / total_train_expected if total_train_expected else 0.0
    test_mean_full = test_correct_present / total_test_expected if total_test_expected else 0.0

    # std over Bernoulli trials (present-only) for compatibility
    train_std = np.sqrt(train_mean_present*(1-train_mean_present))/np.sqrt(train_present_cnt) if train_present_cnt>1 else 0.0
    test_std = np.sqrt(test_mean_present*(1-test_mean_present))/np.sqrt(test_present_cnt) if test_present_cnt>1 else 0.0

    metrics = {
        'train_acc': {
            'mean': f"{train_mean_present*100:.2f}",
            'std': f"{train_std*100:.2f}",
            'fmean': f"{train_mean_full*100:.2f}",
            'fstd': f"{(np.sqrt(train_mean_full*(1-train_mean_full))/np.sqrt(total_train_expected) if total_train_expected>1 else 0)*100:.2f}"
        },
        'test_acc': {
            'mean': f"{test_mean_present*100:.2f}",
            'std': f"{test_std*100:.2f}",
            'fmean': f"{test_mean_full*100:.2f}",
            'fstd': f"{(np.sqrt(test_mean_full*(1-test_mean_full))/np.sqrt(total_test_expected) if total_test_expected>1 else 0)*100:.2f}"
        },
    }

    # add per-agent dictionary
    agent_perf_dict = {}
    for i in range(num_agents):
        s = agent_stats[f"agent_{i}"]
        ignore_train = s['train_correct_present'] / s['train_responses'] if s['train_responses'] else 0.0
        ignore_test = s['test_correct_present'] / s['test_responses'] if s['test_responses'] else 0.0
        all_train = s['train_correct_present'] / total_train_expected if total_train_expected else 0.0
        all_test = s['test_correct_present'] / total_test_expected if total_test_expected else 0.0
        agent_perf_dict[f"agent_{i}"] = {
            'train_ignore': ignore_train,
            'train_all': all_train,
            'test_ignore': ignore_test,
            'test_all': all_test,
            'train_correct': s['train_correct_present'],
            'test_correct': s['test_correct_present'],
            'train_responses': s['train_responses'],
            'test_responses': s['test_responses']
        }

    metrics['agent_performance'] = agent_perf_dict

    return metrics


def save_debate_results(
    debate_data: Dict[str, Any],
    config: Dict[str, Any],
    iter_idx: int
) -> None:
    """
    Save debate results to the standard location.
    
    Args:
        debate_data: Dictionary of debate results
        config: Configuration dictionary
        iter_idx: Iteration index
    """
    os.makedirs(f"{config['experiment_dir']}/debate", exist_ok=True)
    with open(f"{config['experiment_dir']}/debate/debate_iteration_{iter_idx}.json", "w") as f:
        json.dump(debate_data, f)


async def distributed_multiagent_debate(
    data: List[Dict[str, Any]],
    config: Dict[str, Any],
    iter_idx: int
) -> Dict[str, Any]:
    """Run multi-agent debate with automatic GPU allocation and agent lifecycle management."""
    print(f"Starting optimized debate with {len(data)} examples")
    start_time = time.time()
    
    num_agents = config['agents']
    devices = config['devices']
    batch_size = config['batch_debate']
    
    # Calculate optimal number of batches
    num_batches = (len(data) + batch_size - 1) // batch_size
    print(f"Processing dataset in {num_batches} batches of up to {batch_size} examples")
    
    # Check if we're using adapter mode (requires quantization)
    adapter_mode = config.get('use_adapter_mode', False) and config.get('use_quantization', False)
    if config.get('use_adapter_mode', False) and not config.get('use_quantization', False):
        print("Warning: Adapter mode requires quantization. Falling back to normal mode.")
        adapter_mode = False
    
    if adapter_mode:
        print(f"Using adapter swapping mode with quantization")
    else:
        print(f"Using regular model loading mode")
    
    # Initialize agents - ONLY ONCE, using auto device mapping
    print(f"Initializing {num_agents} agents with automatic GPU allocation")
    agent_init_start = time.time()
    checkpoint_path = f"{config['experiment_dir']}/checkpoints/agent"
    
    agents = [
        Agent(
            config['model'],
            agent_id=i,
            device_map='auto',  # Let the framework optimize GPU allocation
            checkpoint_path=f"{checkpoint_path}_{i}" if os.path.exists(f"{checkpoint_path}_{i}") else None,
            seed=config['data_seed'],
            quantization=config['use_quantization'],
            adapter_mode=adapter_mode,
            task=config['dataset'],
        ) for i in range(num_agents)
    ]
    
    # Log agent initialization
    for i, agent in enumerate(agents):
        print(f"Agent {i} initialized with automatic device mapping")
        # Get the primary device for logging
        agent_device = agent.get_device()
        print(f"Agent {i} primary device: {agent_device}")
    
    # Log checkpoint paths to match original implementation
    for i in range(num_agents):
        path = f"{checkpoint_path}_{i}"
        exists = os.path.exists(path)
        print(f"Checkpoint {path}: {exists}")
    
    print(f"Agent initialization took {time.time() - agent_init_start:.2f} seconds")

    # Process all data through single debate
    # Use debate_idx=0 to match original implementation's file naming
    debate_results = await optimized_multiagent_debate(
            data=data,
            config=config,
        agents=agents,
        iter_idx=iter_idx,
        debate_idx=0
    )
    
    # Mimic the original implementation's result consolidation
    debate_dir = f"{config['experiment_dir']}/debate"
    os.makedirs(debate_dir, exist_ok=True)
    
    # Load the saved interim result file
    result_path = f"{debate_dir}/iteration_{iter_idx}_index_0.json"
    assert os.path.exists(result_path), f"Debate result file {result_path} does not exist possibly due to OOM. Use a smaller debate batch size."
    with open(result_path, 'r') as f:
        debate_results = json.load(f)

    # Save consolidated results
    save_debate_results(debate_results, config, iter_idx)
    
    # Cleanup
    cleanup_start = time.time()
    for agent in agents:
        del agent
    cleanup_gpu()
    print(f"Cleanup took {time.time() - cleanup_start:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"Debate completed in {total_time:.2f} seconds")
    print(f"Average time per example: {total_time/len(data):.2f} seconds")
    
    return debate_results

async def optimized_multiagent_debate(
    data: List[Dict[str, Any]],
    config: Dict[str, Any],
    agents: List[Agent],
    iter_idx: int,
    debate_idx: int
) -> Dict[str, Any]:
    """
    Run optimized multi-agent debate with automatic GPU allocation.
    
    Args:
        data: List of examples to debate on
        config: Configuration dictionary
        agents: List of pre-initialized agents
        iter_idx: Iteration index
        debate_idx: Debate process index
        
    Returns:
        Dictionary containing debate results
    """
    start_time = time.time()
    print(f"Starting optimized debate processing for {len(data)} examples")
    
    num_agents = len(agents)
    num_rounds = config['rounds']
    temperature, top_p = config['temperature'], config['top_p']
    batch_size = config['batch_debate']
    diversity_prompt = config['diversity_prompt']

    debate = {}
    
    # Calculate total batches for progress bar
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    # Add main progress bar for batches
    batch_pbar = tqdm(total=total_batches, desc="Debate Batches", unit="batch")
    
    # Process data in batches
    for batch_idx, batch_start in enumerate(range(0, len(data), batch_size)):
        batch_end = min(batch_start + batch_size, len(data))
        batch_data = data[batch_start:batch_end]
        num_examples_in_batch = len(batch_data)
        
        batch_start_time = time.time()
        batch_pbar.set_description(f"Batch {batch_idx + 1}/{total_batches} ({num_examples_in_batch} examples)")
        
        batch_questions = [example['question'] for example in batch_data]
        batch_answers = [example['answer'] for example in batch_data]
        
        # Initialize contexts for all examples and all agents
        context_init_start = time.time()
        # Structure: [example_idx][agent_idx][message_idx]
        batch_agent_contexts = [
            [
                [{"role": "user", "content": construct_question_prompt(q, config['dataset'])}]
                for _ in range(num_agents)
            ] 
            for q in batch_questions
        ]
        # print(f"Context initialization took {time.time() - context_init_start:.2f} seconds")
        
        # Add progress bar for rounds within this batch
        round_pbar = tqdm(total=num_rounds, desc="  ↳ Rounds", unit="round", leave=False)
        
        # Run debate rounds
        for round_idx in range(num_rounds):
            round_start = time.time()
            round_pbar.set_description(f"  ↳ Round {round_idx + 1}/{num_rounds}")
            
            # Add progress bar for agents within this round
            agent_pbar = tqdm(total=num_agents, desc="    ↳ Agents", unit="agent", leave=False)
            
            # For each round, process all agents (potentially in parallel)
            agent_tasks = []
            
            for agent_idx, agent in enumerate(agents):
                # Get the agent's primary device for generation
                device = agent.get_device()
                
                # Prepare contexts for this agent for all examples
                if round_idx > 0:
                    context_prep_start = time.time()
                    
                    # Prepare contexts with other agents' previous responses
                    agent_contexts_other_batch = []
                    for q_idx in range(num_examples_in_batch):
                        # Get responses from other agents for this example
                        agent_contexts_other = batch_agent_contexts[q_idx][:agent_idx] + batch_agent_contexts[q_idx][agent_idx+1:]
                        random.shuffle(agent_contexts_other)
                        agent_contexts_other = agent_contexts_other[:5]  # Take up to 5 agents
                        agent_contexts_other_batch.append(agent_contexts_other)

                    # Optionally summarize other agents' responses
                    if config['summarize']:
                        summaries = await batch_summarize_responses(
                            agent_contexts_other_batch,
                            agent,
                            device,
                            temperature,
                            top_p
                        )
                        summaries = [summary for summary in summaries]  # Match original implementation
                    else:
                        summaries = [None] * num_examples_in_batch
                    
                    # Create messages with feedback for each example
                    for q_idx in range(num_examples_in_batch):
                        message = construct_message(
                            agent_contexts_other_batch[q_idx],
                            batch_questions[q_idx],
                            config['dataset'],
                            summaries[q_idx] if config['summarize'] else None,
                            2 * round_idx - 1,
                            diversity_prompt
                        )
                        batch_agent_contexts[q_idx][agent_idx].append(message)

                    # print(f"Context preparation for agent {agent_idx} took {time.time() - context_prep_start:.2f} seconds")
                
                # Collect all contexts for this agent across all examples
                contexts_for_agent = [ctx[agent_idx] for ctx in batch_agent_contexts]
                
                # Create task for batch generation
                task = process_agent_batch(
                    agent=agent,
                    agent_idx=agent_idx,
                    contexts=contexts_for_agent,
                    device=device,
                    temperature=temperature,
                    top_p=top_p,
                    round_idx=round_idx
                )
                agent_tasks.append(task)
            
            # Wait for all agents to complete their responses
            batch_results = await asyncio.gather(*agent_tasks)
            
            # Update contexts with new responses and progress
            for agent_idx, completions in enumerate(batch_results):
                for q_idx, completion in enumerate(completions):
                    assistant_message = construct_assistant_message(completion)
                    batch_agent_contexts[q_idx][agent_idx].append(assistant_message)
                agent_pbar.update(1)
                agent_pbar.set_description(f"    Agent {agent_idx} [COMPLETE]")

            agent_pbar.close()
            round_pbar.update(1)
            print(f"Round {round_idx + 1} completed in {time.time() - round_start:.2f} seconds")
        
        round_pbar.close()
        
        # Process results for this batch
        result_start = time.time()
        # print("Processing debate results")
        
        for q_idx, (question, ground_truth) in enumerate(zip(batch_questions, batch_answers)):
            answers = []
            agent_contexts = batch_agent_contexts[q_idx]

            # Extract answers from each agent
            for agent_context in agent_contexts:
                answer = parse_answer(agent_context[-1]['content'], config['dataset'])
                if answer is not None:
                    answers.append(answer)

            if len(answers) == 0:
                continue

            # Determine consensus answer
            consensus_answer = majority_vote(answers)

            # Score agents against consensus
            agent_answers = []
            for agent_context in agent_contexts:
                answer = parse_answer(agent_context[-1]['content'], config['dataset'])
                agent_answers.append(1 if grade_answer(answer, consensus_answer) else 0)

            # Store results
            debate[question] = {
                'context': agent_contexts,
                'consensus_answer': consensus_answer,
                'agent_answers': agent_answers,
                'ground_truth': ground_truth,
                'answer_cot': batch_data[q_idx]['answer_cot'],
                'split': batch_data[q_idx]['split'],
            }

        batch_pbar.update(1)
        batch_elapsed = time.time() - batch_start_time
        batch_pbar.set_postfix({"Time": f"{batch_elapsed:.1f}s", "Avg": f"{batch_elapsed/num_examples_in_batch:.1f}s/ex"})
        # print(f"Result processing took {time.time() - result_start:.2f} seconds")
        # print(f"Batch {batch_idx + 1} completed in {batch_elapsed:.2f} seconds")
    
    batch_pbar.close()
    
    # Save interim results with same naming as original implementation
    save_start = time.time()
    debate_dir = f"{config['experiment_dir']}/debate"
    os.makedirs(debate_dir, exist_ok=True)
    
    # Save with the original interim file naming format
    interim_result_path = f"{debate_dir}/iteration_{iter_idx}_index_{debate_idx}.json"
    with open(interim_result_path, "w") as f:
        json.dump(debate, f)
    # print(f"Interim results saved to {interim_result_path} in {time.time() - save_start:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"Total processing completed in {total_time:.2f} seconds")
    # print(f"Average time per example: {total_time/len(data):.2f} seconds")
    
    return debate

async def process_agent_batch(
    agent: Agent,
    agent_idx: int,
    contexts: List[List[Dict[str, str]]],
    device: torch.device,
    temperature: float,
    top_p: float,
    round_idx: int
) -> List[Dict]:
    """
    Process a batch of contexts for a single agent.
    
    Args:
        agent: The agent to generate responses
        agent_idx: Index of the agent
        contexts: List of contexts for each example
        device: Device to run generation on
        temperature: Temperature for generation
        top_p: Top-p for generation
        round_idx: Current round index
    
    Returns:
        List of completions for each context
    """
    generation_start = time.time()
    # print(f"Generating responses for agent {agent_idx} (round {round_idx + 1})")
    
    # Generate responses for all contexts in one batch
    batch_completions = await agent.batch_generate(contexts, device, top_p, temperature)
    
    generation_time = time.time() - generation_start
    # print(f"Agent {agent_idx} batch generation took {generation_time:.2f} seconds for {len(contexts)} examples")
    # print(f"Average time per example: {generation_time/len(contexts):.2f} seconds")
    
    return batch_completions

async def batch_summarize_responses(
    agent_contexts_list: List[List[List[Dict[str, str]]]],
    agent: Agent,
    device: torch.device,
    temperature: float = 1.0,
    top_p: float = 0.9
) -> List[str]:
    """
    Generate summaries for multiple sets of agent responses in a single batch.
    
    Args:
        agent_contexts_list: List of agent contexts for each example
        agent: Agent to use for summarization
        device: Device to run summarization on
        temperature: Temperature for generation
        top_p: Top-p for generation
        
    Returns:
        List of summary strings
    """
    summarization_start = time.time()
    # print(f"Generating batch summaries for {len(agent_contexts_list)} examples")
    
    # Prepare prompts for summarization
    summary_prompts = []
    for agent_contexts in agent_contexts_list:
        prompt = "Here are a list of opinions from different agents: "
        for agent_ctx in agent_contexts:
            prompt += f"\n\n One agent response: ```{agent_ctx[-1]['content']}```"
        prompt += "\n\n Write a precise and concise step by step reasoning of each answer. You are allowed to summarize each response to maximum of three sentences."
        summary_prompts.append([{"role": "user", "content": prompt}])
    
    # Generate all summaries in a single batch
    completions = await agent.batch_generate(summary_prompts, device, top_p, temperature)
    
    # Extract summary content
    summaries = [completion["choices"][0]["message"]["content"] for completion in completions]
    
    # print(f"Batch summarization completed in {time.time() - summarization_start:.2f} seconds")
    # print(f"Average time per summary: {(time.time() - summarization_start)/len(agent_contexts_list):.2f} seconds")
    
    return summaries

async def scheduled_multiagent_debate(
    data: List[Dict[str, Any]],
    config: Dict[str, Any],
    iter_idx: int
) -> Dict[str, Any]:
    """
    Run multi-agent debate using the adapter job scheduler for efficient resource utilization.
    
    Args:
        data: List of examples to debate on
        config: Configuration dictionary
        iter_idx: Iteration index
        
    Returns:
        Dictionary containing debate results
    """
    start_time = time.time()
    
    num_agents = config['agents']
    batch_size = config['batch_debate']
    
    # Calculate number of batches
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    # Check if we're using adapter mode (requires quantization)
    adapter_mode = config.get('use_adapter_mode', False) and config.get('use_quantization', False)
    if config.get('use_adapter_mode', False) and not config.get('use_quantization', False):
        print("Warning: Adapter mode requires quantization. Falling back to normal mode.")
        adapter_mode = False
    
    if not adapter_mode:
        print("Warning: Scheduled debate requires adapter mode with quantization. Falling back to regular debate.")
        return await distributed_multiagent_debate(data, config, iter_idx)
        
    # Initialize scheduler
    scheduler = AdapterJobScheduler(config)
    await scheduler.start()
    
    # Process data with scheduler
    debate_results = await scheduled_debate_processing(
        data=data,
        config=config,
        scheduler=scheduler,
        iter_idx=iter_idx,
        debate_idx=0
    )
    
    # Mimic the original implementation's result consolidation
    debate_dir = f"{config['experiment_dir']}/debate"
    os.makedirs(debate_dir, exist_ok=True)
    
    # Load the saved interim result file
    result_path = f"{debate_dir}/iteration_{iter_idx}_index_0.json"
    assert os.path.exists(result_path), f"Debate result file {result_path} does not exist. Use a smaller debate batch size."
    with open(result_path, 'r') as f:
        debate_results = json.load(f)
    
    # Save consolidated results
    save_debate_results(debate_results, config, iter_idx)
    
    # Shutdown scheduler
    await scheduler.shutdown()
    
    # Cleanup
    cleanup_start = time.time()
    cleanup_gpu()
    
    total_time = time.time() - start_time
    print(f"Debate completed in {total_time:.2f} seconds")
    
    return debate_results

async def scheduled_debate_processing(
    data: List[Dict[str, Any]],
    config: Dict[str, Any],
    scheduler: AdapterJobScheduler,
    iter_idx: int,
    debate_idx: int
) -> Dict[str, Any]:
    """
    Run debate processing using the adapter job scheduler.
    
    Args:
        data: List of examples to debate on
        config: Configuration dictionary
        scheduler: AdapterJobScheduler instance
        iter_idx: Iteration index
        debate_idx: Debate process index
        
    Returns:
        Dictionary containing debate results
    """
    start_time = time.time()
    
    num_agents = config['agents']
    num_rounds = config['rounds']
    temperature, top_p = config['temperature'], config['top_p']
    batch_size = config['batch_debate']
    diversity_prompt = config['diversity_prompt']

    # Split data into batches
    num_batches = (len(data) + batch_size - 1) // batch_size
    data_batches = []
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(data))
        data_batches.append(data[batch_start:batch_end])
    
    
    # Process all batches in parallel with progress tracking
    batch_tasks = []
    for batch_idx, batch_data in enumerate(data_batches):
        task = process_single_batch_with_retry(
            batch_data=batch_data,
            batch_idx=batch_idx,
            num_batches=num_batches,
            config=config,
            scheduler=scheduler,
            num_agents=num_agents,
            num_rounds=num_rounds,
            temperature=temperature,
            top_p=top_p,
            diversity_prompt=diversity_prompt
        )
        batch_tasks.append(task)
    
    # Wait for all batches to complete with progress tracking
    batch_results = []
    completed_batches = 0
    
    # Use as_completed to track progress
    for completed_task in asyncio.as_completed(batch_tasks):
        result = await completed_task
        batch_results.append(result)
        completed_batches += 1
    
    # Consolidate results from all batches
    debate = {}
    for batch_result in batch_results:
        debate.update(batch_result)
    
    # Save interim results with same naming as original implementation
    debate_dir = f"{config['experiment_dir']}/debate"
    os.makedirs(debate_dir, exist_ok=True)
    
    # Save with the original interim file naming format
    interim_result_path = f"{debate_dir}/iteration_{iter_idx}_index_{debate_idx}.json"
    with open(interim_result_path, "w") as f:
        json.dump(debate, f)

    total_time = time.time() - start_time

    return debate

async def process_single_batch_with_retry(
    batch_data: List[Dict[str, Any]],
    batch_idx: int,
    num_batches: int,
    config: Dict[str, Any],
    scheduler: AdapterJobScheduler,
    num_agents: int,
    num_rounds: int,
    temperature: float,
    top_p: float,
    diversity_prompt: bool,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Process a single batch with automatic retry and batch size reduction on OOM.
    
    This function ensures that OOM errors are handled gracefully by:
    1. Detecting OOM errors
    2. Splitting the batch into smaller sub-batches
    3. Processing ALL examples across multiple sub-batches
    4. Never generating dummy/fake responses
    
    Args:
        batch_data: List of examples in this batch
        batch_idx: Index of this batch
        num_batches: Total number of batches
        config: Configuration dictionary
        scheduler: AdapterJobScheduler instance
        num_agents: Number of agents
        num_rounds: Number of debate rounds
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        diversity_prompt: Whether to use diversity prompt
        max_retries: Maximum number of retry attempts with reduced batch size
        
    Returns:
        Dictionary containing debate results for this batch
    """
    original_batch_size = len(batch_data)
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # print(f" Batch {batch_idx + 1}/{num_batches} attempt {retry_count + 1} with {len(batch_data)} examples")
            
            # Try processing the full batch
            result = await process_single_batch(
                batch_data=batch_data,
                batch_idx=batch_idx,
                num_batches=num_batches,
                config=config,
                scheduler=scheduler,
                num_agents=num_agents,
                num_rounds=num_rounds,
                temperature=temperature,
                top_p=top_p,
                diversity_prompt=diversity_prompt
            )
            
            # If we get here, the batch succeeded
            # if retry_count > 0:
            #     print(f" Batch {batch_idx + 1} succeeded after {retry_count} retries")
                
            return result
            
        except RuntimeError as e:
            error_msg = str(e)
            
            # Check if this is an OOM-related error
            is_oom_error = any(oom_indicator in error_msg.lower() for oom_indicator in [
                'oom', 'out of memory', 'cuda out of memory', 'memory'
            ])
            
            if is_oom_error and retry_count < max_retries:
                retry_count += 1
                
                # Calculate sub-batch size (reduce by half, minimum 1)
                sub_batch_size = max(1, len(batch_data) // 2)
                
                print(f" OOM detected in batch {batch_idx + 1}, retry {retry_count}/{max_retries}")
                print(f"   Splitting batch of {len(batch_data)} examples into sub-batches of size {sub_batch_size}")
                
                # If we can't reduce further and still have multiple examples, force to 1
                if sub_batch_size == len(batch_data) and sub_batch_size > 1:
                    sub_batch_size = 1
                    print(f"   Forcing sub-batch size to 1 (processing examples individually)")
                
                # Force aggressive cleanup before retry
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Wait a moment for memory to be fully released
                await asyncio.sleep(2)
                
                # Split the batch into sub-batches and process each one
                print(f"   Processing {len(batch_data)} examples in sub-batches of size {sub_batch_size}")
                
                combined_results = {}
                num_sub_batches = (len(batch_data) + sub_batch_size - 1) // sub_batch_size
                
                for sub_batch_idx in range(num_sub_batches):
                    start_idx = sub_batch_idx * sub_batch_size
                    end_idx = min(start_idx + sub_batch_size, len(batch_data))
                    sub_batch_data = batch_data[start_idx:end_idx]
                    
                    print(f"   Processing sub-batch {sub_batch_idx + 1}/{num_sub_batches} with {len(sub_batch_data)} examples")
                    
                    try:
                        sub_result = await process_single_batch(
                            batch_data=sub_batch_data,
                            batch_idx=batch_idx,  # Keep original batch index for logging
                            num_batches=num_batches,
                            config=config,
                            scheduler=scheduler,
                            num_agents=num_agents,
                            num_rounds=num_rounds,
                            temperature=temperature,
                            top_p=top_p,
                            diversity_prompt=diversity_prompt
                        )
                        
                        # Merge results from this sub-batch
                        combined_results.update(sub_result)
                        print(f"    Sub-batch {sub_batch_idx + 1}/{num_sub_batches} completed successfully")
                        
                        # Small delay between sub-batches to prevent memory buildup
                        if sub_batch_idx < num_sub_batches - 1:  # Don't delay after the last sub-batch
                            await asyncio.sleep(1)
                        
                    except Exception as sub_e:
                        print(f"    Sub-batch {sub_batch_idx + 1}/{num_sub_batches} failed: {sub_e}")
                        # If even the sub-batch fails, we need to go to the next retry level
                        raise sub_e
                
                print(f" Batch {batch_idx + 1} completed using {num_sub_batches} sub-batches")
                print(f"   Total examples processed: {len(combined_results)}/{original_batch_size}")
                
                return combined_results
                
            else:
                # Either not an OOM error, or we've exhausted retries
                if is_oom_error:
                    print(f" CRITICAL: OOM error persists after {max_retries} retries for batch {batch_idx + 1}")
                    print(f"   Original batch size: {original_batch_size}")
                    print(f"   This indicates severe memory constraints that cannot be resolved by batch splitting.")
                    print(f"   Consider:")
                    print(f"   1. Using a smaller model (e.g., qwen2b instead of qwen7b)")
                    print(f"   2. Reducing batch_debate size in config (current: {config.get('batch_debate', 'unknown')})")
                    print(f"   3. Using fewer agents (current: {num_agents})")
                    print(f"   4. Enabling quantization (current: {config.get('use_quantization', False)})")
                    print(f"   5. Reducing number of debate rounds (current: {num_rounds})")
                    
                # Re-raise the error for non-OOM errors or exhausted retries
                raise e
    
    # This should never be reached, but just in case
    raise RuntimeError(f"Unexpected error: exceeded retry limit for batch {batch_idx + 1}")

async def process_single_batch(
    batch_data: List[Dict[str, Any]],
    batch_idx: int,
    num_batches: int,
    config: Dict[str, Any],
    scheduler: AdapterJobScheduler,
    num_agents: int,
    num_rounds: int,
    temperature: float,
    top_p: float,
    diversity_prompt: bool
) -> Dict[str, Any]:
    """
    Process a single batch of examples through debate rounds with optimized parallelization.
    
    Args:
        batch_data: List of examples in this batch
        batch_idx: Index of this batch
        num_batches: Total number of batches
        config: Configuration dictionary
        scheduler: AdapterJobScheduler instance
        num_agents: Number of agents
        num_rounds: Number of debate rounds
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        diversity_prompt: Whether to use diversity prompt
        
    Returns:
        Dictionary containing debate results for this batch
    """
    num_examples_in_batch = len(batch_data)
    batch_start_time = time.time()
    # print(f"Processing batch {batch_idx + 1}/{num_batches} with {num_examples_in_batch} examples")
    
    batch_questions = [example['question'] for example in batch_data]
    batch_answers = [example['answer'] for example in batch_data]
    
    # Initialize contexts for all examples and all agents
    # Structure: [example_idx][agent_idx][message_idx]
    batch_agent_contexts = [
        [
            [{"role": "user", "content": construct_question_prompt(q, config['dataset'])}]
            for _ in range(num_agents)
        ] 
        for q in batch_questions
    ]

    # Track completion state for each agent-round combination
    completion_state = {}  # (agent_idx, round_idx) -> bool
    agent_round_results = {}  # (agent_idx, round_idx) -> List[completions]
    
    # Initialize completion tracking
    for agent_idx in range(num_agents):
        for round_idx in range(num_rounds):
            completion_state[(agent_idx, round_idx)] = False
            agent_round_results[(agent_idx, round_idx)] = None
    
    # Create all tasks upfront for maximum parallelization
    pending_tasks = {}  # task -> (agent_idx, round_idx)
    
    # Add progress tracking
    try:
        from tqdm.asyncio import tqdm
    except ImportError:
        from tqdm import tqdm
    total_tasks = num_agents * num_rounds
    
    # print(f"Creating {total_tasks} parallel agent tasks for batch {batch_idx + 1}")
    
    # Create initial tasks (round 0 for all agents)
    for agent_idx in range(num_agents):
        # Round 0: Initial responses (no dependencies)
        contexts_for_agent = [ctx[agent_idx] for ctx in batch_agent_contexts]
        
        coro = scheduler.schedule_batch_generate(
            contexts=contexts_for_agent,
            agent_id=agent_idx,
            temperature=temperature,
            top_p=top_p,
            round_idx=0
        )
        task = asyncio.create_task(coro)
        pending_tasks[task] = (agent_idx, 0)
    
    # Process tasks as they complete
    completed_tasks = 0
    pbar = tqdm(total=total_tasks, desc=f"Batch {batch_idx + 1}/{num_batches}")
    
    while pending_tasks:
        # Wait for any task to complete
        done, pending = await asyncio.wait(pending_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
        
        for task in done:
            agent_idx, round_idx = pending_tasks[task]
            del pending_tasks[task]
            
            try:
                # Get the results
                completions = await task
                agent_round_results[(agent_idx, round_idx)] = completions
                completion_state[(agent_idx, round_idx)] = True
                completed_tasks += 1
                
                # Update contexts with new responses
                for q_idx, completion in enumerate(completions):
                    assistant_message = construct_assistant_message(completion)
                    batch_agent_contexts[q_idx][agent_idx].append(assistant_message)
                
                pbar.set_description(f"Batch {batch_idx + 1}/{num_batches} | Agent {agent_idx} Round {round_idx + 1} [COMPLETE]")
                pbar.update(1)
                
                # Check if this agent can start the next round
                next_round = round_idx + 1
                if next_round < num_rounds and not completion_state[(agent_idx, next_round)]:
                    # Check if we have enough responses from other agents for context
                    other_agents_ready = True
                    if next_round > 0:  # For rounds > 0, we need other agents' previous responses
                        for other_agent_idx in range(num_agents):
                            if other_agent_idx != agent_idx:
                                if not completion_state[(other_agent_idx, next_round - 1)]:
                                    other_agents_ready = False
                                    break
                    
                    if other_agents_ready:
                        # Prepare contexts for this agent's next round
                        if next_round > 0:
                            # Add other agents' responses to context
                            agent_contexts_other_batch = []
                            for q_idx in range(num_examples_in_batch):
                                # Get responses from other agents for this example
                                agent_contexts_other = batch_agent_contexts[q_idx][:agent_idx] + batch_agent_contexts[q_idx][agent_idx+1:]
                                random.shuffle(agent_contexts_other)
                                agent_contexts_other = agent_contexts_other[:5]  # Take up to 5 agents
                                agent_contexts_other_batch.append(agent_contexts_other)

                            # Create messages with feedback for each example
                            for q_idx in range(num_examples_in_batch):
                                message = construct_message(
                                    agent_contexts_other_batch[q_idx],
                                    batch_questions[q_idx],
                                    config['dataset'],
                                    None,  # No summary
                                    2 * next_round - 1,
                                    diversity_prompt
                                )
                                batch_agent_contexts[q_idx][agent_idx].append(message)
                        
                        # Collect all contexts for this agent across all examples
                        contexts_for_agent = [ctx[agent_idx] for ctx in batch_agent_contexts]
                        
                        # Create task for next round
                        next_coro = scheduler.schedule_batch_generate(
                            contexts=contexts_for_agent,
                            agent_id=agent_idx,
                            temperature=temperature,
                            top_p=top_p,
                            round_idx=next_round
                        )
                        next_task = asyncio.create_task(next_coro)
                        pending_tasks[next_task] = (agent_idx, next_round)
                
                # Check if any other agents can now start their next rounds
                # (this handles the case where this agent's completion unblocks others)
                for other_agent_idx in range(num_agents):
                    if other_agent_idx == agent_idx:
                        continue
                        
                    for check_round in range(num_rounds):
                        if completion_state[(other_agent_idx, check_round)]:
                            continue  # Already completed
                        
                        # Check if this agent is already in pending tasks
                        already_pending = any(agent_round == (other_agent_idx, check_round) 
                                            for agent_round in pending_tasks.values())
                        if already_pending:
                            continue
                        
                        # Check if this agent can start this round
                        can_start = True
                        if check_round > 0:  # For rounds > 0, need previous round completions
                            # Ensure the agent itself completed its previous round
                            if not completion_state[(other_agent_idx, check_round - 1)]:
                                can_start = False
                            else:
                                # Ensure all other agents completed their previous rounds
                                for dep_agent_idx in range(num_agents):
                                    if dep_agent_idx != other_agent_idx:
                                        if not completion_state[(dep_agent_idx, check_round - 1)]:
                                            can_start = False
                                            break
                        
                        if can_start:
                            # Prepare contexts for this round
                            if check_round > 0:
                                # Add other agents' responses to context
                                agent_contexts_other_batch = []
                                for q_idx in range(num_examples_in_batch):
                                    # Get responses from other agents for this example
                                    agent_contexts_other = batch_agent_contexts[q_idx][:other_agent_idx] + batch_agent_contexts[q_idx][other_agent_idx+1:]
                                    random.shuffle(agent_contexts_other)
                                    agent_contexts_other = agent_contexts_other[:5]  # Take up to 5 agents
                                    agent_contexts_other_batch.append(agent_contexts_other)

                                # Create messages with feedback for each example
                                for q_idx in range(num_examples_in_batch):
                                    message = construct_message(
                                        agent_contexts_other_batch[q_idx],
                                        batch_questions[q_idx],
                                        config['dataset'],
                                        None,  # No summary
                                        2 * check_round - 1,
                                        diversity_prompt
                                    )
                                    batch_agent_contexts[q_idx][other_agent_idx].append(message)
                            
                            # Collect all contexts for this agent across all examples
                            contexts_for_agent = [ctx[other_agent_idx] for ctx in batch_agent_contexts]
                            
                            # Create task for this round
                            new_coro = scheduler.schedule_batch_generate(
                                contexts=contexts_for_agent,
                                agent_id=other_agent_idx,
                                temperature=temperature,
                                top_p=top_p,
                                round_idx=check_round
                            )
                            new_task = asyncio.create_task(new_coro)
                            pending_tasks[new_task] = (other_agent_idx, check_round)
                            break  # Only start one round per agent per iteration
                
            except Exception as e:
                error_msg = str(e)
                print(f" Error in agent {agent_idx} round {round_idx}: {error_msg}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                
                # Check if this is an OOM error that we can potentially recover from
                is_oom_error = any(oom_indicator in error_msg.lower() for oom_indicator in [
                    'cuda out of memory', 'out of memory', 'oom', 'memory'
                ])
                
                if is_oom_error:
                    print(f" CRITICAL: OOM detected for Agent {agent_idx} Round {round_idx}")
                    print(f"   This indicates insufficient GPU memory for the current batch size.")
                    print(f"   Current batch size: {len(batch_agent_contexts)}")
                    print(f"   REFUSING to generate dummy responses - this would corrupt the debate!")
                    
                    # Force aggressive cleanup
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    # Print memory status
                    if torch.cuda.is_available():
                        for gpu_id in range(torch.cuda.device_count()):
                            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                            total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                            print(f"   GPU {gpu_id}: {allocated:.2f}/{total:.2f} GB allocated")
                    
                    # CRITICAL: Do not generate dummy responses - fail the entire batch
                    raise RuntimeError(f"OOM error in Agent {agent_idx} Round {round_idx}. "
                                     f"Batch processing failed due to insufficient GPU memory. "
                                     f"Consider reducing batch_debate size in config or using smaller models.")
                else:
                    # For non-OOM errors, also fail rather than generate dummy data
                    print(f" CRITICAL: Non-OOM error for Agent {agent_idx} Round {round_idx}")
                    print(f"   Error type: {type(e).__name__}")
                    print(f"   REFUSING to generate dummy responses - this would corrupt the debate!")
                    
                    # Re-raise the original exception to maintain error transparency
                    raise RuntimeError(f"Agent {agent_idx} Round {round_idx} failed with error: {error_msg}") from e
    
    pbar.close()

    # Process results for this batch
    result_start = time.time()
    # print("Processing debate results for batch", batch_idx + 1)
    
    batch_debate = {}
    for q_idx, (question, ground_truth) in enumerate(zip(batch_questions, batch_answers)):
        answers = []
        agent_contexts = batch_agent_contexts[q_idx]

        # Extract answers from each agent
        for agent_context in agent_contexts:
            answer = parse_answer(agent_context[-1]['content'], config['dataset'])
            if answer is not None:
                answers.append(answer)

        if len(answers) == 0:
            continue

        # Determine consensus answer
        consensus_answer = majority_vote(answers)

        # Score agents against consensus
        agent_answers = []
        for agent_context in agent_contexts:
            answer = parse_answer(agent_context[-1]['content'], config['dataset'])
            agent_answers.append(1 if grade_answer(answer, consensus_answer) else 0)

        # Store results
        batch_debate[question] = {
            'context': agent_contexts,
            'consensus_answer': consensus_answer,
            'agent_answers': agent_answers,
            'ground_truth': ground_truth,
            'answer_cot': batch_data[q_idx]['answer_cot'],
            'split': batch_data[q_idx]['split'],
        }

    # print(f"Result processing took {time.time() - result_start:.2f} seconds")
    # print(f"Batch {batch_idx + 1} completed in {time.time() - batch_start_time:.2f} seconds")
    
    return batch_debate
