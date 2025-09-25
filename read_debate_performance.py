#!/usr/bin/env python3
"""
Simple script to check initial debate performance of a specific debate file.
Automatically detects dataset type from filename.
"""

import json
import argparse
import numpy as np
import re
from pathlib import Path
from typing import Dict, Any, List
import sys
import os

from parser import parse_answer, grade_answer


def detect_dataset_from_filename(filename: str) -> str:
    """Detect dataset type from filename."""
    # Extract dataset from filename pattern: model_dataset_agents_rounds_seed_diversity_summarize.json
    # Examples: llama8b_gsm8k_3_2_0_False_False.json, qwen2b_math_3_2_0_False_False.json
    
    # Common dataset patterns in filenames
    dataset_patterns = {
        'gsm8k': r'gsm8k',
        'math': r'math',
        'arithmatic': r'arithmatic',
        'aime_amc': r'aime_amc',
        'amc_aime': r'amc_aime',
        'gpqa': r'gpqa'
    }
    
    for dataset, pattern in dataset_patterns.items():
        if re.search(pattern, filename, re.IGNORECASE):
            return dataset
    
    # Default fallback
    print(f"Warning: Could not detect dataset from filename '{filename}', defaulting to 'gsm8k'")
    return 'gsm8k'


def load_debate_file(file_path: str) -> Dict[str, Any]:
    """Load a debate JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Remove metrics entry if present
    return {k: v for k, v in data.items() if k != "metrics"}


def analyze_debate_performance(
    debate_data: Dict[str, Any],
    dataset: str = "gsm8k"
) -> Dict[str, Any]:
    """Analyze debate performance metrics."""
    
    train_examples = []
    test_examples = []
    
    for example_id, example in debate_data.items():
        if example_id == "metrics":
            continue
            
        ground_truth = example["ground_truth"]
        consensus_answer = example["consensus_answer"]
        contexts = example["context"]
        
        # Check consensus correctness
        consensus_correct = grade_answer(consensus_answer, ground_truth)
        
        # Analyze individual agent performance
        agent_accuracies = []             # final-round
        agent_init_accuracies = []        # first assistant reply
        agent_answers = []
        
        for agent_idx, context in enumerate(contexts):
            # Get the final response from each agent
            completion = context[-1]["content"]
            parsed_answer = parse_answer(completion, dataset=dataset)
            is_correct = grade_answer(parsed_answer, ground_truth)
            agent_accuracies.append(is_correct)

            # --- initial round ---
            init_correct = False
            try:
                # old format: list of messages
                first_reply = None
                for msg in context:
                    if msg.get('role') == 'assistant':
                        first_reply = msg['content']
                        break
                if first_reply:
                    init_parsed = parse_answer(first_reply, dataset=dataset)
                    init_correct = grade_answer(init_parsed, ground_truth)
            except Exception:
                pass
            agent_init_accuracies.append(init_correct)
        
        # Calculate metrics
        example_data = {
            "example_id": example_id,
            "ground_truth": ground_truth,
            "consensus_answer": consensus_answer,
            "consensus_correct": consensus_correct,
            "agent_accuracies": agent_accuracies,
            "agent_init_accuracies": agent_init_accuracies,
            "agent_answers": agent_answers,
            "avg_agent_accuracy": np.mean(agent_accuracies),
            "split": example["split"]
        }
        
        if example["split"] == "train":
            train_examples.append(example_data)
        else:
            test_examples.append(example_data)
    
    # ------------------------------------------------------------
    # Helper: compute split-level metrics WITHOUT double averaging
    # ------------------------------------------------------------
    def compute_overall_metrics(examples):
        """Return dict with consensus & average-agent accuracy.
        The average agent accuracy is computed as
            total_correct_tokens / total_attempts
        so every (agent, example) pair has equal weight.  Missing
        responses count as incorrect automatically because their
        entry is simply absent from `agent_accuracies`."""
        if not examples:
            return {
                "count": 0,
                "consensus_accuracy": 0.0,
                "avg_agent_accuracy": 0.0,
                "consensus_correct": 0,
                "consensus_incorrect": 0,
            }

        # Consensus statistics
        consensus_correct = sum(1 for ex in examples if ex["consensus_correct"])

        # Agent statistics – aggregate over *all* replies
        total_agent_correct = 0
        total_agent_responses = 0
        for ex in examples:
            total_agent_correct += sum(ex["agent_accuracies"])
            total_agent_responses += len(ex["agent_accuracies"])

        avg_agent_acc = (
            total_agent_correct / total_agent_responses
            if total_agent_responses else 0.0
        )

        return {
            "count": len(examples),
            "consensus_accuracy": consensus_correct / len(examples),
            "avg_agent_accuracy": avg_agent_acc,
            "consensus_correct": consensus_correct,
            "consensus_incorrect": len(examples) - consensus_correct,
        }

    train_metrics = compute_overall_metrics(train_examples)
    test_metrics  = compute_overall_metrics(test_examples)
    
    # Agent-specific analysis
    num_agents = len(train_examples[0]["agent_accuracies"]) if train_examples else (len(test_examples[0]["agent_accuracies"]) if test_examples else 0)
    agent_performance = {}
    
    for agent_idx in range(num_agents):
        # Helper to safely pull accuracy, counting missing replies as 0
        def _safe(idx, lst):
            return lst[idx] if idx < len(lst) else 0

        # ----- FINAL ROUND ACCURACY (missing reply = wrong) -----
        train_agent_correct = sum(_safe(agent_idx, ex["agent_accuracies"]) for ex in train_examples)
        test_agent_correct  = sum(_safe(agent_idx, ex["agent_accuracies"]) for ex in test_examples)

        # Round-1 (final reply)
        train_agent_round1 = train_agent_correct / len(train_examples) if train_examples else 0.0
        test_agent_round1  = test_agent_correct  / len(test_examples)  if test_examples  else 0.0

        # ----- INITIAL ROUND ACCURACY -----
        train_agent_init_correct = sum(_safe(agent_idx, ex["agent_init_accuracies"]) for ex in train_examples)
        test_agent_init_correct  = sum(_safe(agent_idx, ex["agent_init_accuracies"]) for ex in test_examples)

        # Round-0 (first reply)
        train_agent_round0 = train_agent_init_correct / len(train_examples) if train_examples else 0.0
        test_agent_round0  = test_agent_init_correct  / len(test_examples)  if test_examples  else 0.0
        
        agent_performance[f"agent_{agent_idx}"] = {
            "train_round1_accuracy": train_agent_round1,
            "test_round1_accuracy":  test_agent_round1,
            "train_round0_accuracy": train_agent_round0,
            "test_round0_accuracy":  test_agent_round0,
        }
    
    return {
        "train": train_metrics,
        "test": test_metrics,
        "agent_performance": agent_performance,
        "total_examples": len(train_examples) + len(test_examples)
    }


def main():
    parser = argparse.ArgumentParser(description="Check initial debate performance")
    parser.add_argument("debate_file", type=str, help="Path to debate JSON file")
    parser.add_argument("--dataset", type=str, help="Dataset type for parsing (auto-detected from filename if not specified)")
    args = parser.parse_args()
    
    # Auto-detect dataset from filename
    filename = os.path.basename(args.debate_file)
    detected_dataset = detect_dataset_from_filename(filename)
    
    # Use provided dataset or detected dataset
    dataset = args.dataset if args.dataset else detected_dataset
    
    # Load debate data
    print(f"Loading debate file: {args.debate_file}")
    print(f"Detected dataset: {detected_dataset}")
    if args.dataset and args.dataset != detected_dataset:
        print(f"Using specified dataset: {args.dataset} (overriding detected: {detected_dataset})")
    
    debate_data = load_debate_file(args.debate_file)
    print(f"Loaded {len(debate_data)} debate examples")
    
    # Analyze performance
    print(f"\nAnalyzing debate performance for dataset: {dataset}")
    metrics = analyze_debate_performance(debate_data, dataset)
    
    # Print results
    print("\n" + "="*60)
    print("INITIAL DEBATE PERFORMANCE ANALYSIS")
    print("="*60)
    
    if metrics['train']['count'] > 0:
        print(f"\nTRAIN SET ({metrics['train']['count']} examples):")
        print(f"  Consensus Accuracy: {metrics['train']['consensus_accuracy']:.3f} ({metrics['train']['consensus_correct']}/{metrics['train']['count']})")
        print(f"  Average Agent Accuracy: {metrics['train']['avg_agent_accuracy']:.3f}")
    else:
        print(f"\nTRAIN SET: No examples (test-only mode)")
    
    if metrics['test']['count'] > 0:
        print(f"\nTEST SET ({metrics['test']['count']} examples):")
        print(f"  Consensus Accuracy: {metrics['test']['consensus_accuracy']:.3f} ({metrics['test']['consensus_correct']}/{metrics['test']['count']})")
        print(f"  Average Agent Accuracy: {metrics['test']['avg_agent_accuracy']:.3f}")
    else:
        print(f"\nTEST SET: No examples (train-only mode)")
    
    print(f"\nINDIVIDUAL AGENT PERFORMANCE (Round-1 vs Round-0):")
    for agent_id, perf in metrics['agent_performance'].items():
        print(f"  {agent_id}:")
        # Round-1
        print(f"    Round-1 Train Accuracy : {perf.get('train_round1_accuracy',0):.3f}")
        print(f"    Round-1 Test  Accuracy : {perf.get('test_round1_accuracy',0):.3f}")

        # Round-0
        if 'train_round0_accuracy' in perf:
            print(f"    Round-0 Train Accuracy : {perf['train_round0_accuracy']:.3f}")
            print(f"    Round-0 Test  Accuracy : {perf['test_round0_accuracy']:.3f}")
    
    print(f"\nTOTAL EXAMPLES: {metrics['total_examples']}")

if __name__ == "__main__":
    main() 